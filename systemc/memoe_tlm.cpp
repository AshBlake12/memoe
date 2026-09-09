// tlm model of the tiered expert path.
//
// we have the closed form and we have the runtime, but the closed form was
// calibrated against the runtime's baseline so the two aren't independent.
// this is a third crack at the same thing that shares no code with either.
// if the crossover turns up here as well then it's a property of the problem,
// not of our arithmetic.
//
// build with make. one point:
//   ./memoe_tlm --batch 4096 --residency 0.3
// whole sweep: scripts/run_systemc.py

#include <systemc.h>
#include <tlm.h>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace sc_core;
using namespace tlm;

// defaults are olmoe on ramesh, so these line up with the runtime numbers
struct Params {
    int    layers        = 16;          // MoE layers
    int    experts       = 64;          // experts per layer
    int    topk          = 8;           // experts activated per token
    double expert_bytes  = 12.58e6;     // 3 * d_model * d_ff * 2 bytes
    double expert_flops  = 12.58e6;     // per token, 2 * 3 * d_model * d_ff
    double gpu_flops     = 33.7e12;     // calibrated from the resident baseline
    double tier_bw       = 40.8e9;      // measured PCIe 5.0 x16 plateau
    double tier_latency  = 300e-9;      // modelled CXL: device + controller + link
    double hbm_bw        = 3120e9;      // effective HBM, resident experts
    double residency     = 0.30;        // share of experts kept on the GPU
    int    depth         = 1;           // prefetch lookahead, in layers
    int    batch         = 4096;        // tokens
};

static Params P;

// a memory tier. delay is latency + bytes/bandwidth.
// latency gets charged once per transaction, not per byte, which is the whole
// reason big transfers win. move 12 MB and latency is noise. move 4 KB and it's
// most of your time. same thing the granularity table shows.
SC_MODULE(MemoryTier) {
    tlm_utils::simple_target_socket<MemoryTier> socket;

    double bw;          // bytes per second
    double latency;     // seconds, per transaction
    double bytes_served = 0;
    long   transactions = 0;

    MemoryTier(sc_module_name n, double bw_, double lat_)
        : sc_module(n), socket("socket"), bw(bw_), latency(lat_) {
        socket.register_b_transport(this, &MemoryTier::b_transport);
    }

    void b_transport(tlm_generic_payload &trans, sc_time &delay) {
        const double bytes = static_cast<double>(trans.get_data_length());
        bytes_served += bytes;
        transactions += 1;
        delay += sc_time(latency + bytes / bw, SC_SEC);
        trans.set_response_status(TLM_OK_RESPONSE);
    }
};

//  the copy engine. One transaction per layer, carrying every
//  non-resident expert of that layer, issued `depth` layers ahead of the
//  compute engine. It blocks when the staging ring is full, which is
//  what bounds staging memory to depth+1 slots.
SC_MODULE(CopyEngine) {
    tlm_utils::simple_initiator_socket<CopyEngine> socket;

    sc_event  fetch_done;       // a layer has landed
    sc_event  slot_freed;       // compute finished with a layer
    int       fetched  = -1;    // highest layer index resident in staging
    int       consumed = -1;    // highest layer index compute has finished
    double    bytes_per_layer = 0;

    SC_HAS_PROCESS(CopyEngine);
    CopyEngine(sc_module_name n) : sc_module(n), socket("socket") {
        const double offloaded = (1.0 - P.residency) * P.experts;
        bytes_per_layer = offloaded * P.expert_bytes;
        SC_THREAD(run);
    }

    void run() {
        for (int L = 0; L < P.layers; ++L) {
            // wait until the slot this layer will occupy has been released.
            while (L - consumed > P.depth + 1) wait(slot_freed);

            if (bytes_per_layer > 0) {
                tlm_generic_payload trans;
                sc_time delay = SC_ZERO_TIME;
                trans.set_command(TLM_READ_COMMAND);
                trans.set_address(static_cast<sc_dt::uint64>(L) * 0x1000000);
                trans.set_data_length(
                    static_cast<unsigned int>(bytes_per_layer));
                trans.set_streaming_width(
                    static_cast<unsigned int>(bytes_per_layer));
                trans.set_data_ptr(nullptr);
                trans.set_dmi_allowed(false);
                trans.set_response_status(TLM_INCOMPLETE_RESPONSE);

                socket->b_transport(trans, delay);
                wait(delay);
            }
            fetched = L;
            fetch_done.notify(SC_ZERO_TIME);
        }
    }
};

//  the compute engine. Consumes one layer at a time, stalling whenever
//  the copy engine has not yet delivered it. Stall time accumulated here
//  is the quantity the runtime reports as stall_fraction.
SC_MODULE(ComputeEngine) {
    CopyEngine *copy = nullptr;

    sc_time total_stall = SC_ZERO_TIME;
    sc_time total_compute = SC_ZERO_TIME;

    SC_HAS_PROCESS(ComputeEngine);
    ComputeEngine(sc_module_name n) : sc_module(n) { SC_THREAD(run); }

    void run() {
        const double flops_per_layer =
            static_cast<double>(P.batch) * P.topk * P.expert_flops;
        const sc_time t_compute(flops_per_layer / P.gpu_flops, SC_SEC);

        for (int L = 0; L < P.layers; ++L) {
            if (copy->fetched < L) {
                const sc_time t0 = sc_time_stamp();
                while (copy->fetched < L) wait(copy->fetch_done);
                total_stall += sc_time_stamp() - t0;
            }
            wait(t_compute);
            total_compute += t_compute;
            copy->consumed = L;
            copy->slot_freed.notify(SC_ZERO_TIME);
        }
        sc_stop();
    }
};

struct Result {
    double wall, compute, stall, stall_frac, tokens_per_s, staging_gb;
};

static Result run_one() {
    MemoryTier    cxl("cxl", P.tier_bw, P.tier_latency);
    CopyEngine    copy("copy");
    ComputeEngine comp("comp");
    comp.copy = &copy;
    copy.socket.bind(cxl.socket);

    sc_start();

    const double wall    = sc_time_stamp().to_seconds();
    const double stall   = comp.total_stall.to_seconds();
    const double compute = comp.total_compute.to_seconds();

    Result r;
    r.wall         = wall;
    r.compute      = compute;
    r.stall        = stall;
    r.stall_frac   = wall > 0 ? stall / wall : 0.0;
    r.tokens_per_s = wall > 0 ? P.batch / wall : 0.0;
    r.staging_gb   = (P.depth + 1) * copy.bytes_per_layer / 1e9;
    return r;
}

// closed form from the report, for comparison in the same row.
static double b_star() {
    const double h = P.residency;
    return (1.0 - h) * P.experts * P.gpu_flops / (P.tier_bw * P.topk * 1.0);
}

int sc_main(int argc, char *argv[]) {
    sc_report_handler::set_actions("/IEEE_Std_1666/deprecated",
                                   SC_DO_NOTHING);

    bool sweep = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sweep"))       sweep = true;
        if (!std::strcmp(argv[i], "--residency") && i + 1 < argc)
            P.residency = std::atof(argv[++i]);
        if (!std::strcmp(argv[i], "--batch") && i + 1 < argc)
            P.batch = std::atoi(argv[++i]);
        if (!std::strcmp(argv[i], "--depth") && i + 1 < argc)
            P.depth = std::atoi(argv[++i]);
        if (!std::strcmp(argv[i], "--bw") && i + 1 < argc)
            P.tier_bw = std::atof(argv[++i]) * 1e9;
    }

    if (sweep) {
        std::printf("batch,residency,depth,tier_bw_gbs,wall_s,compute_s,"
                    "stall_s,stall_fraction,tokens_per_second,staging_gb,"
                    "b_star\n");
    }

    Result r = run_one();
    std::printf("%d,%.2f,%d,%.2f,%.6e,%.6e,%.6e,%.6f,%.1f,%.3f,%.0f\n",
                P.batch, P.residency, P.depth, P.tier_bw / 1e9, r.wall,
                r.compute, r.stall, r.stall_frac, r.tokens_per_s,
                r.staging_gb, b_star());
    return 0;
}
