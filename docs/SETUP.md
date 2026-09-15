# MEMoE on one machine: setup and verification

Everything here was built and run on **bhaskar** (NVIDIA A10 23 GB, PCIe 4.0,
48 cores, 125 GB RAM, openSUSE, gcc 7.5, Python 3.6 system) **without root**.
Each section gives the build, the command that proves it works, and the output
you should see. `scripts/run_a10.sh` wraps the verification commands, one per
component.

Python environments are managed with `uv` throughout. Nothing is installed
system-wide; user-local builds live under `~/local`.

| Component | What it provides | Location |
|---|---|---|
| Analysis pipeline | tables, figures, dashboard, 38 tests | repo `.venv` via `uv` |
| MEMoE-RT on OLMoE-1B-7B | tiered runtime, model fits on the A10 | `scripts/run_runtime.sh` |
| MEMoE-RT on DeepSeek-V2-Lite | model that does **not** fit on the A10 | `scripts/run_deepseek.sh` |
| DRAMSim3 | CXL and HBM tier bandwidth and latency | `~/DRAMsim3` |
| SystemC 2.3.4 | independent TLM model of the data path | `~/local/systemc` |
| gem5 v22.1 | channel scaling of the DDR4 tier | `~/local/src/gem5` |
| QEMU 9.1 + Linux 6.8 | Linux CXL stack, region commit | `~/qemu`, `~/linux-cxl`, `~/cxlvm` |

---

## 1. Analysis pipeline (CPU only)

```bash
cd ~/nebula/memoe-proj
uv sync --extra dev                       # creates .venv from uv.lock, with pytest
uv run python -m pytest tests -q          # expect: 38 passed
uv run bash scripts/reproduce.sh          # ~8 min, regenerates results/
```

Open `results/dashboard.html` in any browser. It is self-contained.

## 2. MEMoE-RT on the GPU

The project `.venv` is Python 3.14, for which no CUDA 12.1 torch wheel exists,
so the GPU scripts create a throwaway Python 3.11 environment with `uv run
--no-project`. The driver on bhaskar is CUDA 12.2, which runs cu121 wheels.

### DeepSeek-V2-Lite (31.4 GB of weights on a 23 GB card)

One-time: the checkpoint's remote code imports `flash_attn` unconditionally.

```bash
bash scripts/setup_dsmod.sh               # writes ~/dsmod with the import guarded
```

Run:

```bash
bash scripts/run_deepseek.sh --check
bash scripts/run_deepseek.sh --batch 16384 --residency 0.0 --reps 1 --out /tmp/memoe-a10/deepseek.csv
```

`--check` compares a 25%-resident run against a 0%-resident run on identical
input and prints `PASS`. It used to compare 50% against 0%, but 50% residency
peaks at 22.5 of 23 GB on the A10 and cuBLAS then fails with
`CUBLAS_STATUS_EXECUTION_FAILED`, which is an out-of-memory in disguise.

Expected at batch 4,096, full offload: 28.79 GB streamed, 8.33 GB peak VRAM,
about 3,300 tok/s. At batch 16,384 the committed figure is 7,241 tok/s in
15.1 GB.

### OLMoE-1B-7B (fits, so there is a real baseline)

```bash
bash scripts/run_runtime.sh --check
bash scripts/run_runtime.sh --batch 16384 --residency 1.0 0.0 --depth 1 --out-dir /tmp/memoe-a10/olmoe
```

Expected on the A10 at batch 16,384: fully resident 13,391 tok/s in 17.7 GB,
fully offloaded 12,239 tok/s in 6.4 GB, which is 91.4% of throughput for 2.75x
less GPU memory, with stall under 0.1%.

`--check` runs the unmodified model once, then the tiered model at 25% and at
100% residency, and prints three things:

- agreement with the unmodified model: 89.45% argmax, 99.41% of tiered top-1
  inside the reference top-5 on the A10 (92.8% argmax on Blackwell). The gap is
  bf16 rounding in MEMoE-RT's re-implemented MoE forward. It is identical at
  100%, 25% and 0% residency, and half of the random-token positions have a
  top-1/top-2 logit gap under 0.5, so argmax flips on rounding alone.
- offload invariance: 25% resident against 100% resident, max abs diff. This
  must be exactly 0; it is what proves the transfer path is correct.
- `PASS` when invariance is exact and top-5 containment is above 99%.

The committed `results/runtime_bench*.csv` are the Blackwell (PCIe 5.0)
numbers used in the report. Always pass `--out-dir` on bhaskar so an A10 run
does not overwrite them.

## 3. DRAMSim3: the tier itself

Already built at `~/DRAMsim3/build/dramsim3main`. To rebuild from scratch:

```bash
git clone https://github.com/umd-memsys/DRAMsim3.git ~/DRAMsim3
cd ~/DRAMsim3 && mkdir -p build && cd build && cmake .. && make -j16
```

Verify both tiers against the committed calibration:

```bash
bash scripts/run_a10.sh dramsim
```

Expected: CXL tier (DDR4-3200, 1 channel) **16.9 GB/s, 179.4 ns**; HBM tier
(HBM2, 8 channels) **63.0 GB/s, 61.4 ns**. Each run takes about 3 seconds.
Ignore DRAMSim3's own `average_bandwidth`: it divides by the full cycle budget
including idle cycles. `parse_dramsim.py` uses `average_interarrival` instead.

## 4. SystemC: independent model of the data path

SystemC is not packaged for openSUSE, so build 2.3.4 into `~/local`:

```bash
git clone --depth 1 --branch 2.3.4 https://github.com/accellera-official/systemc.git ~/local/src/systemc
cd ~/local/src/systemc && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local/systemc -DCMAKE_CXX_STANDARD=17 -DBUILD_SHARED_LIBS=ON
make -j16 && make install
```

Build and run the model:

```bash
export SYSTEMC_HOME=$HOME/local/systemc
make -C systemc
./systemc/memoe_tlm --batch 4626 --residency 0.3
uv run python scripts/run_systemc.py      # full sweep -> results/tables/15_systemc_sweep.csv
```

Expected single point (CSV columns batch, residency, depth, tier_bw_gbs,
wall_s, compute_s, stall_s, stall_fraction, tokens_per_second, staging_gb,
b_star):

```
4626,0.30,1,40.80,2.348513e-01,2.210377e-01,1.381363e-02,0.058819,19697.6,1.127,4625
```

The sweep regenerates `15_systemc_sweep.csv` byte-identical to the committed
file. Fixed along the way: CMake installs the library in `lib64`, the Makefile
only looked in `lib-linux64`, and `run_systemc.py` called `g++` directly and
ignored `SYSTEMC_HOME`. Both now go through the Makefile.

## 5. gem5: channel scaling

gem5 v24 needs gcc 10 or newer; bhaskar has gcc 7.5. **v22.1 is the newest
release that accepts gcc 7**, so use it rather than installing a compiler. Only
the `NULL` target is needed: the experiment is a traffic generator and memory
controllers, with no CPU model.

```bash
git clone --depth 1 --branch v22.1.0.0 https://github.com/gem5/gem5.git ~/local/src/gem5
cd ~/local/src/gem5
uv tool run --python 3.10 --from "scons==4.4.0" scons build/NULL/gem5.opt -j40 --ignore-style
```

Two traps, both of which fail with misleading messages:

- **SCons 4.11** breaks gem5 v22.1's configure checks. The symptom is
  `Did not find needed zlib compression library` even though zlib is installed;
  `build/NULL/gem5.build/scons_config.log` shows the check's source string split
  into single characters and passed as `-l(` `-l)` libraries. Pin SCons 4.4.
- **SCons under Python 3.11** fails in `arch/arm/fastmodel/SConscript` with
  `global flags not at the start of the expression`, because 3.11 made that regex
  form an error. Run SCons on Python 3.10.

`uv tool run` handles both without touching the system Python.

Run the sweep:

```bash
bash gem5/run_channels.sh                 # 1, 2, 4, 8 channels -> results/gem5/ch*/stats.txt
```

The config is `gem5/channel_scaling.py`: one linear read generator at period 0,
so the generator never paces the channels, through a CommMonitor into N
interleaved DDR4-2400 x8 `MemCtrl` channels, 64-byte reads, 10 ms simulated.
About 45 seconds for the whole sweep.

Measured on bhaskar:

```
channels          GB/s    ratio
1                13.86    1.000
2                27.73    2.001
4                55.13    3.978
8        bus saturated: panic: Packet queue system.mem_ctrls0.port-RespPacketQueue has grown beyond 128 packets
```

One channel reproduces the report's 13.86 GB/s exactly. Four channels give
3.978x against the report's 3.999x, linear to within 0.6%, and the four-channel
figure depends on exactly that. At eight channels the shared bus cannot drain
responses and gem5 v22.1 panics on its 128-packet sanity limit instead of
reporting a throughput, so this config does not reproduce the report's
eight-channel row.

Two gem5 constraints shaped the config: a traffic generator block may not exceed
the 64-byte cache line, and a request blocked at tick 0 trips a
`retryPktTick != 0` assertion, so the trace idles for 1,000 ticks first.

## 6. QEMU: the Linux CXL stack

QEMU 9.1 is built at `~/qemu`, the guest image is `~/cxlvm/noble.qcow2`
(Ubuntu 24.04, cloud-init user `ubuntu`, password `memoe`, with `cxl-cli`,
`ndctl`, `daxctl`, `numactl` preinstalled), and the guest kernel is Linux 6.8
at `~/linux-cxl` built with `CONFIG_CXL_REGION_INVALIDATION_TEST=y`.

Why the custom kernel: before committing a region the kernel must invalidate
CPU caches over the range, and it gates that on a check that returns false
whenever `X86_FEATURE_HYPERVISOR` is set, which is every virtual machine. On the
stock Ubuntu kernel everything enumerates and every decoder programs, then
commit fails with `Failed to synchronize CPU cache state`
(`results/qemu/cxl_qemu_evidence.txt`). The config option enables the
documented test bypass.

Boot (serial console; exit with `Ctrl-a x`):

```bash
bash ~/cxlvm/run-cxl.sh
```

From a second terminal, about 20 seconds later:

```bash
ssh -p 2222 ubuntu@localhost              # password memoe
uname -r                                  # 6.8.0
sudo cxl list -M                          # mem0, 2 GiB, host 0000:0d:00.0
sudo cxl create-region -m -t ram -d decoder0.0 -w 1 mem0
cat /sys/bus/cxl/devices/region0/commit   # 1
sudo dmesg | grep -i "bypassing cpu_cache"
```

Expected: `created 1 region`, `decode_state: commit`, resource `0x490000000`,
size 2 GiB, target `decoder2.0`, and the kernel logging the bypass. Regions do
not persist across boots; create it again after each boot.

QEMU's CXL device is functional emulation, so this step validates the kernel's
region management. Tier performance comes from DRAMSim3 and gem5 (sections 3
and 5).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `scripts/run_deepseek.sh: No such file` | not in repo root | `cd ~/nebula/memoe-proj` |
| `CUBLAS_STATUS_EXECUTION_FAILED` | GPU memory full | lower residency; check `nvidia-smi` for other jobs |
| SystemC `cannot find -lsystemc` | `SYSTEMC_HOME` unset | `export SYSTEMC_HOME=$HOME/local/systemc` |
| gem5 "zlib not found" | SCons too new | `scons==4.4.0` via `uv tool run` |
| `ssh -p 2222` refused | guest still booting | wait for the login prompt in the console |
| `region0/commit: No such file` | region not created this boot | run `cxl create-region` again |
