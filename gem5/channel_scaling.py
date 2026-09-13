# gem5 channel-scaling run for the CXL capacity tier.
#
# A single linear traffic generator reads as fast as the memory system accepts
# (period 0, so the generator never paces the channels), through a CommMonitor
# and a bus, into N interleaved DDR4-2400 MemCtrl channels.
#
#   build/NULL/gem5.opt --outdir=results/gem5/ch4 gem5/channel_scaling.py --channels 4
#
# See gem5/run_channels.sh for the full 1/2/4/8 sweep.

import argparse

import m5
from m5.objects import *
from m5.util import addToPath

p = argparse.ArgumentParser()
p.add_argument("--gem5-configs", required=True,
               help="path to the gem5 source tree's configs/ directory")
p.add_argument("--channels", type=int, default=1)
p.add_argument("--mem-type", default="DDR4_2400_8x8")
p.add_argument("--intlv", type=int, default=128,
               help="channel interleave granularity, bytes")
p.add_argument("--bus-width", type=int, default=128,
               help="bus datapath width per port")
p.add_argument("--block", type=int, default=64,
               help="read size, bytes; gem5 caps it at the 64-byte cache line")
p.add_argument("--ms", type=float, default=10.0, help="simulated time, ms")
args = p.parse_args()

addToPath(args.gem5_configs)
from common import MemConfig

system = System(membus=IOXBar(width=args.bus_width))
system.clk_domain = SrcClockDomain(clock="2.0GHz",
                                   voltage_domain=VoltageDomain(voltage="1V"))
system.mem_ranges = [AddrRange("1GB")]
system.mmap_using_noreserve = True

mem = argparse.Namespace(mem_type=args.mem_type, mem_channels=args.channels,
                         mem_ranks=None, mem_size="1GB",
                         mem_channels_intlv=args.intlv,
                         external_memory_system=0, tlm_memory=0,
                         elastic_trace_en=0)
MemConfig.config_mem(mem, system)
for c in system.mem_ctrls:
    c.dram.null = True

system.tgen = PyTrafficGen()
system.monitor = CommMonitor()
system.tgen.port = system.monitor.cpu_side_port
system.monitor.mem_side_port = system.membus.cpu_side_ports
system.system_port = system.membus.cpu_side_ports

root = Root(full_system=False, system=system)
root.system.mem_mode = "timing"
m5.instantiate()

duration = int(args.ms * 1e9)          # ticks are ps
end = system.mem_ranges[0].end


def trace():
    # a request blocked at tick 0 trips gem5's retryPktTick != 0 assertion
    yield system.tgen.createIdle(1000)
    yield system.tgen.createLinear(duration, 0, end - args.block, args.block,
                                   0, 0, 100, 0)
    yield system.tgen.createExit(0)


system.tgen.start(trace())
ev = m5.simulate()
print(f"channels={args.channels} exit='{ev.getCause()}' tick={m5.curTick()}")
