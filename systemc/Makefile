# SystemC TLM-2.0 model of the tiered expert path.
#
#   make          build
#   make run      one point at the defaults
#   make clean
#
# Needs SystemC 2.3.x. On Debian/Ubuntu: apt-get install libsystemc-dev
# From source: https://github.com/accellera-official/systemc
# If SystemC is somewhere non-standard, set SYSTEMC_HOME.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall
TARGET    = memoe_tlm

ifdef SYSTEMC_HOME
  CXXFLAGS += -I$(SYSTEMC_HOME)/include
  LDFLAGS  += -L$(SYSTEMC_HOME)/lib-linux64 -Wl,-rpath,$(SYSTEMC_HOME)/lib-linux64
endif
LDLIBS += -lsystemc

all: $(TARGET)

$(TARGET): memoe_tlm.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

run: $(TARGET)
	./$(TARGET) --batch 4096 --residency 0.30

sweep: $(TARGET)
	cd .. && python3 scripts/run_systemc.py

clean:
	rm -f $(TARGET)

.PHONY: all run sweep clean
