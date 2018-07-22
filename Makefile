# Environment configuration
CXX = g++
CUDA_ROOT ?= /usr/local/cuda
NVCC ?= $(CUDA_ROOT)/bin/nvcc -ccbin=$(CXX)
NVCCFLAGS = -O2 -arch=sm_20
# CXXFLAGS = -g -O0
CXXFLAGS = -O2 -g -gdb
CXXFLAGS += -I$(CUDA_ROOT)/include


# Where to place outputs
BINDIR = bin
BUILDDIR = build

# Where to find inputs
SRCDIR = $(PWD)
#SRCDIR = src


TARGETS = $(BINDIR)/main

# Find all the cu files in SRCDIR
CU_SRCS := $(wildcard $(SRCDIR)/*.cu) $(wildcard $(SRCDIR)/**/*.cu)
CU_OBJS = $(patsubst $(SRCDIR)/%.cu, $(BUILDDIR)/%.o, $(CU_SRCS))

# Find all the cpp files in SRCDIR
CPP_SRCS := $(wildcard $(SRCDIR)/*.cpp) $(wildcard $(SRCDIR)/**/*.cpp)
CPP_OBJS = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(CPP_SRCS))

all: $(TARGETS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	mkdir -p `dirname $@`
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	mkdir -p `dirname $@`
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(BINDIR)/main: $(CU_OBJS) $(CPP_OBJS)
	mkdir -p `dirname $@`
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -f $(TARGETS)
	rm -rf $(BUILDDIR)
