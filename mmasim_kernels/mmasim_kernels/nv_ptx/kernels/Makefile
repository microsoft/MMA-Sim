# NVIDIA CUDA Configuration
NVCC       := /usr/local/cuda/bin/nvcc
NVCCFLAGS  := -shared -Xcompiler -fPIC
NVTARGETS  := volta.so turing.so ampere.so ada_lovelace.so hopper.so blackwell.so rtx_blackwell.so

# NVIDIA GPU targets
volta.so: mma_sm70.cu
	$(NVCC) $(NVCCFLAGS) -arch=compute_70 -code=sm_70 -o $@ $^

turing.so: mma_sm75.cu
	$(NVCC) $(NVCCFLAGS) -arch=compute_75 -code=sm_75 -o $@ $^

ampere.so: mma_sm80.cu mma_sm75.cu
	$(NVCC) $(NVCCFLAGS) -arch=compute_80 -code=sm_80 -o $@ $^

ada_lovelace.so: mma_sm89.cu mma_sm80.cu mma_sm75.cu
	$(NVCC) $(NVCCFLAGS) -arch=compute_89 -code=sm_89 -o $@ $^

hopper.so: wgmma_sm90a.cu mma_sm90.cu mma_sm80.cu mma_sm75.cu
	$(NVCC) $(NVCCFLAGS) -arch=compute_90a -code=sm_90a -o $@ $^

blackwell.so: tcgen05mma_sm100a.cu mma_sm80.cu mma_sm75.cu
	$(NVCC) $(NVCCFLAGS) -arch=compute_100a -code=sm_100a -o $@ $^

rtx_blackwell.so: mma_sm120a.cu mma_sm89.cu mma_sm80.cu mma_sm75.cu
	$(NVCC) $(NVCCFLAGS) -arch=compute_120a -code=sm_120a -o $@ $^

# Build all targets
all: $(NVTARGETS)

# Clean up build artifacts
clean:
	rm -f $(NVTARGETS)

.PHONY: all clean
