# NVIDIA CUDA Configuration
NVCC       := /usr/local/cuda/bin/nvcc
NVCCFLAGS  := -shared -Xcompiler -fPIC
NVTARGETS  := nv_volta.so nv_turing.so nv_ampere.so nv_adalovelace.so nv_hopper.so nv_blackwell.so nv_rtxblackwell.so

# AMD ROCm Configuration
HIPCC      := /opt/rocm/bin/hipcc
HIPFLAGS   := -shared -fPIC
AMDTARGETS := amd_cdna1.so amd_cdna2.so amd_cdna3.so

# Build all NVIDIA targets
all_nv: $(NVTARGETS)

# NVIDIA GPU targets
nv_volta.so: mma_sm70.cu mma.h
	$(NVCC) $(NVCCFLAGS) -arch=compute_70 -code=sm_70 -o $@ $^

nv_turing.so: mma_sm75.cu mma.h
	$(NVCC) $(NVCCFLAGS) -arch=compute_75 -code=sm_75 -o $@ $^

nv_ampere.so: mma_sm80.cu mma_sm75.cu mma.h
	$(NVCC) $(NVCCFLAGS) -arch=compute_80 -code=sm_80 -o $@ $^

nv_adalovelace.so: mma_sm89.cu mma_sm80.cu mma_sm75.cu mma.h
	$(NVCC) $(NVCCFLAGS) -arch=compute_89 -code=sm_89 -o $@ $^

nv_hopper.so: wgmma_sm90a.cu mma_sm90.cu mma_sm80.cu mma_sm75.cu wgmma.h mma.h
	$(NVCC) $(NVCCFLAGS) -arch=compute_90a -code=sm_90a -o $@ $^

nv_blackwell.so: tcgen05mma_sm100a.cu mma_sm80.cu mma_sm75.cu tcgen05mma.h mma.h
	$(NVCC) $(NVCCFLAGS) -arch=compute_100a -code=sm_100a -o $@ $^

nv_rtxblackwell.so: mma_sm120a.cu mma_sm89.cu mma_sm80.cu mma_sm75.cu mma.h
	$(NVCC) $(NVCCFLAGS) -arch=compute_120a -code=sm_120a -o $@ $^

# Build all AMD targets
all_amd: $(AMDTARGETS)

# AMD GPU targets
amd_cdna1.so: mfma_cdna1.hip mfma.h
	$(HIPCC) $(HIPFLAGS) --offload-arch=gfx908 -o $@ $^

amd_cdna2.so: mfma_cdna2.hip mfma_cdna1.hip mfma.h
	$(HIPCC) $(HIPFLAGS) --offload-arch=gfx90a -o $@ $^

amd_cdna3.so: mfma_cdna3.hip mfma.h
	$(HIPCC) $(HIPFLAGS) --offload-arch=gfx942 -o $@ $^

# Build all targets
all: all_nv all_amd

# Clean up build artifacts
clean:
	rm -f $(NVTARGETS) $(AMDTARGETS)

.PHONY: all all_nv all_amd clean
