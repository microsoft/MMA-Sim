NVCC := /usr/local/cuda/bin/nvcc
CXXFLAGS := -shared -Xcompiler -fPIC

TARGETS := volta.so turing.so ampere.so adalovelace.so hopper.so

all: $(TARGETS)

volta.so: mma_sm70.cu
	$(NVCC) $(CXXFLAGS) -arch=compute_70 -code=sm_70 -o $@ $^

turing.so: mma_sm75.cu
	$(NVCC) $(CXXFLAGS) -arch=compute_75 -code=sm_75 -o $@ $^

ampere.so: mma_sm75.cu mma_sm80.cu
	$(NVCC) $(CXXFLAGS) -arch=compute_80 -code=sm_80 -o $@ $^

adalovelace.so: mma_sm75.cu mma_sm80.cu mma_sm89.cu mma_sm89_ptx87.cu
	$(NVCC) $(CXXFLAGS) -arch=compute_89 -code=sm_89 -o $@ $^

hopper.so: mma_sm75.cu mma_sm80.cu mma_sm90.cu
	$(NVCC) $(CXXFLAGS) -arch=compute_90a -code=sm_90a -o $@ $^

rtxblackwell.so: mma_sm75.cu mma_sm80.cu mma_sm89.cu mma_sm89_ptx87.cu
	$(NVCC) $(CXXFLAGS) -arch=compute_120 -code=sm_120 -o $@ $^

clean:
	rm -f $(TARGETS)

.PHONY: all clean
