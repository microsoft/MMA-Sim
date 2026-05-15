# AMD ROCm Configuration
HIPCC      := /opt/rocm/bin/hipcc
HIPFLAGS   := -shared -fPIC
AMDTARGETS := cdna1.so cdna2.so cdna3.so

# AMD GPU targets
cdna1.so: mfma_cdna1.hip
	$(HIPCC) $(HIPFLAGS) --offload-arch=gfx908 -o $@ $^

cdna2.so: mfma_cdna2.hip mfma_cdna1.hip
	$(HIPCC) $(HIPFLAGS) --offload-arch=gfx90a -o $@ $^

cdna3.so: mfma_cdna3.hip
	$(HIPCC) $(HIPFLAGS) --offload-arch=gfx942 -o $@ $^

# Build all targets
all: $(AMDTARGETS)

# Clean up build artifacts
clean:
	rm -f $(AMDTARGETS)

.PHONY: all clean
