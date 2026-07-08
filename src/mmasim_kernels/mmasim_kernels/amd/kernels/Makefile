# AMD ROCm Configuration
HIPCC      := /opt/rocm/bin/hipcc
HIPFLAGS   := -shared -fPIC
AMDTARGETS := cdna1 cdna2 cdna3
AMDSOFILES := $(AMDTARGETS:=.so)

.PHONY: all clean $(AMDTARGETS)
all: $(AMDSOFILES)
$(AMDTARGETS): %: %.so
clean:
	rm -f $(AMDSOFILES)

# AMD GPU targets
cdna1.so: mfma_cdna1.hip
	$(HIPCC) $(HIPFLAGS) --offload-arch=gfx908 -o $@ $^

cdna2.so: mfma_cdna2.hip mfma_cdna1.hip
	$(HIPCC) $(HIPFLAGS) --offload-arch=gfx90a -o $@ $^

cdna3.so: mfma_cdna3.hip
	$(HIPCC) $(HIPFLAGS) --offload-arch=gfx942 -o $@ $^
