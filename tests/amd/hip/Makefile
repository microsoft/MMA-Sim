HIPCC := /opt/rocm/bin/hipcc
CXXFLAGS := -shared -fPIC

TARGETS := cdna1.so cdna2.so cdna3.so

all: $(TARGETS)

cdna1.so: cdna1.hip
	$(HIPCC) $(CXXFLAGS) --offload-arch=gfx908 -o $@ $^

cdna2.so: cdna1.hip cdna2.hip
	$(HIPCC) $(CXXFLAGS) --offload-arch=gfx90a -o $@ $^

cdna3.so: cdna3.hip
	$(HIPCC) $(CXXFLAGS) --offload-arch=gfx942 -o $@ $^

clean:
	rm -f $(TARGETS)

.PHONY: all clean
