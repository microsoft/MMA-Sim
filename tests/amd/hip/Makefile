HIPCC := /opt/rocm/bin/hipcc
CXXFLAGS := -shared -fPIC

TARGETS := cdna1.so cdna2.so

all: $(TARGETS)

cdna1.so: cdna1.hip
	$(HIPCC) $(CXXFLAGS) --offload-arch=gfx908 -o $@ $^

cdna2.so: cdna1.hip cdna2.hip
	$(HIPCC) $(CXXFLAGS) --offload-arch=gfx90a -o $@ $^

clean:
	rm -f $(TARGETS)

.PHONY: all clean
