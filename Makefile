CC=gcc
CFLAGS=-g -Wall
LDFLAGS=-g -Wall
LDLIBS=-lpthread -lm -ljpeg
DEBUG=y
ifeq ($(DEBUG),y)
	DEBUGFLAGS=-D DEBUG
endif

NVCC=nvcc
NVCCFLAGS=-O2 -arch=sm_89 $(DEBUGFLAGS)

CFILES=$(wildcard *.c)
CUDAFILES=$(wildcard *.cu)
COBJ=$(subst .c,.o,$(CFILES))
CUDAOBJ=$(subst .cu,.o,$(CUDAFILES))

EXECNAME=prog

build: $(COBJ) $(CUDAOBJ)
	 $(NVCC) $(COBJ) $(CUDAOBJ) -o $(EXECNAME) $(LDLIBS)

run: build
	./$(EXECNAME)
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -c

.PHONY: clean all
clean:
	-rm $(CUDAOBJ) $(COBJ) $(EXECNAME)
