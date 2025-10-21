GENERALFLAGS=-I ./include

CC=gcc
CFLAGS=-g -Wall $(GENERALFLAGS)
# LDFLAGS=-g -Wall
LDLIBS=-lpthread -lm -ljpeg
DEBUG=y
ifeq ($(DEBUG),y)
	DEBUGFLAGS=-D DEBUG
endif

NVCC=nvcc
NVCCFLAGS=-O2 -arch=sm_89 $(DEBUGFLAGS) $(GENERALFLAGS)

CFILES=$(wildcard src/*.c)
CUDAFILES=$(wildcard src/*.cu)
# COBJ=$(subst src/,build/,$(subst .c,.o,$(CFILES)))
# CUDAOBJ=$(subst src/,build/,$(subst .cu,.o,$(CUDAFILES)))
COBJ=$(patsubst src/%.c,build/%.o,$(CFILES))
CUDAOBJ=$(patsubst src/%.cu,build/%.o,$(CUDAFILES))

EXECNAME=bin/prog

build: $(COBJ) $(CUDAOBJ)
	 $(NVCC) $(COBJ) $(CUDAOBJ) -o $(EXECNAME) $(LDLIBS)

run: build
	./$(EXECNAME)
build/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -c

build/%.o: src/%.c
	$(CC) $(CFLAGS) $< -o $@ -c

.PHONY: clean all
clean:
	-rm build/* bin/*
