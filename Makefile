INCLUDEFLAGS=-I include

CC=clang
CFLAGS=-g -Wall $(INCLUDEFLAGS)
# -fsanitize=address
# LDFLAGS=-g -Wall
LDLIBS=-lpthread -lm -ljpeg
DEBUG=y
ifeq ($(DEBUG),y)
	DEBUGFLAGS=-D DEBUG
endif

NVCC=nvcc
NVCCFLAGS=-O2 -arch=sm_89 -g -G $(DEBUGFLAGS) $(INCLUDEFLAGS)
# -Xcompiler -fsanitize=address

CFILES=$(wildcard src/*.c)
CUDAFILES=$(wildcard src/*.cu)
# COBJ=$(subst src/,build/,$(subst .c,.o,$(CFILES)))
# CUDAOBJ=$(subst src/,build/,$(subst .cu,.o,$(CUDAFILES)))
OBJ=$(patsubst src/%.c,build/%.o,$(CFILES))
OBJ+=$(patsubst src/%.cu,build/%.o,$(CUDAFILES))
CDEP=$(patsubst src/%.c,dependencies/%.d,$(CFILES))
CUDADEP=$(patsubst src/%.cu,dependencies/%.du,$(CUDAFILES))
TESTS=$(wildcard tests/src/*)
TESTOBJ=$(patsubst tests/src/%.cu,tests/build/%.o,$(patsubst tests/src/%.cpp,tests/build/%.o,$(TESTS)))

EXECNAME=bin/prog

all: $(CDEP) $(CUDADEP) build

.PHONY: clean all check build
clean:
	-rm -r build/ bin/ dependencies/ tests/build/ tests/test tests/test_outputs

dependencies/%.d: src/%.c Makefile
	mkdir -p dependencies
	mkdir -p build
	echo -n "build/" >$@
	$(CC) $(CFLAGS) -M $< >>$@
	echo "\t$(CC) $(CFLAGS) $< -o build/$*.o -c" >>$@

dependencies/%.du: src/%.cu Makefile
	mkdir -p dependencies
	mkdir -p build
	echo -n "build/" >$@
	$(NVCC) $(NVCCFLAGS) -M $< >>$@
	echo "\t$(NVCC) $(NVCCFLAGS) $< -o build/$*.o -dc" >>$@

include $(CDEP) $(CUDADEP)

build: $(OBJ)
	mkdir -p bin
	 $(NVCC) $(OBJ) -o $(EXECNAME) $(LDLIBS)

run: build
	./$(EXECNAME)

CPPFLAGS=$(shell pkg-config gtest_main --cflags --libs) -I tests/include
LDLIBS+=$(shell pkg-config gtest_main --libs --cflags) -ljpeg -lgtest_main
tests/build/%.o:NVCCFLAGS+=-I src
tests/build/%.o:: tests/src/%.cu
	mkdir -p tests/build
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -dc $< -o $@
tests/build/%.o:: tests/src/%.cpp
	mkdir -p tests/build
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

tests/test: $(OBJ) $(TESTOBJ)
	echo $(TESTOBJ)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(LDLIBS) $(TESTOBJ) $(filter-out build/main.o, $(OBJ)) -o tests/test

check: tests/test
	mkdir -p tests/test_outputs
	cd tests && ./test
