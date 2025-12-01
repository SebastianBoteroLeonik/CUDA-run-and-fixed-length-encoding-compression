INCLUDEFLAGS=-I include

CC=gcc
CFLAGS=-g -Wall $(INCLUDEFLAGS)
# -fsanitize=address
# LDFLAGS=-g -Wall
LDLIBS=-lpthread -lm
#-ljpeg
DEBUG=y
ifeq ($(DEBUG),y)
	DEBUGFLAGS=-D DEBUG
endif

NVCC=nvcc
NVCCFLAGS=-O2 -arch=sm_80 -g -G $(DEBUGFLAGS) $(INCLUDEFLAGS) --std c++17
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

EXECNAME=bin/frle

all: $(CDEP) $(CUDADEP) build

.PHONY: clean all check build
clean:
	-rm -r build/ bin/ dependencies/ tests/build/ tests/test tests/test_outputs
	cd lib && $(MAKE) clean

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
	 $(NVCC) $(OBJ) -o $(EXECNAME) $(LDLIBS) $(NVCCFLAGS)

run: build
	./$(EXECNAME)

# CPPFLAGS=$(shell pkg-config gtest_main --cflags --libs) 
# CPPFLAGS=-L./lib/usr/local/lib -I ./lib/usr/local/include -I tests/include -DGTEST_HAS_PTHREAD=1 
#-lgtest_main -lgtest
# LDLIBS+=$(shell pkg-config gtest_main --libs --cflags) -ljpeg -lgtest_main
# tests/build/%.o:NVCCFLAGS+=-I src
TESTFLAGS=-I tests/include -I lib/usr/local/include
GTESTLIB=lib/usr/local/lib/libgtest.a lib/usr/local/lib/libgtest_main.a

libs:
	cd lib && $(MAKE)

tests/build/%.o:: tests/src/%.cu libs
	mkdir -p tests/build
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(TESTFLAGS) -dc $< -o $@
tests/build/%.o:: tests/src/%.cpp libs
	mkdir -p tests/build
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(TESTFLAGS) -c $< -o $@

tests/test: $(OBJ) $(TESTOBJ) libs
	echo $(TESTOBJ)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(LDLIBS) $(TESTOBJ) $(TESTFLAGS) \
		$(filter-out build/main.o, $(OBJ)) \
		$(GTESTLIB) \
		-o tests/test

check: tests/test
	mkdir -p tests/test_outputs
	cd tests && ./test
