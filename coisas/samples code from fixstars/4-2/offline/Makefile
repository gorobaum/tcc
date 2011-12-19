all: foxc

is_win=$(shell s=`uname -s`; if (echo $$s | grep CYGWIN > /dev/null) || (echo $$s | grep MINGW > /dev/null); then echo 1; fi)
is_64=$(shell s=`uname -m`; if (echo $$s | grep x86_64 > /dev/null); then echo 1; fi)

CFLAGS=-Wall -W -O2

EXE=""

ifeq ($(is_win), 1)
EXE=.exe
endif

PATH_TO_FOXC:=$(HOME)/foxc-install

EXE_NAME=$(patsubst %.cpp,%,$(wildcard *.cpp))

foxc: $(patsubst %,%.foxc,$(EXE_NAME))$(EXE)

%.foxc$(EXE):%.cpp
	$(PATH_TO_FOXC)/bin/foxc -o kernel.clbin kernel.cl
	g++ $(CFLAGS) -I$(PATH_TO_FOXC)/include -L$(PATH_TO_FOXC)/lib -Wl,-rpath,$(PATH_TO_FOXC)/lib -o $@ $< -lOpenCL

clean:
	rm -f *.foxc$(EXE) kernel.clbin
