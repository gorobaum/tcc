all: nvidia apple foxc amd

is_win=$(shell s=`uname -s`; if (echo $$s | grep CYGWIN > /dev/null) || (echo $$s | grep MINGW > /dev/null); then echo 1; fi)
is_64=$(shell s=`uname -m`; if (echo $$s | grep x86_64 > /dev/null); then echo 1; fi)

CFLAGS=-Wall -W -O2

EXE=""

ifeq ($(is_win), 1)
PATH_TO_NVIDIA_SDK:="c:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK"
PATH_TO_NVIDIA_INC:=$(PATH_TO_NVIDIA_SDK)/OpenCL/common/inc
PATH_TO_NVIDIA_LIB:=$(PATH_TO_NVIDIA_SDK)/OpenCL/common/lib/Win32

PATH_TO_AMD="c:/Program Files/ATI Stream"
PATH_TO_AMD_INC=$(PATH_TO_AMD)/include
PATH_TO_AMD_LIB=$(PATH_TO_AMD)/bin/x86

EXE=.exe
else


ifeq ($(is_64), 1)
PATH_TO_NVIDIA_SDK:=$(HOME)/NVIDIA_GPU_Computing_SDK
PATH_TO_NVIDIA_INC:=$(PATH_TO_NVIDIA_SDK)/OpenCL/common/inc
PATH_TO_NVIDIA_LIB:=$(PATH_TO_NVIDIA_SDK)/OpenCL/common/lib/Linux64

PATH_TO_AMD:=$(HOME)/ati-stream-sdk-v2.0-beta4-lnx64
PATH_TO_AMD_LIB=$(PATH_TO_AMD)/lib/x86_64
else
PATH_TO_NVIDIA_SDK:=$(HOME)/NVIDIA_GPU_Computing_SDK
PATH_TO_NVIDIA_INC:=$(PATH_TO_NVIDIA_SDK)/OpenCL/common/inc
PATH_TO_NVIDIA_LIB:=$(PATH_TO_NVIDIA_SDK)/OpenCL/common/lib/Linux32

PATH_TO_AMD:=$(HOME)/ati-stream-sdk-v2.0-beta4-lnx32
PATH_TO_AMD_LIB=$(PATH_TO_AMD)/lib/x86
endif

PATH_TO_AMD_INC=$(PATH_TO_AMD)/include

endif

PATH_TO_FOXC:=$(HOME)/foxc-install

EXE_NAME=$(patsubst %.cpp,%,$(wildcard *.cpp))

nvidia: $(patsubst %,%.nvidia,$(EXE_NAME))$(EXE)
apple: $(patsubst %,%.apple,$(EXE_NAME))$(EXE)
foxc: $(patsubst %,%.foxc,$(EXE_NAME))$(EXE)
amd: $(patsubst %,%.amd,$(EXE_NAME))$(EXE)


%.nvidia$(EXE):%.cpp
	g++ $(CFLAGS) -I$(PATH_TO_NVIDIA_INC) -L$(PATH_TO_NVIDIA_LIB) -o $@ $< -lOpenCL

%.apple$(EXE):%.cpp
	g++ $(CFLAGS) -framework opencl -o $@ $<

%.foxc$(EXE):%.cpp
	g++ $(CFLAGS) -I$(PATH_TO_FOXC)/include -L$(PATH_TO_FOXC)/lib -Wl,-rpath,$(PATH_TO_FOXC)/lib -o $@ $< -lOpenCL

%.amd$(EXE):%.cpp
	g++ $(CFLAGS) -I$(PATH_TO_AMD_INC) -L$(PATH_TO_AMD_LIB) -Wl,-rpath,$(PATH_TO_AMD_LIB)  -o $@ $< -lOpenCL

clean:
	rm -f *.nvidia$(EXE) *.apple$(EXE) *.foxc$(EXE) *.amd$(EXE)
