# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds

# Include any dependencies generated for this target.
include CMakeFiles/multi.x.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/multi.x.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/multi.x.dir/flags.make

CMakeFiles/multi.x.dir/main.cpp.o: CMakeFiles/multi.x.dir/flags.make
CMakeFiles/multi.x.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/multi.x.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/multi.x.dir/main.cpp.o -c /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/main.cpp

CMakeFiles/multi.x.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/multi.x.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/main.cpp > CMakeFiles/multi.x.dir/main.cpp.i

CMakeFiles/multi.x.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/multi.x.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/main.cpp -o CMakeFiles/multi.x.dir/main.cpp.s

CMakeFiles/multi.x.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/multi.x.dir/main.cpp.o.requires

CMakeFiles/multi.x.dir/main.cpp.o.provides: CMakeFiles/multi.x.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/multi.x.dir/build.make CMakeFiles/multi.x.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/multi.x.dir/main.cpp.o.provides

CMakeFiles/multi.x.dir/main.cpp.o.provides.build: CMakeFiles/multi.x.dir/main.cpp.o

# Object files for target multi.x
multi_x_OBJECTS = \
"CMakeFiles/multi.x.dir/main.cpp.o"

# External object files for target multi.x
multi_x_EXTERNAL_OBJECTS =

multi.x: CMakeFiles/multi.x.dir/main.cpp.o
multi.x: libmulti.a
multi.x: /usr/lib/libOpenCL.so
multi.x: CMakeFiles/multi.x.dir/build.make
multi.x: CMakeFiles/multi.x.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable multi.x"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/multi.x.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/multi.x.dir/build: multi.x
.PHONY : CMakeFiles/multi.x.dir/build

CMakeFiles/multi.x.dir/requires: CMakeFiles/multi.x.dir/main.cpp.o.requires
.PHONY : CMakeFiles/multi.x.dir/requires

CMakeFiles/multi.x.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/multi.x.dir/cmake_clean.cmake
.PHONY : CMakeFiles/multi.x.dir/clean

CMakeFiles/multi.x.dir/depend:
	cd /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds/CMakeFiles/multi.x.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/multi.x.dir/depend

