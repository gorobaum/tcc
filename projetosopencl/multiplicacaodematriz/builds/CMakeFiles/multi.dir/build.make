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
include CMakeFiles/multi.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/multi.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/multi.dir/flags.make

CMakeFiles/multi.dir/cll.cpp.o: CMakeFiles/multi.dir/flags.make
CMakeFiles/multi.dir/cll.cpp.o: ../cll.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/multi.dir/cll.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/multi.dir/cll.cpp.o -c /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/cll.cpp

CMakeFiles/multi.dir/cll.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/multi.dir/cll.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/cll.cpp > CMakeFiles/multi.dir/cll.cpp.i

CMakeFiles/multi.dir/cll.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/multi.dir/cll.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/cll.cpp -o CMakeFiles/multi.dir/cll.cpp.s

CMakeFiles/multi.dir/cll.cpp.o.requires:
.PHONY : CMakeFiles/multi.dir/cll.cpp.o.requires

CMakeFiles/multi.dir/cll.cpp.o.provides: CMakeFiles/multi.dir/cll.cpp.o.requires
	$(MAKE) -f CMakeFiles/multi.dir/build.make CMakeFiles/multi.dir/cll.cpp.o.provides.build
.PHONY : CMakeFiles/multi.dir/cll.cpp.o.provides

CMakeFiles/multi.dir/cll.cpp.o.provides.build: CMakeFiles/multi.dir/cll.cpp.o

CMakeFiles/multi.dir/part1.cpp.o: CMakeFiles/multi.dir/flags.make
CMakeFiles/multi.dir/part1.cpp.o: ../part1.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/multi.dir/part1.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/multi.dir/part1.cpp.o -c /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/part1.cpp

CMakeFiles/multi.dir/part1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/multi.dir/part1.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/part1.cpp > CMakeFiles/multi.dir/part1.cpp.i

CMakeFiles/multi.dir/part1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/multi.dir/part1.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/part1.cpp -o CMakeFiles/multi.dir/part1.cpp.s

CMakeFiles/multi.dir/part1.cpp.o.requires:
.PHONY : CMakeFiles/multi.dir/part1.cpp.o.requires

CMakeFiles/multi.dir/part1.cpp.o.provides: CMakeFiles/multi.dir/part1.cpp.o.requires
	$(MAKE) -f CMakeFiles/multi.dir/build.make CMakeFiles/multi.dir/part1.cpp.o.provides.build
.PHONY : CMakeFiles/multi.dir/part1.cpp.o.provides

CMakeFiles/multi.dir/part1.cpp.o.provides.build: CMakeFiles/multi.dir/part1.cpp.o

CMakeFiles/multi.dir/util.cpp.o: CMakeFiles/multi.dir/flags.make
CMakeFiles/multi.dir/util.cpp.o: ../util.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/multi.dir/util.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/multi.dir/util.cpp.o -c /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/util.cpp

CMakeFiles/multi.dir/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/multi.dir/util.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/util.cpp > CMakeFiles/multi.dir/util.cpp.i

CMakeFiles/multi.dir/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/multi.dir/util.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/util.cpp -o CMakeFiles/multi.dir/util.cpp.s

CMakeFiles/multi.dir/util.cpp.o.requires:
.PHONY : CMakeFiles/multi.dir/util.cpp.o.requires

CMakeFiles/multi.dir/util.cpp.o.provides: CMakeFiles/multi.dir/util.cpp.o.requires
	$(MAKE) -f CMakeFiles/multi.dir/build.make CMakeFiles/multi.dir/util.cpp.o.provides.build
.PHONY : CMakeFiles/multi.dir/util.cpp.o.provides

CMakeFiles/multi.dir/util.cpp.o.provides.build: CMakeFiles/multi.dir/util.cpp.o

# Object files for target multi
multi_OBJECTS = \
"CMakeFiles/multi.dir/cll.cpp.o" \
"CMakeFiles/multi.dir/part1.cpp.o" \
"CMakeFiles/multi.dir/util.cpp.o"

# External object files for target multi
multi_EXTERNAL_OBJECTS =

libmulti.a: CMakeFiles/multi.dir/cll.cpp.o
libmulti.a: CMakeFiles/multi.dir/part1.cpp.o
libmulti.a: CMakeFiles/multi.dir/util.cpp.o
libmulti.a: CMakeFiles/multi.dir/build.make
libmulti.a: CMakeFiles/multi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libmulti.a"
	$(CMAKE_COMMAND) -P CMakeFiles/multi.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/multi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/multi.dir/build: libmulti.a
.PHONY : CMakeFiles/multi.dir/build

CMakeFiles/multi.dir/requires: CMakeFiles/multi.dir/cll.cpp.o.requires
CMakeFiles/multi.dir/requires: CMakeFiles/multi.dir/part1.cpp.o.requires
CMakeFiles/multi.dir/requires: CMakeFiles/multi.dir/util.cpp.o.requires
.PHONY : CMakeFiles/multi.dir/requires

CMakeFiles/multi.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/multi.dir/cmake_clean.cmake
.PHONY : CMakeFiles/multi.dir/clean

CMakeFiles/multi.dir/depend:
	cd /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds /home/thiago/repositorios/tcc/projetosopencl/multiplicacaodematriz/builds/CMakeFiles/multi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/multi.dir/depend

