# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
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

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/macbad2012/assignment4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/macbad2012/assignment4/build

# Include any dependencies generated for this target.
include CMakeFiles/draw_scene.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/draw_scene.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/draw_scene.dir/flags.make

CMakeFiles/draw_scene.dir/draw_scene.cc.o: CMakeFiles/draw_scene.dir/flags.make
CMakeFiles/draw_scene.dir/draw_scene.cc.o: ../draw_scene.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/macbad2012/assignment4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/draw_scene.dir/draw_scene.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/draw_scene.dir/draw_scene.cc.o -c /home/macbad2012/assignment4/draw_scene.cc

CMakeFiles/draw_scene.dir/draw_scene.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/draw_scene.dir/draw_scene.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/macbad2012/assignment4/draw_scene.cc > CMakeFiles/draw_scene.dir/draw_scene.cc.i

CMakeFiles/draw_scene.dir/draw_scene.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/draw_scene.dir/draw_scene.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/macbad2012/assignment4/draw_scene.cc -o CMakeFiles/draw_scene.dir/draw_scene.cc.s

CMakeFiles/draw_scene.dir/draw_scene.cc.o.requires:

.PHONY : CMakeFiles/draw_scene.dir/draw_scene.cc.o.requires

CMakeFiles/draw_scene.dir/draw_scene.cc.o.provides: CMakeFiles/draw_scene.dir/draw_scene.cc.o.requires
	$(MAKE) -f CMakeFiles/draw_scene.dir/build.make CMakeFiles/draw_scene.dir/draw_scene.cc.o.provides.build
.PHONY : CMakeFiles/draw_scene.dir/draw_scene.cc.o.provides

CMakeFiles/draw_scene.dir/draw_scene.cc.o.provides.build: CMakeFiles/draw_scene.dir/draw_scene.cc.o


CMakeFiles/draw_scene.dir/shader_program.cc.o: CMakeFiles/draw_scene.dir/flags.make
CMakeFiles/draw_scene.dir/shader_program.cc.o: ../shader_program.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/macbad2012/assignment4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/draw_scene.dir/shader_program.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/draw_scene.dir/shader_program.cc.o -c /home/macbad2012/assignment4/shader_program.cc

CMakeFiles/draw_scene.dir/shader_program.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/draw_scene.dir/shader_program.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/macbad2012/assignment4/shader_program.cc > CMakeFiles/draw_scene.dir/shader_program.cc.i

CMakeFiles/draw_scene.dir/shader_program.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/draw_scene.dir/shader_program.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/macbad2012/assignment4/shader_program.cc -o CMakeFiles/draw_scene.dir/shader_program.cc.s

CMakeFiles/draw_scene.dir/shader_program.cc.o.requires:

.PHONY : CMakeFiles/draw_scene.dir/shader_program.cc.o.requires

CMakeFiles/draw_scene.dir/shader_program.cc.o.provides: CMakeFiles/draw_scene.dir/shader_program.cc.o.requires
	$(MAKE) -f CMakeFiles/draw_scene.dir/build.make CMakeFiles/draw_scene.dir/shader_program.cc.o.provides.build
.PHONY : CMakeFiles/draw_scene.dir/shader_program.cc.o.provides

CMakeFiles/draw_scene.dir/shader_program.cc.o.provides.build: CMakeFiles/draw_scene.dir/shader_program.cc.o


CMakeFiles/draw_scene.dir/model.cc.o: CMakeFiles/draw_scene.dir/flags.make
CMakeFiles/draw_scene.dir/model.cc.o: ../model.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/macbad2012/assignment4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/draw_scene.dir/model.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/draw_scene.dir/model.cc.o -c /home/macbad2012/assignment4/model.cc

CMakeFiles/draw_scene.dir/model.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/draw_scene.dir/model.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/macbad2012/assignment4/model.cc > CMakeFiles/draw_scene.dir/model.cc.i

CMakeFiles/draw_scene.dir/model.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/draw_scene.dir/model.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/macbad2012/assignment4/model.cc -o CMakeFiles/draw_scene.dir/model.cc.s

CMakeFiles/draw_scene.dir/model.cc.o.requires:

.PHONY : CMakeFiles/draw_scene.dir/model.cc.o.requires

CMakeFiles/draw_scene.dir/model.cc.o.provides: CMakeFiles/draw_scene.dir/model.cc.o.requires
	$(MAKE) -f CMakeFiles/draw_scene.dir/build.make CMakeFiles/draw_scene.dir/model.cc.o.provides.build
.PHONY : CMakeFiles/draw_scene.dir/model.cc.o.provides

CMakeFiles/draw_scene.dir/model.cc.o.provides.build: CMakeFiles/draw_scene.dir/model.cc.o


CMakeFiles/draw_scene.dir/model_loader.cc.o: CMakeFiles/draw_scene.dir/flags.make
CMakeFiles/draw_scene.dir/model_loader.cc.o: ../model_loader.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/macbad2012/assignment4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/draw_scene.dir/model_loader.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/draw_scene.dir/model_loader.cc.o -c /home/macbad2012/assignment4/model_loader.cc

CMakeFiles/draw_scene.dir/model_loader.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/draw_scene.dir/model_loader.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/macbad2012/assignment4/model_loader.cc > CMakeFiles/draw_scene.dir/model_loader.cc.i

CMakeFiles/draw_scene.dir/model_loader.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/draw_scene.dir/model_loader.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/macbad2012/assignment4/model_loader.cc -o CMakeFiles/draw_scene.dir/model_loader.cc.s

CMakeFiles/draw_scene.dir/model_loader.cc.o.requires:

.PHONY : CMakeFiles/draw_scene.dir/model_loader.cc.o.requires

CMakeFiles/draw_scene.dir/model_loader.cc.o.provides: CMakeFiles/draw_scene.dir/model_loader.cc.o.requires
	$(MAKE) -f CMakeFiles/draw_scene.dir/build.make CMakeFiles/draw_scene.dir/model_loader.cc.o.provides.build
.PHONY : CMakeFiles/draw_scene.dir/model_loader.cc.o.provides

CMakeFiles/draw_scene.dir/model_loader.cc.o.provides.build: CMakeFiles/draw_scene.dir/model_loader.cc.o


CMakeFiles/draw_scene.dir/transformations.cc.o: CMakeFiles/draw_scene.dir/flags.make
CMakeFiles/draw_scene.dir/transformations.cc.o: ../transformations.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/macbad2012/assignment4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/draw_scene.dir/transformations.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/draw_scene.dir/transformations.cc.o -c /home/macbad2012/assignment4/transformations.cc

CMakeFiles/draw_scene.dir/transformations.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/draw_scene.dir/transformations.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/macbad2012/assignment4/transformations.cc > CMakeFiles/draw_scene.dir/transformations.cc.i

CMakeFiles/draw_scene.dir/transformations.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/draw_scene.dir/transformations.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/macbad2012/assignment4/transformations.cc -o CMakeFiles/draw_scene.dir/transformations.cc.s

CMakeFiles/draw_scene.dir/transformations.cc.o.requires:

.PHONY : CMakeFiles/draw_scene.dir/transformations.cc.o.requires

CMakeFiles/draw_scene.dir/transformations.cc.o.provides: CMakeFiles/draw_scene.dir/transformations.cc.o.requires
	$(MAKE) -f CMakeFiles/draw_scene.dir/build.make CMakeFiles/draw_scene.dir/transformations.cc.o.provides.build
.PHONY : CMakeFiles/draw_scene.dir/transformations.cc.o.provides

CMakeFiles/draw_scene.dir/transformations.cc.o.provides.build: CMakeFiles/draw_scene.dir/transformations.cc.o


CMakeFiles/draw_scene.dir/camera_utils.cc.o: CMakeFiles/draw_scene.dir/flags.make
CMakeFiles/draw_scene.dir/camera_utils.cc.o: ../camera_utils.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/macbad2012/assignment4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/draw_scene.dir/camera_utils.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/draw_scene.dir/camera_utils.cc.o -c /home/macbad2012/assignment4/camera_utils.cc

CMakeFiles/draw_scene.dir/camera_utils.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/draw_scene.dir/camera_utils.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/macbad2012/assignment4/camera_utils.cc > CMakeFiles/draw_scene.dir/camera_utils.cc.i

CMakeFiles/draw_scene.dir/camera_utils.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/draw_scene.dir/camera_utils.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/macbad2012/assignment4/camera_utils.cc -o CMakeFiles/draw_scene.dir/camera_utils.cc.s

CMakeFiles/draw_scene.dir/camera_utils.cc.o.requires:

.PHONY : CMakeFiles/draw_scene.dir/camera_utils.cc.o.requires

CMakeFiles/draw_scene.dir/camera_utils.cc.o.provides: CMakeFiles/draw_scene.dir/camera_utils.cc.o.requires
	$(MAKE) -f CMakeFiles/draw_scene.dir/build.make CMakeFiles/draw_scene.dir/camera_utils.cc.o.provides.build
.PHONY : CMakeFiles/draw_scene.dir/camera_utils.cc.o.provides

CMakeFiles/draw_scene.dir/camera_utils.cc.o.provides.build: CMakeFiles/draw_scene.dir/camera_utils.cc.o


# Object files for target draw_scene
draw_scene_OBJECTS = \
"CMakeFiles/draw_scene.dir/draw_scene.cc.o" \
"CMakeFiles/draw_scene.dir/shader_program.cc.o" \
"CMakeFiles/draw_scene.dir/model.cc.o" \
"CMakeFiles/draw_scene.dir/model_loader.cc.o" \
"CMakeFiles/draw_scene.dir/transformations.cc.o" \
"CMakeFiles/draw_scene.dir/camera_utils.cc.o"

# External object files for target draw_scene
draw_scene_EXTERNAL_OBJECTS =

bin/draw_scene: CMakeFiles/draw_scene.dir/draw_scene.cc.o
bin/draw_scene: CMakeFiles/draw_scene.dir/shader_program.cc.o
bin/draw_scene: CMakeFiles/draw_scene.dir/model.cc.o
bin/draw_scene: CMakeFiles/draw_scene.dir/model_loader.cc.o
bin/draw_scene: CMakeFiles/draw_scene.dir/transformations.cc.o
bin/draw_scene: CMakeFiles/draw_scene.dir/camera_utils.cc.o
bin/draw_scene: CMakeFiles/draw_scene.dir/build.make
bin/draw_scene: lib/libglfw3.a
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libGLU.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libGL.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libGLEW.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/librt.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libm.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libX11.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libXrandr.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libXinerama.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libXxf86vm.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libXcursor.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libGL.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libgflags.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libglog.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libGLEW.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/librt.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libm.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libX11.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libXrandr.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libXinerama.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libXxf86vm.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libXcursor.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libgflags.so
bin/draw_scene: /usr/lib/x86_64-linux-gnu/libglog.so
bin/draw_scene: CMakeFiles/draw_scene.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/macbad2012/assignment4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable bin/draw_scene"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/draw_scene.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/draw_scene.dir/build: bin/draw_scene

.PHONY : CMakeFiles/draw_scene.dir/build

CMakeFiles/draw_scene.dir/requires: CMakeFiles/draw_scene.dir/draw_scene.cc.o.requires
CMakeFiles/draw_scene.dir/requires: CMakeFiles/draw_scene.dir/shader_program.cc.o.requires
CMakeFiles/draw_scene.dir/requires: CMakeFiles/draw_scene.dir/model.cc.o.requires
CMakeFiles/draw_scene.dir/requires: CMakeFiles/draw_scene.dir/model_loader.cc.o.requires
CMakeFiles/draw_scene.dir/requires: CMakeFiles/draw_scene.dir/transformations.cc.o.requires
CMakeFiles/draw_scene.dir/requires: CMakeFiles/draw_scene.dir/camera_utils.cc.o.requires

.PHONY : CMakeFiles/draw_scene.dir/requires

CMakeFiles/draw_scene.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/draw_scene.dir/cmake_clean.cmake
.PHONY : CMakeFiles/draw_scene.dir/clean

CMakeFiles/draw_scene.dir/depend:
	cd /home/macbad2012/assignment4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/macbad2012/assignment4 /home/macbad2012/assignment4 /home/macbad2012/assignment4/build /home/macbad2012/assignment4/build /home/macbad2012/assignment4/build/CMakeFiles/draw_scene.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/draw_scene.dir/depend

