workspaceFolder = .

# Detect OS
ifeq ($(OS),Windows_NT) # Windows
    CPPFLAGS = g++ --std=c++17 -fdiagnostics-color=always -Wall -g \
        -I${workspaceFolder}/includes \
        -I${workspaceFolder}/src \
        -I"C:/msys64/ucrt64/include/opencv4"

            # add this so src/glad.c rule works on Windows
    CFLAGS = gcc -std=c11 -Wall -g \
        -I${workspaceFolder}/includes \
        -I${workspaceFolder}/src \
        -I"C:/msys64/ucrt64/include/opencv4"

    CLIBS = -L${workspaceFolder}/lib/windows \
        -L"C:/msys64/ucrt64/lib"

    LDFLAGS = -lglfw3dll -lopengl32 -lglu32 -lgdi32 \
        -lopencv_calib3d -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
        -lopencv_features2d -lopencv_flann
    all: copy_lib_w copy_res_w build
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S), Darwin) # macOS
        CPPFLAGS = clang++ -std=c++17 -fcolor-diagnostics -fansi-escape-codes -Wall -g -I${workspaceFolder}/include -I${workspaceFolder}/src
        CFLAGS = clang -std=c11 -Wall -g -I${workspaceFolder}/include -I${workspaceFolder}/src
        CLIBS = -L${workspaceFolder}/lib/macOS ${workspaceFolder}/bin/libglfw.3.dylib
        LDFLAGS = -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo -framework CoreFoundation -Wno-deprecated -Wl,-rpath,.
        all: copy_lib_m copy_res_m build
    else ifeq ($(UNAME_S), Linux) # Linux
        CPPFLAGS = g++ --std=c++17 -fdiagnostics-color=always -Wall -g -I${workspaceFolder}/include -I${workspaceFolder}/src
        CFLAGS = gcc -std=c11 -Wall -g -I${workspaceFolder}/include -I${workspaceFolder}/src
        CLIBS = -L${workspaceFolder}/lib/linux
        LDFLAGS = -lglfw -lGL -lX11 -lpthread -lXrandr -lXi -ldl
        all: copy_lib_l copy_res_l build
    else
        $(error Unsupported OS: $(UNAME_S))
    endif
endif

# Source and object files
SRC_FILES = $(wildcard ${workspaceFolder}/src/*.cpp)
OBJ_FILES = $(patsubst ${workspaceFolder}/src/%.cpp, ${workspaceFolder}/bin/%.o, $(SRC_FILES)) ${workspaceFolder}/bin/glad.o

# Rule to compile .o files from .cpp files
${workspaceFolder}/bin/%.o: ${workspaceFolder}/src/%.cpp | $(workspaceFolder)/bin
	$(CPPFLAGS) -c $< -o $@

# Rule to compile glad.o
${workspaceFolder}/bin/glad.o: ${workspaceFolder}/src/glad.c | $(workspaceFolder)/bin
	$(CFLAGS) -c $< -o $@

build: $(OBJ_FILES) | $(workspaceFolder)/bin
	$(CPPFLAGS) $(CLIBS) $(OBJ_FILES) -o ${workspaceFolder}/bin/main $(LDFLAGS)

# Copy library and resources (MacOS)
copy_lib_m:
	@echo "Copying library for MacOS..."
	cp -rf ${workspaceFolder}/lib/macOS/libglfw.3.dylib ${workspaceFolder}/bin/libglfw.3.dylib

copy_res_m:
	@echo "Copying resources for MacOS..."
	mkdir -p ${workspaceFolder}/bin/res && cp -rf ${workspaceFolder}/src/res/* ${workspaceFolder}/bin/res

# Copy library and resources (Windows)
copy_lib_w:
	@echo "Copying library for Windows..."
	xcopy /D /I ${workspaceFolder}\lib\windows\glfw3.dll ${workspaceFolder}\bin\

copy_res_w:
	@echo "Copying resources for Windows..."
	xcopy /E /I /Y ${workspaceFolder}\src\res ${workspaceFolder}\bin\res

# Copy library and resources (Linux)
copy_lib_l:
	@echo "Copying library for Linux... [Skipped!]"
	# cp -rf ${workspaceFolder}/lib/linux/* ${workspaceFolder}/bin/

copy_res_l:
	@echo "Copying resources for Linux..."
	mkdir -p ${workspaceFolder}/bin/res && cp -rf ${workspaceFolder}/src/res/* ${workspaceFolder}/bin/res

# Parallel build (add -jN option to run with N jobs)
.PHONY: all copy_res_m copy_res_w
