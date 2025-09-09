# Basic OpenGL


## How to install "make" command and C/C++ compilers:


### For Windows (make):


#### Option 1 (Install with `Chocolatey`):

1. Install `Chocolatey` from following website: \
   https://chocolatey.org/install

2. After finishing the installation intall `make` using the following command on the CMD (Run as adimistrator):
   ```
   choco install make
   ```


#### Option 2 (Install with `MSYS2`):

1. Install `MSYS2` from the following website: \
   https://www.msys2.org/

2. Open `MSYS2` and intall `make` using the following command (on the opened console):
   ```
   pacman -S make
   ```


### For Windows (C/C++ compilers):

1. Go to the following website: \
   https://code.visualstudio.com/docs/cpp/config-mingw

2. Follow the guide on how to install the C/C++ compilers and validate that they are installed by using the following commands on the CMD:
   ```
   gcc --version
   g++ --version
   ```


### For MacOS (make `+` C/C++ compilers):

1. Install `Xcode Command Line Tools` using the following command on the Terminal:
   ```
   xcode-select --install
   ```

2. (Optional) Install `brew` from the following website: \
   https://brew.sh/

3. (Optional) Install with `brew` the `GLFW` library using the following command on the Terminal:
   ```
   brew install glfw
   ```


## For Linux (make `+` C/C++ compilers):

1. (Optional) Update the Linux OS using the following command on the Terminal:
   ```
   sudo apt update
   ```

2. Install `g++` using the following command on the Terminal:
   ```
   sudo apt install build-essential g++
   ```

3. Install `OpenGL` and `GLFW` using the following command on the Terminal:
   ```
   sudo apt install libgl-dev libglfw3-dev libxi-dev
   ```


## How to compile and run the OpenGL Engine using the Makefile:


### Using the CMD/Terminal:

1. Open the CMD/Terminal.

2. Navigate to the project folder using `cd` commands.

3. Run the following command:
   ```
   make
   ```

4. Run the following commands:
   ```
   cd bin
   ./main
   ```


### Using Visual Studio Code:

1. Download Visual Studio Code from: \
   https://code.visualstudio.com/download

2. Download the `C/C++ Extension Pack` from the Visual Studio Code Extensions or from: \
   https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack

3. Download the `Makefile Tools Extension` from the Visual Studio Code Extensions or from: \
   https://marketplace.visualstudio.com/items?itemName=ms-vscode.makefile-tools

4. Configure the `Makefile Tools Extension` to compile the engine: using the available [Makefile](Makefile):

   1. Edit the `Makefile` option, and set `The path to the makefile of the project` as: `Makefile`.

   2. Click on the `...` button near the `Play` button, and then choose: `Makefile:Configure`.

   3. Edit the `Lunch target` option, and choose the first option: `main()`.

   4. Click on the `Play` button and validate that the program compiles and runs successfully.

`Notice:` With this tool you can run the OpenGL in Debugging mode as well.


## MacOS known issue with "libglfw.3.dylib" file:

The MacOS tends to block the file: "libglfw.3.dylib" which is crucial for running the OpenGL Engine. 
To remove the block, open the MacOS "Settings", go to "Privacy & Security", scroll down until you find the "libglfw.3.dylib" and choose to enable permission for it.


## Useful guides for building the OpenGL Engine:

- For Windows: \
  https://www.youtube.com/watch?v=hRInLNR9iRg

- For MacOS: \
  https://www.youtube.com/watch?v=7-dL6a5_B3I

- For Linux: \
  https://www.youtube.com/watch?v=JxDLGHil-Cw


## Useful YouTube guides:

- The Cherno: \
  https://www.youtube.com/watch?v=W3gAzLwfIP0&list=PLlrATfBNZ98foTJPJ_Ev03o2oq3-GGOS2

- Victor Gordan: \
  https://www.youtube.com/watch?v=XpBGwZNyUh0&list=PLPaoO-vpZnumdcb4tZc4x5Q-v7CkrQ6M-


## Library sources:

- GLFW: \
  https://www.glfw.org/

- GLAD: \
  https://glad.dav1d.de/

- GLEW (Not used, But works similar to GLAD): \
  https://glew.sourceforge.net/

- stb: \
  https://github.com/nothings/stb

- glm: \
  https://github.com/g-truc/glm/releases/tag/1.0.1

  ///unitl 27.5
