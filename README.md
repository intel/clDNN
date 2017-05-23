
# Compute Library for Deep Neural Networks (clDNN)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
![v1.0](https://img.shields.io/badge/1.0-RC1-green.svg)

*Compute Library for Deep Neural Networks* (*clDNN*) is an open source performance
library for Deep Learning (DL) applications intended for acceleration of
DL Inference on Intel® Processor Graphics – including HD Graphics and
Iris® Graphics.  
*clDNN* includes highly optimized building blocks for implementation of
convolutional neural networks (CNN) with C and C++ interfaces. We created
this project to enable the DL community to innovate on Intel® processors.

**Usages supported:** Image recognition, image detection, and image segmentation.

**Topologies:** AlexNet\*, VGG\*, GoogleNet\* and ResNet\*.

As with any technical preview, APIs may change in future updates.

## License
clDNN is licensed is licensed under
[Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

### Attached licenses
clDNN uses 3<sup>rd</sup>-party components licensed under following licenses:
- *boost* under [Boost\* Software License - Version 1.0](http://www.boost.org/LICENSE_1_0.txt)
- *googletest* under [Google\* License](https://github.com/google/googletest/blob/master/googletest/LICENSE)
- *OpenCL™ ICD and C++ Wrapper* under [Khronos™ License](https://github.com/KhronosGroup/OpenCL-CLHPP/blob/master/LICENSE.txt)

## Documentation
The latest clDNN documentation is at [GitHub pages](http://01org.github.io/cldnn/).

There is also inline documentation available that can be [generated with Doxygen](#generating-documentation).

## Support
Please report issues and suggestions 
[GitHub issues](https://github.com/01org/cldnn/issues).

## How to Contribute
We welcome community contributions to clDNN. If you have an idea how to improve the library:

- Share your proposal via
 [GitHub issues](https://github.com/01org/cldnn/issues)
- Ensure you can build the product and run all the examples with your patch
- In the case of a larger feature, create a test
- Submit a [pull request](https://github.com/01org/cldnn/pulls)

We will review your contribution and, if any additional fixes or modifications
are necessary, may provide feedback to guide you. When accepted, your pull
request will be merged into our internal and GitHub repositories.

## System Requirements
clDNN supports Intel® HD Graphics and Intel® Iris® Graphics and is optimized for
- Codename *Skylake*:
    * Intel® HD Graphics 510 (GT1, *client* market)
    * Intel® HD Graphics 515 (GT2, *client* market)
    * Intel® HD Graphics 520 (GT2, *client* market)
    * Intel® HD Graphics 530 (GT2, *client* market)
    * Intel® Iris® Graphics 540 (GT3e, *client* market)
    * Intel® Iris® Graphics 550 (GT3e, *client* market)
    * Intel® Iris® Pro Graphics 580 (GT4e, *client* market)
    * Intel® HD Graphics P530 (GT2, *server* market)
    * Intel® Iris® Pro Graphics P555 (GT3e, *server* market)
    * Intel® Iris® Pro Graphics P580 (GT4e, *server* market)
- Codename *Apollolake*:
    * Intel® HD Graphics 500
    * Intel® HD Graphics 505

clDNN currently uses OpenCL™ with multiple Intel® OpenCL™ extensions and requires Intel® Graphics Driver to run.

clDNN requires CPU with Intel® SSE/Intel® AVX support.

---

The software dependencies are:
- [CMake\*](https://cmake.org/download/) 3.6 or later  
(the project is compatible with CMake 3.1, but, due to issues with boost libraries resolution
in CMake 3.4.3 and with CheckCXXCompilerFlag module in CMake 3.5.2, we strongly recommend 3.6+)
- C++ compiler with partiall or full C++14 standard support compatible with:
    * GNU\* Compiler Collection 5.2 or later
    * clang 3.5 or later
    * [Intel® C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe) 17.0 or later
    * Visual C++ 2015 (MSVC++ 19.0) or later

> Intel® CPU intrinsics header (`<immintrin.h>`) must be available during compilation.

- [python™](https://www.python.org/downloads/) 2.7 or later (scripts are both compatible with python™ 2.7.x and python™ 3.x)
- *(optional)* [Doxygen\*](http://www.stack.nl/~dimitri/doxygen/download.html) 1.8.13 or later  
    Needed for manual generation of documentation from inline comments or running `docs` custom target which will generate it automatically.

> [GraphViz\*](http://www.graphviz.org/Download..php) (2.38 or later) is also recommended to generate documentation with all embedded diagrams.  
(Make sure that `dot` application is visible in the `PATH` environment variable.)

---

The software was validated on:
- CentOS\* 7 with GNU\* Compiler Collection 5.2 (64-bit only)
- Windows® 10 and Windows® Server 2012 R2 with MSVC 14.0

We have validated using [Intel® intel-opencl-r4.1 (SRB4.1) Linux driver package](http://registrationcenter-download.intel.com/akdlm/irc_nas/11396/SRB4.1_linux64.zip).

## Installation

### Building

Download [clDNN source code](https://github.com/01org/cldnn/archive/master.zip)
or clone the repository to your system:

```
    git clone https://github.com/01org/cldnn.git
```

Satisfy all software dependencies and ensure that the versions are correct before building.

clDNN uses multiple 3<sup>rd</sup>-party components. They are stored in binary form in `common` subdirectory. Currently they are prepared for MSVC++ and GCC\*. They will be cloned with repository.

---

clDNN uses a CMake-based build system. You can use CMake command-line tool or CMake GUI (`cmake-gui`) to generate required solution.  
For Windows system, you can call in `cmd` (or `powershell`):
```shellscript
    @REM Generate 32-bit solution (solution contains multiple build configurations)...
    cmake -E make_directory build && cd build && cmake -G "Visual Studio 14 2015" ..
    @REM Generate 64-bit solution (solution contains multiple build configurations)...
    cmake -E make_directory build && cd build && cmake -G "Visual Studio 14 2015 Win64" ..
```  
Created solution can be opened in Visual Studio 2015 or built using appropriate `msbuild` tool
(you can also use `cmake --build .` to select build tool automatically).

For Unix and Linux systems:
```shellscript
    @REM Create GNU makefile for release clDNN and build it...
    cmake -E make_directory build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make
    @REM Create Ninja makefile for debug clDNN and build it...
    cmake -E make_directory build && cd build && cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug .. && ninja -k 20
```

You can call also scripts in main directory of project which will create solutions/makefiles for clDNN (they
will generate solutions/makefiles in `build` subdirectory and binary outputs will be written to `build/out` subdirectory):
- `create_msvc_mscc.bat` (Windows\*, Visual Studio\* 2015)
- `create_unixmake_gcc.sh [Y|N] [<devtoolset-version>]` (Linux\*, GNU\* or Ninja\* makefiles, optional devtoolset support)
    * If you specify the first parameter as `Y`, the Ninja makefiles will be generated.
    * If you specify second parameter (number), the CMake will be called via `scl` with selected `devtoolset` version.

CMake solution offers multiple options which you can specify using normal CMake syntax (`-D<option-name>=<value>`):

| CMake option                              | Type     | Description                                                                  |
|:------------------------------------------|:---------|:-----------------------------------------------------------------------------|
| CMAKE\_BUILD\_TYPE                        | STRING   | Build configuration that will be used by generated makefiles (it does not affect multi-configuration generators like generators for Visual Studio solutions). Currently supported: `Debug` (default), `Release` |
| CMAKE\_INSTALL\_PREFIX                    | PATH     | Install directory prefix.                                                    |
| CLDNN\_\_ARCHITECTURE\_TARGET             | STRING   | Architecture of target system (where binary output will be deployed). CMake will try to detect it automatically (based on selected generator type, host OS and compiler properties). Specify this option only if CMake has problem with detection. Currently supported: `Windows32`, `Windows64`, `Linux64` |
| CLDNN\_\_OUTPUT\_DIR (CLDNN\_\_OUTPUT\_BIN\_DIR, CLDNN\_\_OUTPUT\_LIB\_DIR) | PATH | Location where built artifacts will be written to. It is set automatically to roughly `build/out/<arch-target>/<build-type>` subdirectory. For more control use: `CLDNN__OUTPUT_LIB_DIR` (specifies output path for static libraries) or `CLDNN__OUTPUT_BIN_DIR` (for shared libs and executables). |
|                                           |          |                                                                              |
| **CMake advanced option**                 | **Type** | **Description**                                                              |
| PYTHON\_EXECUTABLE                        | FILEPATH | Path to Python interpreter. CMake will try to detect Python. Specify this option only if CMake has problem with locating Python. |
| CLDNN\_\_BOOST\_VERSION                   | STRING   | Version of boost prebuilded binaries to use (from `common` subdirectory). It is automatically setected by CMake (highest version). Specify, if you have multiple versions and want to use different than automatically selected. |
| CLDNN\_\_IOCL\_ICD\_USE\_EXTERNAL         | BOOL     | Use this option to enable use of external Intel® OpenCL™ SDK as a source for ICD binaries and headers (based on `INTELOCLSDKROOT` environment variable). Default: `OFF` |
| CLDNN\_\_IOCL\_ICD\_VERSION               | STRING   | Version of Intel® OpenCL™ ICD binaries and headers to use (from `common` subdirectory). It is automatically setected by CMake (highest version). Specify, if you have multiple versions and want to use different than automatically selected. |
|                                           |          |                                                                              |
| CLDNN__COMPILE_LINK_ALLOW_UNSAFE_SIZE_OPT | BOOL     | Allow unsafe optimizations during linking (like aggressive dead code elimination, etc.). Default: `ON` |
| CLDNN__COMPILE_LINK_USE_STATIC_RUNTIME    | BOOL     | Link with static C++ runtime. Default: `OFF` (shared C++ runtime is used)    |
|                                           |          |                                                                              |
| CLDNN__INCLUDE_CORE                       | BOOL     | Include core clDNN library project in generated makefiles/solutions. Default: `ON` |
| CLDNN__INCLUDE_TESTS                      | BOOL     | Include tests application project (based on googletest framework) in generated makefiles/solutions . Default: `ON` |
|                                           |          |                                                                              |
| CLDNN__RUN_TESTS                          | BOOL     | Run tests after building `tests` project. This option requires `CLDNN__INCLUDE_TESTS` option to be `ON`. Default: `OFF` |
|                                           |          |                                                                              |
| CLDNN__CMAKE_DEBUG                        | BOOL     | Enable extended debug messages in CMake. Default: `OFF`                      |
    
---

clDNN includes unit tests implemented using the googletest framework. To validate your build, run `tests` target, e.g.:

```
    make tests
```

(Make sure that both `CLDNN__INCLUDE_TESTS` and `CLDNN__RUN_TESTS` were set to `ON` when invoking CMake.)

### Generating documentation

Documentation is provided inline and can be generated in HTML format with Doxygen. We recommend to use latest
[Doxygen\*](http://www.stack.nl/~dimitri/doxygen/download.html) and [GraphViz\*](http://www.graphviz.org/Download..php).

Documentation templates and configuration files are stored in `docs` subdirectory. You can simply call:

```shellscript
    cd docs && doxygen
```
to generate HTML documentation in `docs/html` subdirectory.

There is also custom CMake target named `docs` which will generate documentation in `CLDNN__OUTPUT_BIN_DIR/html` directory. For example, when using Unix makefiles, you can run:
```
    make docs
```
in order to create it.

### Deployment

Special `install` target will place the API header files and libraries in `/usr/local`
(`C:/Program Files/clDNN` or `C:/Program Files (x86)/clDNN` on Windows). To change
the installation path, use the option `-DCMAKE_INSTALL_PREFIX=<prefix>` when invoking CMake.

## Disclaimers, Warranties, Legal Notices and Limitations of Liability
> You may not use or facilitate the use of this document in connection with any
infringement or other legal analysis concerning Intel products described herein.
You agree to grant Intel® a non-exclusive, royalty-free license to any patent claim
thereafter drafted which includes subject matter disclosed herein.  
No license (express or implied, by estoppel or otherwise) to any intellectual property
rights is granted by this document.  
Intel® technologies' features and benefits depend on system configuration and may require
enabled hardware, software or service activation. Performance varies depending on system
configuration. No computer system can be absolutely secure. Check with your system manufacturer
or retailer or learn more at [intel.com](http://www.intel.com).  
Intel® technologies may require enabled hardware, specific software, or services activation.
Check with your system manufacturer or retailer.  
The products described may contain design defects or errors known as errata which may cause
the product to deviate from published specifications. Current characterized errata are
available on request.  
Intel® disclaims all express and implied warranties, including without limitation, the implied
warranties of merchantability, fitness for a particular purpose, and non-infringement,
as well as any warranty arising from course of performance, course of dealing, or usage in trade.  
All information provided here is subject to change without notice. Contact your Intel®
representative to obtain the latest Intel product specifications and roadmaps.  
Copies of documents which have an order number and are referenced in this document may be
obtained by calling 1-800-548-4725 or visit [here](http://www.intel.com/design/literature.htm).  
Intel, Centrino, vPro, Core, Thunderbolt, Ultrabook and the Intel logo are trademarks of
Intel® Corporation in the U.S. and other countries.


---


\* Other names and brands may be claimed as the property of others.

Copyright © 2017, Intel® Corporation
