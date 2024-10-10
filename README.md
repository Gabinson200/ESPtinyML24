| Supported Targets | ESP32 | ESP32-C2 | ESP32-C3 | ESP32-C6 | ESP32-H2 | ESP32-P4 | ESP32-S2 | ESP32-S3 |
| ----------------- | ----- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |

# _Sample project_

(See the README.md file in the upper level 'examples' directory for more information about examples.)

This is the simplest buildable example. The example is used by command `idf.py create-project`
that copies the project to user specified path and set it's name. For more information follow the [docs page](https://docs.espressif.com/projects/esp-idf/en/latest/api-guides/build-system.html#start-a-new-project)

This github consists of 4 main parts
```
├── micro_speech
├── model_making
├── tamagotchi
├── combined
```
The tamagotchi esp-idf folder that encapuslates all non-ML functionality of our microcontroller for hardware components such as OLED, joystick, Wifi, etc. and their software such as the animations and game functionality. 

The model_making folder which will have the python files used to build, train, quantize, and convert the speech recognition model using the Tensorflow Lite Micro pipeline that will be used by the micro_speech application.

The micro_speech folder which also defines an esp idf project similar to tamagotchi but is purely responsible for the implementation of the wake word detection functionality.

Finally the combined file will have the combined esp idf code for the combination of the tamagotchi functionality and the wake word detection. 

Documentation for the different files will be housed inside their respective folders.



## How to use example
We encourage the users to use the example as a template for the new projects.
A recommended way is to follow the instructions on a [docs page](https://docs.espressif.com/projects/esp-idf/en/latest/api-guides/build-system.html#start-a-new-project).

## Example folder contents

The project **sample_project** contains one source file in C language [main.c](main/main.c). The file is located in folder [main](main).

ESP-IDF projects are built using CMake. The project build configuration is contained in `CMakeLists.txt`
files that provide set of directives and instructions describing the project's source files and targets
(executable, library, or both). 

Below is short explanation of remaining files in the project folder.

```
├── CMakeLists.txt
├── main
│   ├── CMakeLists.txt
│   └── main.c
└── README.md                  This is the file you are currently reading
```
Additionally, the sample project contains Makefile and component.mk files, used for the legacy Make based build system. 
They are not used or needed when building with CMake and idf.py.
