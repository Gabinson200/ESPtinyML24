# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "C:/Users/adamk/esp/v5.2.1/esp-idf/components/bootloader/subproject"
  "C:/Users/adamk/nagama_just_games/build/bootloader"
  "C:/Users/adamk/nagama_just_games/build/bootloader-prefix"
  "C:/Users/adamk/nagama_just_games/build/bootloader-prefix/tmp"
  "C:/Users/adamk/nagama_just_games/build/bootloader-prefix/src/bootloader-stamp"
  "C:/Users/adamk/nagama_just_games/build/bootloader-prefix/src"
  "C:/Users/adamk/nagama_just_games/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/adamk/nagama_just_games/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/adamk/nagama_just_games/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
