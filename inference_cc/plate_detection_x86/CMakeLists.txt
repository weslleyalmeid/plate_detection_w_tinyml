# cmake --version = 3.17
CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

# nome do projeto
PROJECT(platedetection)

# SET(VARIAVEL valor): defina variáveis normais ou de ambiente para um valor 
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE Debug)

# OpenCV Integration
# FIND_PACKAGE(NOME_PACOTE REQUIRED): encontra o pacote e carregue suas dependências
FIND_PACKAGE(OpenCV REQUIRED)
# incluir diretórios para que seja possível compilar o pacote
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# ADD_EXECUTABLE(NOME_DO_PROJETO path/main.cc <path/arquivos>): Arquivos a sere executados
# ADD_EXECUTABLE(${PROJECT_NAME} compare.cc)
ADD_EXECUTABLE(${PROJECT_NAME} main.cc)
# ADD_EXECUTABLE(${PROJECT_NAME} experiments.cc)

# TARGET_LINK_LIBRARIES(TARGET itens): especificar bibliotecas para o executável do projeto, no caso a main.cc
TARGET_LINK_LIBRARIES(platedetection  ${OpenCV_LIBS} tesseract)

# TensorFlow Lite Integration
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/tflite-dist/include/)

# ADD_LIBRARY(TARGET SHARED IMPORTED)
ADD_LIBRARY(tensorflowlite SHARED IMPORTED)

SET_PROPERTY(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/tflite-dist/libs/linux_x64/libtensorflowlite.so)
TARGET_LINK_LIBRARIES(platedetection tensorflowlite)
