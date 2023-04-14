cmake

- pasta_do_projeto
- fazer CMakeList.txt
- mkdir build e cd build
- cmake ..
- make
- ./TFLiteImageClassification ../../models/classification/mobilenet_v1_1.0_224_quant.tflite ../../models/classification/labels_mobilenet_quant_v1_224.txt ../../images/classification_example.jpg

- nome_projeto
	- src
		- main.cc (main acessa ClasseTeste)
		- CalsseTeste
			- ClasseTeste.cc
	- build
		- CMakeLists
	- headers
		- ClasseTeste.h.h

## Cmake Exemplo

touch CMakeLists.txt
```makefile
# cmake --version = 3.17
cmake_minimum_required(VERSION 3.17)

# nome do projeto
project(TFLiteImageClassification)

# qual tipo de c++ está sendo utilizado
set(CMAKE_CXX_STANDARD 14)

# OpenCV Integration
# encontrar o path do pacote
find_package(OpenCV REQUIRED)
# adicionar o diretório no include
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorFlow Lite Integration
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)

# nome do projeto e path/main.cc <posso adicionar outros arquivos>
add_executable(${PROJECT_NAME} main.cc)


ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/linux_x64/libtensorflowlite.so)
target_link_libraries(TFLiteImageClassification tensorflowlite ${OpenCV_LIBS})
target_link_libraries(TFLiteImageClassification tensorflowlite)


```


**Encontrar arquivos de cabeçalho**
diretorio_cabecalho
```makefile
add_executable(${PROJECT_NAME} ../src/main.cc ../src/CalsseTeste/CalsseTeste.cc)
target_include_directories(NomeProjeto PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/headers)
```
