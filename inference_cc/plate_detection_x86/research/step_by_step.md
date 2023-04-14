# Hello World Tensorflow

## Clonar repositório Tensorflow
```sh
https://github.com/tensorflow/tflite-micro
```
**Caminho dos exemplos**
```sh
tensorflow/lite/micro/examples
```

## Baixar arquivos necessários
```sh
cd tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads
```

## Compilar script Hello World
```sh
# make -f tensorflow/lite/micro/tools/make/Makefile name_my_project
make -f tensorflow/lite/micro/tools/make/Makefile hello_world
```

## Output
**Costuma aparecer após o último -o**
```sh
# gen/linux_x86_64_default/bin/name_my_project
gen/linux_x86_64_default/bin/hello_world
```

## Estrutura de pastas necessárias para executar fora do repositório

pastas importantes
- gen
- tensorflow
- third_party

```sh
./research/directories.md
```


# Person Detection Tensorflow

## Compilar script
```sh
make -f tensorflow/lite/micro/tools/make/Makefile test_person_detection_test
```

## Output
**Costuma aparecer após o último -o**
```sh
# gen/linux_x86_64_default/bin/name_my_project
gen/linux_x86_64_default/bin/person_detection_test
```



# Plate Detection Tensorflow

## Compilar script
```sh
make -f tensorflow/lite/micro/tools/make/Makefile plate_detection_test

gen/linux_x86_64_default/bin/plate_detection_test
```

## Output
**Costuma aparecer após o último -o**
```sh
# gen/linux_x86_64_default/bin/name_my_project
gen/linux_x86_64_default/bin/person_detection_test


# Ajustando estrutura para detecção de placas

- Copiando estrutura do detecção de pessoa e alterando para plate_detection

- Tudo se inicia na main_funtions, acessando cada dependência e fazendo os devidos ajustes
- tensorflow/lite/micro/examples/plate_detection/main_functions.cc

- MICRO_LITE_GEN_MUTABLE_OP_RESOLVER_TEST 
```sh
tensorflow/lite/micro/tools/make/Makefile
```

- gen/linux_x86_64_default/bin/plate_detection_test




### instalar opencv


 sudo apt-get install libjpeg-dev libpng-dev libtiff5-dev libdc1394-dev libeigen3-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev

sudo apt-get install libdc1394-dev
sudo apt-get -y install libswresample-dev
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev


RODAR make install

https://www.youtube.com/watch?v=AmLCJdHUPEk
continuar tutorial do link acima 


# mkdir funcionando

# debug model
gdb --args ./TFLiteImageClassification ../../models/classification/mobilenet_v1_1.0_224_quant.tflite ../../models/classification/labels_mobilenet_quant_v1_224.txt ../../images/classification_example.jpg


# Executando projeto com cmake

- ctrl + shift + p
- pesquisar cmake: start
- configurar launch.json, settings e tasks



# Unlock file do tflite.so
sudo chown -R $USER: $HOME


gcc -I/tflite-dist/include/tensorflow main.cc `pkg-config --cflags --libs opencv4` -o model -ldl -ltensorflow-lite

valgrind --tool=massif 

valgrind --tool=massif main

g++ -g -I/home/wa/Desktop/2022-1/TCC2/DESENVOLVIMENTO/plate_detection/tflite-dist/include -I/home/wa/Desktop/2022-1/TCC2/DESENVOLVIMENTO/plate_detection/tflite-dist/include -o /home/wa/Desktop/2022-1/TCC2/DESENVOLVIMENTO/plate_detection/main.o `pkg-config --cflags --libs opencv4` -c /home/wa/Desktop/2022-1/TCC2/DESENVOLVIMENTO/plate_detection/main.cc




lauch.json

```json
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/local/include/opencv4",
                "${workspaceFolder}/tflite-dist/include",
            ],
            "defines": [
                "_DEBUG",
                "UNICODE",
                "_UNICODE"
            ],
            "windowsSdkVersion": "10.0.19041.0",
            "compilerPath": "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30037/bin/Hostx64/x64/cl.exe",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "windows-msvc-x64",
            "configurationProvider": "ms-vscode.cmake-tools"
        }
    ],
    "version": 4
}
```

### Verificar devices conectados
```sh
sudo apt-get install v4l-utils 
v4l2-ctl --list-devices 
```

### [Deploy Rasp](https://www.youtube.com/watch?v=GkBskiRatwM&list=PLEB5F4gTNK68ax-Ekhej32jreRgFWYR5o&index=2)
[ref - blog](https://learnembeddedsystems.co.uk/headless-raspberry-pi-setup)
1 - Baixar imagem lite
2 - salvar em um sd
3 - criar dois arquivos, um para conectar na rede wifi e outro para habilitar o ssh
```vi
country=BR
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
	scan_ssid=1
	ssid="Casa_Almeida_2G"
	psk="tanageladeira"
}
```
4 - descobrir o ip de acesso ssh
```sh
# broadcast
ping -b 192.168.0.255 

# listar devices
sudo nmap -sn 192.168.0.1/24
```
5 - Acessar com ssh e logar
```sh
# porta padrão 22 do ssh nome_user@localhost
ssh -p 22 pi@192.168.0.xx
ssh -p 22 pi@192.168.0.23
senha: xlesqu85


# - the default username is “pi” and the default password: “raspberry“
```
6 - Alterar a senha do ssh
```sh
passwd
```
7 - Atualizar sistema
```sh
sudo apt update
sudo apt upgrade
```
8 - Atribuir IP estático
```sh
sudo vim /etc/dhcpcd.conf

# procure pelo nome abaixo e altere
static ip_address=192.168.0.200/24

# reiniciar para salvar
sudo reboot
```


## Enviando arquivos

```sh
# problemas com host
ssh-keygen -R 192.168.0.23


# copiando do remoto para o local
# scp username@b:/path/to/file /path/to/destination
scp pi@192.168.0.23:/home/pi/test.jpg /home/wa/Desktop


# copiando do local para o remoto, -r é opcional para enviar pasta
# scp /path/to/file username@a:/path/to/destination
scp -R ./test_run pi@192.168.0.23:/home/pi

```

## Compilação cruzada para arm

```
sudo apt-get install gcc-arm-linux-gnueabihf

```





g++ `pkg-config --cflags --libs opencv4` -o main.o -c main.cc -ltensorflowlite_c

gcc main.c -o test -ltensorflowlite_c

export TENSORFLOW_ROOT_DIR=./tflite-dist/include/tensorflow

g++ `pkg-config --cflags --libs opencv4` -ltesseract -I${TENSORFLOW_ROOT_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include -I${TENSORFLOW_ROOT_DIR} -pthread -Wall -Wextra -pedantic -o teste main.cc -L${TENSORFLOW_ROOT_DIR}/tensorflow/lite/tools/make/gen/aarch64_armv8/lib -ltensorflow-lite


onde ${TENSORFLOW_ROOT_DIR} é o caminho para a raiz do repositório tensorflow clonado.
A tag -pthread parece ser exigida pela biblioteca eigen




g++ main.cc -o minimal -std=c++11 -I. `pkg-config --cflags --libs opencv4` -ltesseract  -I/home/pi/plate_detection/tflite-dist/include/ -Itensorflow/lite/tools/make/downloads/absl/ -I/home/pi/plate_detection/tensorflow/lite/tools/make/downloads/flatbuffers/include/ -L/tf-lite/ -ltensorflow-lite -lrt -ldl -pthread


## FUNCIONAL
g++ main.cc -o test_agora `pkg-config --cflags --libs opencv4` -I/usr/local/include -Ltesseract -L/tf-lite/ -ltensorflow-lite -lrt -ldl -pthread


g++ main.cc -o quase `pkg-config --cflags --libs opencv4` -I/usr/local/include -ltesseract -L/tf-lite/ -ltensorflow-lite -lrt -ldl -pthread


### ÚLTIMO TESTADO

g++ main.cc -o ./build/plate_detection `pkg-config --cflags --libs opencv4` -I/usr/local/include -ltesseract -L/tf-lite/ -ltensorflow-lite -lrt -ldl -pthread
g++ main.cc -o ./build/plate_detection `pkg-config --cflags --libs opencv4` -I/usr/local/include -ltesseract -L/tf-lite/ -ltensorflow-lite -lrt -ldl
g++ main.cc -o ./build/minimal `pkg-config --cflags --libs opencv4` -I/usr/local/include -ltesseract -ltensorflow-lite -lrt -ldl


Testando modelo opencv puro para quantificar consumo de memória
```sh
g++ raw.cc -o raw `pkg-config --cflags --libs opencv4` -I/usr/local/include -ltesseract -I./plate_detection/tflite-dist/include/
```


Testando execução nas imagens
```sh
g++ compare.cc -o ./build/compare `pkg-config --cflags --libs opencv4` -I/usr/local/include -ltesseract -ltensorflow-lite
g++ compare.cc -o ./build/compare2 `pkg-config --cflags --libs opencv4` -I/usr/local/include -ltesseract -ltensorflow-lite
```

Testando execução no vídeo
```sh
g++ main.cc -o ./build/video2 `pkg-config --cflags --libs opencv4` -I/usr/local/include -ltesseract -ltensorflow-lite -lrt -ldl
```

## Passo a passo Rasp3

- ~~OpenCV~~
- ~~Tesseract~~
- ~~Tensorflow~~
- ~~Ajustar Camera nas confs~~
- ~~ler camera usb~~
- ~~migrar executável~~
- ~~ajustar processador~~
- ~~compilar e gravar~~
