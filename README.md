# Inteligência artificial em sistemas embarcados utilizando abordagem TinyML.


Trabalho de Conclusão de Curso - UTFPR-PB

## Resumo

<div style="text-align: justify"> 
A internet das coisas está presente em inúmeras áreas da nossa rotina e em diferentes locais, conquistando feitos impressionantes e percorrendo um caminho de avanço constante, o que nos leva a necessidade de desenvolver  implementações cada vez mais sofisticadas. Com isso, um novo desafio surge, implantar aprendizado de máquina em pequenos dispositivos do dia a dia. O campo responsável por conectar sistemas embarcados e o aprendizado de máquina é o tinyML e baseado nesse método, o presente trabalho busca realizar a implantação de uma rede neural artificial que realize o reconhecimento de placas veiculares em um hardware de baixo custo, buscando exemplificar a detecção de placas automotivas em um sistema embarcado Raspberry Pi Zero 2W, utilizando-se da abordagem tinyML e o framework do TensorFlow Lite para quantização do modelo MobileNetV2-SSDLite.
</div>


## Preparar ambiente

- Instalar OpenCV
- Instalar Tesseract
- Instalar Tensorflow 
- Instalar Tensorflow Lite

## Execução em C++

```
g++ main.cc -o ./build/plate `pkg-config --cflags --libs opencv4` -I/usr/local/include -ltesseract -ltensorflow-lite
```

## Demonstração

- [link-youtube](https://youtu.be/9tQOT_NEWOQ)


## Licença
Esse projeto está sob a [licença](LICENSE) MIT.