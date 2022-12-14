# Inteligência artificial em sistemas embarcados utilizando abordagem TinyML.


Trabalho de Conclusão de Curso - UTFPR-PB

## Resumo

<div style="text-align: justify">  
A internet das coisas está presente em inúmeras áreas da nossa rotina e em diferentes locais, conquistando feitos impressionantes e percorrendo um caminho de avanço constante, o que nos leva a necessidade de desenvolver implementações cada vez mais sofisticadas. Com isso, um novo desafio surge, implantar aprendizado de máquina em pequenos dispositivos do dia a dia. O campo responsável por conectar sistemas embarcados e o aprendizado de máquina é o tinyML e baseando-se nesse método, o presente trabalho busca realizar a implantação de uma rede neural artificial que realize o reconhecimento de placas veiculares em um hardware de baixo custo. Dessa forma, treinou-se um modelo MobileNetV2-SSDLite para detecção de placas automotivas, efetuando a quantização fazendo uso do \textit{framework} do TensorFlow Lite e implantando a solução em um sistema embarcado Raspberry Pi Zero 2W. Os resultados obtidos para os modelos otimizados, obtiveram um bom desempenho em termos de latência, pouca variação na precisão ao ser comparado com o modelo original, mostrando que o tinyML é uma ótima alternativa para aplicações embarcadas.
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