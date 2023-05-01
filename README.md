# Inteligência artificial em sistemas embarcados utilizando abordagem TinyML.


Trabalho de Conclusão de Curso - UTFPR-PB

## Resumo

<div style="text-align: justify"> 
Atualmente, existem mais de 250 bilhões de microcontroladores no mundo, logo, viabilizar
a possibilidade de executar modelos de aprendizado de máquina em sistemas embarcados é
motivado porque muitos dos dados captados por sensores são descartados devido ao custo,
necessidade de conexão com internet ou restrições de energia. Diante desse cenário, surge uma
nova abordagem de soluções de aprendizado de máquina, que integram sistemas com recursos
limitados e inteligência artificial, o Tiny Machine Learning (TinyML), essa abordagem tem como
objetivo viabilizar a implantação de modelos de inteligência artificial em sistemas embarcados
de baixo custo e pouco poder de processamento. Entre as aplicações que poderiam se beneficiar
da integração de aprendizado de máquina e hardware com recursos limitados, são os dispositivos
de reconhecimento automático. Portanto, o presente trabalho busca realizar a implantação de
uma rede neural artificial que realize o reconhecimento de placas veiculares em um hardware de
baixo custo. Dessa forma, treinou-se um modelo MobileNetV2 SSD FPN-Lite para detecção
de placas automotivas, efetuando a quantização fazendo uso do framework do TensorFlow Lite
e implantando a solução em um sistema embarcado Raspberry Pi Zero 2W. Foram realizados
experimentos em quatro formas de quantização e entre duas linguagens distintas, Python e
C++. O melhor resultado apresentado considerando o tamanho de armazenamento, índice de
confiabilidade e tempo de latência, foi o da quantização dinâmica em C++, pois, comparado ao
modelo não quantizado, obteve uma redução em armazenamento de 75%, apresentando um score
de 72,38% contra 72,28% do modelo não quantizado, e, uma eficiência no tempo de execução
de 20%. Tornando assim, o tinyML uma alternativa viável para aplicações em sistemas com
limitações de recursos.
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