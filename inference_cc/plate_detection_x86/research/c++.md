

### Inicialização

```c
#include <iostream>

int main(int argc, char** argv){

    // std é o pacote e tem uma função chamada cout
    // = é o <<
    std::cout << "Hello, World!";

    system("pause")
    return 0;
}
```

### Types

```c
#include <iostream>

int main(int argc, char** argv){

    int myInt =10;

    // unsigned int
    unsigned int myIntUnsigned = -10;
    float myFloat = 0.0f;
    double myDouble = 0.0;
    bool myBool = false;
    char myChar = 't';

    // unsigned não pode receber numeros negativos
    size_t mySizeT = 0;

    return 0;
}
```

### String

```c
#include <iostream>
#include <string>

int main(int argc, char** argv){

    char myChar = 't';

    // tamanho = elemento + '\0'
    char myArray[11] = "0123456789";


    // usando string permite alocação dinâmica e operações simplificadas
    std::string myString = "Teste";

    // abacate é bom bem
    MyString = "abacate";
    MyString += "é bom bem";

    return 0;
}
```

### Console input e output

```c
#include <iostream>
#include <string>

int main(int argc, char** argv){

    while (true){
        // << output 
        // >> cin input em x
        std:cin >> x;

        if (x == 99){
            break;
        }

        if (x > 0){
            std:count << "X positivo\n";
        }
        else if (x < 0){
            std:count << "X negativo\n";
        }
        else{
            std:count << "X zero\n";
        }
    }


    for(int i = 0; i < 10; i++){
        std:cin >> x;

        if (x > 0){
            std:count << "X positivo\n";
        }
        else if (x < 0){
            std:count << "X negativo\n";
        }
        else{
            std:count << "X zero\n";
        }
    }


    return 0;
}
```


### Looping


```c
#include <iostream>
#include <string>

int main(int argc, char** argv){

    while (true){
        // << output 
        // >> cin input em x
        std:cin >> x;

        if (x == 99){
            break;
        }

        if (x > 0){
            std:count << "X positivo\n";
        }
        else if (x < 0){
            std:count << "X negativo\n";
        }
        else{
            std:count << "X zero\n";
        }
    }

    for(int i = 0; i < 10; i++){
        std:cin >> x;

        if (x > 0){
            std:count << "X positivo\n";
        }
        else if (x < 0){
            std:count << "X negativo\n";
        }
        else{
            std:count << "X zero\n";
        }
    }


```


### Array

```c
#include <iostream>
#include <string>

int main(int argc, char** argv){
    int myArray[10]

    for(int i = 0; i < 10; i++){
        myArray[i] = 0;
    }

    for(int i = 0; i < 10; i++){
        std::count << myArray[i] << "\n";
    }

    return 0;
}
```


### Funções

```c
#include <iostream>
#include <string>

// <type de return> <name function> (<argumentos>){
    // {<escopo>}
// }
// obtem o endereço de memoria do value e altera o valor
void printTest(int x, int &avalue){
    std::cout << x << "argumentos";
    avalue += 1;
}

int main(int argc, char** argv){
    
    int x = 10;
    int abacate = 20;
    printTest(10, abacate);

    return 0;
}
```

## Struct e Classes

**C**
```c

struct Human{
    
    std:string name;
    int age;
    float height;
    float weight;
}

int main(int argc, char** argv){
    
    Human me;

    me.name = "Weslley";
    me.name = 28;
    me.height = 1.76f;
    me.weight = 90.f;

    return 0;
}

```

**C++ - Struct**
```c

// Classe e Struct pouco diferem, porém, em struct todos atributos são PUBLIC
struct Human{
    // constructor
    Human(){
        this->name = "Abacate";
        this->age = 22;
        this->height = 1.50f;
        this->weight = 50.f;
    }

    // segundo construtor
    Human(std::string n, int age=10){
        this->name = n;
        this->age = age;
        this->height = 1.50f;
        this->weight = 50.f;
    }

    void ShowID(){
        std::cout << name;
        std::cout << age;
        std::cout << height;
        std::cout << weight;    
    }

    std:string name;
    int age;
    float height;
    float weight;
}

int main(int argc, char** argv){
    
    Human me;

    me.name = "Weslley";
    me.age = 28;
    me.height = 1.76f;
    me.weight = 90.f;

    me.ShowID();

    // abacate setado no construtor 
    Human abacate;

    // laranja setado no segundo construtor 
    Human laranja("Jaca", 1000);

    return 0;
}

```


**C++ - Classe**
```c

// Classe e Struct pouco diferem, porém, em struct todos atributos são PUBLIC
// Construtores e atributos são privados, é preciso especificar o que é public e private
class Human{

public:
    // constructor
    Human(){
        this->name = "Abacate";
        this->age = 22;
        this->height = 1.50f;
        this->weight = 50.f;
    }

    // segundo construtor
    Human(std::string n, int age=10){
        this->name = n;
        this->age = age;
        this->height = 1.50f;
        this->weight = 50.f;
    }

    void ShowID(){
        std::cout << name;
        std::cout << age;
        std::cout << height;
        std::cout << weight;    
    }

    void  Birthday(){
        this->age++;
        std::count << this->name << " now is " << age << " years old\n";
    }

private:
    std:string name;
    int age;
    float height;
    float weight;
}

int main(int argc, char** argv){
    
    Human me;

    // gera erro, não é possível setar manualmente o atributo
    // me.name = "Weslley";
    // me.age = 28;
    // me.height = 1.76f;
    // me.weight = 90.f;

    me.ShowID();

    // abacate setado no construtor 
    Human abacate;

    // laranja setado no segundo construtor 
    Human laranja("Jaca", 1000);

    laranja.Birthday();
    return 0;
}

```

## Templates e Struct

```c
// template< typename T > struct Allowed
std::unique_ptr<tflite::Interpreter> interpreter;
```

// gcc main.cc -I${fileDirname}/tflite-dist/include/tensorflow `pkg-config --cflags --libs opencv4` -o model



## Instalar tesseract-ocr
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev 