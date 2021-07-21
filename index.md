# PROCESSAMENTO DIGITAL DE IMAGENS
## Resolução dos exercicos práticos

## Tutorial 2
### Exemplo 2.1
Utilizando o programa exemplos/pixels.cpp como referência, implemente um programa regions.cpp. Esse programa deverá solicitar ao usuário as coordenadas de dois pontos P1 e P2 localizados dentro dos limites do tamanho da imagem e exibir que lhe for fornecida. Entretanto, a região definida pelo retângulo de vértices opostos definidos pelos pontos P1 e P2 será exibida com o negativo da imagem na região correspondente.

**implementação em c++**
~~~c++
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    
    int p1x, p1y, p2x, p2y;
    cout << "Coordenadas de P1: ";
    cin >> p1y >> p1x;

    cout << "Coordenadas de P2: ";
    cin >> p2y >> p2x;
    
    cv::Mat image;
    image = cv::imread(argv[1],cv::IMREAD_GRAYSCALE);

    if(!image.data)
        std::cout << "Não abriu " << argv[1] << std::endl;
    
    cv::namedWindow("janela", cv::WINDOW_AUTOSIZE);

    for (int i = p1x; i < p2x; i++) {
        for (int j = p1y; j < p2y; j++) {
            image.at<uchar>(i,j) = 255 - image.at<uchar>(i,j);
        }
    }

    cv::imshow("janela", image);
    cv::waitKey();

    return 0; 
}
~~~

**exemplo de entrada**
~~~
cyber pixels 
$ ./regions biel.png 
Coordenadas de P1: 30 50
Coordenadas de P2: 100 230
~~~

**Comparações entre a imagem original e o resultado obtido**

![Image](images/ex1/biel.png) ![Image](images/ex1/ex1.png)

### Exemplo 2.1
Utilizando o programa exemplos/pixels.cpp como referência, implemente um programa trocaregioes.cpp. Seu programa deverá trocar os quadrantes em diagonal na imagem.

**implementação em c++**
~~~c++
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    
    cv::Mat image;
    
    image= cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
    if(!image.data)
        std::cout << "nao abriu " << argv[1] << std::endl;

    int r = image.rows;
    int c = image.cols;
    uchar aux;
    
    for(int i = 0; i < c/2; i++){
        for(int j = 0; j < r/2; j++){
            aux = image.at<uchar>(i,j);
            image.at<uchar>(i,j) = image.at<uchar>(i+c/2,j+r/2);
            image.at<uchar>(i+c/2,j+r/2) = aux;
        }
    }
    
    for(int i = 0; i < c/2; i++){
        for(int j = 0; j < r/2; j++){
            aux = image.at<uchar>(i,j+c/2);
            image.at<uchar>(i,j+c/2) = image.at<uchar>(i+r/2,j);
            image.at<uchar>(i+r/2,j) = aux;
        }
    }

    cv::imshow("janela", image);
    cv::waitKey();

    return 0; 
}
~~~

**exemplo de entrada**
~~~
cyber pixels 
$ ./trocaregioes biel.png
~~~

**Resultados**

![Image](images/ex1/ex2.png)

