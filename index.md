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

### Exemplo 2.2
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

## Tutorial 3
### Exemplo 3.1
Observando-se o programa labeling.cpp como exemplo, é possível verificar que caso existam mais de 255 objetos na cena, o processo de rotulação poderá ficar comprometido. Identifique a situação em que isso ocorre e proponha uma solução para este problema.

R.: Como nós estamos rotulando as bolhas utilizando as cores apenas na escala de cinza (0 à 255), excluindo o 0 pois é a cor do fundo, esse processo só funciona se tivermos ate 255 bolhas na tela. A solução proposta é usar todas a escolas de cores do RGB, tendo asssim um número incrivelmente grande de cores possíveis para rotular os objetos. Usando esse método podemos rotular até 255^3 objetos. O código abaixo implementa essa funcionalidade.

~~~c++
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv){
  cv::Mat image, realce;
  int width, height;
  int nobjects;
  cv::Vec3b objcolor;
  cv::Vec3b branco;

  cv::Point p;
  image = cv::imread(argv[1], cv::IMREAD_COLOR);

  if(!image.data){
    std::cout << "imagem nao carregou corretamente\n";
    return(-1);
  }

  width=image.cols;
  height=image.rows;
  std::cout << width << "x" << height << std::endl;

  p.x=0;
  p.y=0;

  // busca objetos presentes
  nobjects=0;
  
  int b=0;
  int g=0;
  int r=0;
  objcolor[0]=b;
  objcolor[1]=g;
  objcolor[2]=r;

  branco[0]=255;
  branco[1]=255;
  branco[2]=255;

  for(int i=0; i<height; i++){
    for(int j=0; j<width; j++){
      if(image.at<Vec3b>(i,j) == branco){
        // achou um objeto
        nobjects++;
        b+=1;
        if(b==255) {
            g++;
            b=0;
        }else if(b==255 && r==255) {
            r++;
            b=0;
            g=0;
        }

        objcolor[0]=b;
        objcolor[1]=g;
        objcolor[2]=r;
        p.x=j;
        p.y=i;
  		// preenche o objeto com o contador
		cv::floodFill(image, p, objcolor);
      }
    }
  }
  std::cout << "a figura tem " << nobjects << " bolhas\n";
  cv::imshow("image", image);
  cv::imwrite("labeling.png", image);
  cv::waitKey();

  return 0;
}
~~~

**Resultados**

![Image](images/ex2/bolhas1.png) ![Image](images/ex2/labeling.png)
