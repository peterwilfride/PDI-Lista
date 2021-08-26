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

![Image](images/ex2/biel.png) ![Image](images/ex2/ex1.png)


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

![Image](images/ex2/ex2.png)

## Tutorial 3
### Exemplo 3.1
Observando-se o programa labeling.cpp como exemplo, é possível verificar que caso existam mais de 255 objetos na cena, o processo de rotulação poderá ficar comprometido. Identifique a situação em que isso ocorre e proponha uma solução para este problema.

**R.:** Como nós estamos rotulando as bolhas utilizando as cores apenas na escala de cinza (0 à 255), excluindo o 0 pois é a cor do fundo, esse processo só funciona se tivermos ate 255 bolhas na tela. A solução proposta é usar todas a escolas de cores do RGB, tendo asssim um número incrivelmente grande de cores possíveis para rotular os objetos. Usando esse método podemos rotular até 255^3 objetos. O código abaixo implementa essa funcionalidade.

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

![Image](images/ex3/bolhas1.png) ![Image](images/ex3/labeling.png)

## Tutorial 4
### Exemplo 4.1
Utilizando o programa exemplos/histogram.cpp como referência, implemente um programa equalize.cpp. Este deverá, para cada imagem capturada, realizar a equalização do histogram antes de exibir a imagem. Teste sua implementação apontando a câmera para ambientes com iluminações variadas e observando o efeito gerado. Assuma que as imagens processadas serão em tons de cinza.

**Implementação**

~~~c++
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv){
  cv::Mat image;
  int width, height;
  cv::VideoCapture cap;
  std::vector<cv::Mat> planes;
  cv::Mat hist;
  int nbins = 64;
  float range[] = {0, 255};
  const float *histrange = { range };
  bool uniform = true;
  bool acummulate = false;
  int key;

	cap.open(0);

  if(!cap.isOpened()){
    std::cout << "cameras indisponiveis";
    return -1;
  }

  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

  std::cout << "largura = " << width << std::endl;
  std::cout << "altura  = " << height << std::endl;

  int histw = nbins, histh = nbins/2;
  cv::Mat histImg(histh, histw, CV_8UC3, cv::Scalar(0,0,0));

  while(1){
    cap >> image;
    cv::split (image, planes);
    equalizeHist(planes[0], planes[0]);
    cv::calcHist(&planes[0], 1, 0, cv::Mat(), hist, 1, &nbins, &histrange, uniform, acummulate);
    cv::normalize(hist, hist, 0, histImg.rows, cv::NORM_MINMAX, -1, cv::Mat());

    histImg.setTo(cv::Scalar(0));

    for(int i=0; i<nbins; i++){
      cv::line(histImg,
               cv::Point(i, histh),
               cv::Point(i, histh-cvRound(hist.at<float>(i))),
               cv::Scalar(255, 0, 0), 1, 8, 0);
    }

    histImg.copyTo(image(cv::Rect(0, 2*histh ,nbins, histh)));

    cv::imshow("image", image);
    
    key = cv::waitKey(30);
    if(key == 27) break;
  }
  return 0;
}
~~~

### Exemplo 4.2
Utilizando o programa exemplos/histogram.cpp como referência, implemente um programa motiondetector.cpp. Este deverá continuamente calcular o histograma da imagem (apenas uma componente de cor é suficiente) e compará-lo com o último histograma calculado. Quando a diferença entre estes ultrapassar um limiar pré-estabelecido, ative um alarme. Utilize uma função de comparação que julgar conveniente.

~~~c++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;

int main(int argc, char** argv){
  cv::Mat image;
  int width, height;
  cv::VideoCapture cap;
  std::vector<cv::Mat> planes;
  cv::Mat hist;
  int nbins = 64;
  float range[] = {0, 255};
  const float *histrange = { range };
  bool uniform = true;
  bool acummulate = false;
  int key;

	cap.open(0);

  if(!cap.isOpened()){
    std::cout << "cameras indisponiveis";
    return -1;
  }

  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

  std::cout << "largura = " << width << std::endl;
  std::cout << "altura  = " << height << std::endl;

  int histw = nbins, histh = nbins/2;
  cv::Mat histImg(histh, histw, CV_8UC3, cv::Scalar(0,0,0));

  cv::Mat old_hist;

    cap >> image;

    cv::split (image, planes);

    cv::calcHist(&planes[0], 1, 0, cv::Mat(), old_hist, 1, &nbins, &histrange, uniform, acummulate);
    cv::normalize(old_hist, old_hist, 0, histImg.rows, cv::NORM_MINMAX, -1, cv::Mat());

  std::cout << "Starting..." << std::endl;

    int aux = 0.0;

  while(1){
    cap >> image;

    cv::split (image, planes);

    cv::calcHist(&planes[0], 1, 0, cv::Mat(), hist, 1, &nbins, &histrange, uniform, acummulate);
    cv::normalize(hist, hist, 0, histImg.rows, cv::NORM_MINMAX, -1, cv::Mat());

    double compar_chi = cv::compareHist(hist, old_hist, 2);

    if (abs(compar_chi - aux) > 25.0)
        cout << "Está em movimento!" << abs(compar_chi - aux) << endl;

    histImg.setTo(cv::Scalar(0));

    for(int i=0; i<nbins; i++){
      cv::line(histImg,
               cv::Point(i, histh),
               cv::Point(i, histh-cvRound(hist.at<float>(i))),
               cv::Scalar(255, 0, 0), 1, 8, 0);
    }

    histImg.copyTo(image(cv::Rect(0, 2*histh ,nbins, histh)));
    
    aux = compar_chi;
    old_hist = hist;

    cv::imshow("image", image);
    key = cv::waitKey(30);
    if(key == 27) break;
  }
  return 0;
}
~~~

## Tutorial 5
### Exemplo 5.1
Utilizando o programa exemplos/filtroespacial.cpp como referência, implemente um programa laplgauss.cpp. O programa deverá acrescentar mais uma funcionalidade ao exemplo fornecido, permitindo que seja calculado o laplaciano do gaussiano das imagens capturadas. Compare o resultado desse filtro com a simples aplicação do filtro laplaciano.

**Máscara usada para aplicar o filtro laplaciano do gaussiano**

![Image](images/ex5/matrix.png)

**Implementação**

~~~c++
#include <iostream>
#include <opencv2/opencv.hpp>

void printmask(cv::Mat &m) {
  for (int i = 0; i < m.size().height; i++) {
    for (int j = 0; j < m.size().width; j++) {
      std::cout << m.at<float>(i, j) << ",";
    }
    std::cout << "\n";
  }
}

int main(int, char **) {
  cv::VideoCapture cap;  // open the default camera
  float media[] = {0.1111, 0.1111, 0.1111, 0.1111, 0.1111,
                   0.1111, 0.1111, 0.1111, 0.1111};
  float gauss[] = {0.0625, 0.125,  0.0625, 0.125, 0.25,
                   0.125,  0.0625, 0.125,  0.0625};
  float horizontal[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  float vertical[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  float laplacian[] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
  float boost[] = {0, -1, 0, -1, 5.2, -1, 0, -1, 0};
  float laplgauss[] = { 0, 0, 1, 0, 0,
                        0, 1, 2, 1, 0,
                        1, 2, -16, 2, 1,
                        0, 1, 2, 1, 0,
                        0, 0, 1, 0, 0};

  cv::Mat frame, framegray, frame32f, frameFiltered;
  cv::Mat mask(3, 3, CV_32F);
  cv::Mat result;
  double width, height;
  int absolut;
  char key;

  cap.open(0);

  if (!cap.isOpened())  // check if we succeeded
    return -1;

  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  std::cout << "largura=" << width << "\n";
  ;
  std::cout << "altura =" << height << "\n";
  ;
  std::cout << "fps    =" << cap.get(cv::CAP_PROP_FPS) << "\n";
  std::cout << "format =" << cap.get(cv::CAP_PROP_FORMAT) << "\n";

  cv::namedWindow("filtroespacial", cv::WINDOW_NORMAL);
  cv::namedWindow("original", cv::WINDOW_NORMAL);

  mask = cv::Mat(3, 3, CV_32F, media);

  absolut = 1;  // calcs abs of the image

  for (;;) {
    cap >> frame;  // get a new frame from camera
    cv::cvtColor(frame, framegray, cv::COLOR_BGR2GRAY);
    cv::flip(framegray, framegray, 1);
    cv::imshow("original", framegray);
    framegray.convertTo(frame32f, CV_32F);
    cv::filter2D(frame32f, frameFiltered, frame32f.depth(), mask,
                 cv::Point(1, 1), 0);
    if (absolut) {
      frameFiltered = cv::abs(frameFiltered);
    }

    frameFiltered.convertTo(result, CV_8U);

    cv::imshow("filtroespacial", result);

    key = (char)cv::waitKey(10);
    if (key == 27) break;  // esc pressed!
    switch (key) {
      case 'a':
        absolut = !absolut;
        break;
      case 'm':
        mask = cv::Mat(3, 3, CV_32F, media);
        printmask(mask);
        break;
      case 'g':
        mask = cv::Mat(3, 3, CV_32F, gauss);
        printmask(mask);
        break;
      case 'h':
        mask = cv::Mat(3, 3, CV_32F, horizontal);
        printmask(mask);
        break;
      case 'v':
        mask = cv::Mat(3, 3, CV_32F, vertical);
        printmask(mask);
        break;
      case 'l':
        mask = cv::Mat(3, 3, CV_32F, laplacian);
        printmask(mask);
        break;
      case 'b':
        mask = cv::Mat(3, 3, CV_32F, boost);
        break;
      case 'w':
        mask = cv::Mat(5, 5, CV_32F, laplgauss);
        printmask(mask);
        break;
      default:
        break;
    }
  }
  return 0;
}
~~~

**Resultado**

![Image](images/ex5/ex5-2.png) 

Imagem aplicado filtro laplaciano

![Image](images/ex5/ex5.png)

Imagem aplicado filtro laplaciano do gaussiano

## Tutorial 6
### Exemplo 6.1
Utilizando o programa exemplos/addweighted.cpp como referência, implemente um programa tiltshift.cpp. Três ajustes deverão ser providos na tela da interface:

* um ajuste para regular a altura da região central que entrará em foco;
* um ajuste para regular a força de decaimento da região borrada;
* um ajuste para regular a posição vertical do centro da região que entrará em foco. Finalizado o programa, a imagem produzida deverá ser salva em arquivo.

**Implementação**

~~~c++
#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>

double alfa;
int alfa_slider = 0;
int alfa_slider_max = 100;

int top_slider = 0;
int top_slider_max = 100;

int pos_slider = 0;
int pos_slider_max = 100;

cv::Mat image1, image2, blended;
cv::Mat imageTop;

char TrackbarName[50];

void on_trackbar_blend(int, void*){
 alfa = (double) alfa_slider/alfa_slider_max ;
 cv::addWeighted(image1, 1-alfa, imageTop, alfa, 0.0, blended);
 cv::imshow("addweighted", blended);
}

void on_trackbar_line(int, void*){
  image2.copyTo(imageTop);

  int h = image2.size().height;
  int w = image2.size().width;

  int limit_line = top_slider*h/100;
  int limit_pos  = pos_slider*h/100;

  if(limit_line > 0){
    if(limit_pos >= 0 && limit_pos <= h - limit_line){
        cv::Mat tmp = image1(cv::Rect(0, limit_pos, w, limit_line));
        tmp.copyTo(imageTop(cv::Rect(0, limit_pos, w, limit_line)));
    }else{
        cv::Mat tmp = image1(cv::Rect(0, 0, w, limit_line));
        tmp.copyTo(imageTop(cv::Rect(0, 0, w, limit_line)));
    }
  }
  on_trackbar_blend(alfa_slider,0);
}

int main(int argvc, char** argv){
  image1 = cv::imread("blend1.jpg");
  image1.copyTo(image2);
  cv::namedWindow("addweighted", 1);

  image2.convertTo(image2, CV_32F);
  float media[] = {0.111, 0.111, 0.111,
                 0.111, 0.111, 0.111,
                 0.111, 0.111, 0.111};

  cv::Mat mask;
  mask = cv::Mat(3, 3, CV_32F, media);

  for (int i = 0; i < 10; ++i) {
      filter2D(image2, image2, image2.depth(), mask, cv::Point(1,1), 0);
  }

  image2.convertTo(image2, CV_8U);
  image2.copyTo(imageTop);

  std::sprintf( TrackbarName, "Alpha x %d", alfa_slider_max );
  cv::createTrackbar( TrackbarName, "addweighted",
                      &alfa_slider,
                      alfa_slider_max,
                      on_trackbar_blend );
  on_trackbar_blend(alfa_slider, 0 );

  std::sprintf( TrackbarName, "height x %d", top_slider_max );
  cv::createTrackbar( TrackbarName, "addweighted",
                      &top_slider,
                      top_slider_max,
                      on_trackbar_line );
  on_trackbar_line(top_slider, 0 );

  std::sprintf( TrackbarName, "position x %d", top_slider_max );
  cv::createTrackbar( TrackbarName, "addweighted",
                      &pos_slider,
                      pos_slider_max,
                      on_trackbar_line );
  on_trackbar_line(pos_slider, 0 );

  cv::waitKey(0);
  return 0;
}
~~~

**Resultado**

![Image](images/ex6/ex6.png)

Como podemos observar a região do meio está mais nítida enquanto o restante está borrado.

## Tutorial 7
### Exemplo 7.1
Utilizando o programa exemplos/dft.cpp como referência, implemente o filtro homomórfico para melhorar imagens com iluminação irregular. Crie uma cena mal iluminada e ajuste os parâmetros do filtro homomórfico para corrigir a iluminação da melhor forma possível. Assuma que a imagem fornecida é em tons de cinza.

**Implementação**
~~~C++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

//variavais gamaLobais
double    gamaH, gamaL, c, d0;
int       gamaH_slider = 2, gamaL_slider = 2, c_slider = 1, d0_slider = 3;
const int gamaH_max = 100, gamaL_max = 100, c_max = 100, d0_max = 500;

//imagem original, imagem para o filtro e imagem filtrada, respectivamente
Mat original, padded, output;

// troca os quadrantes da imagem da DFT
void deslocaDFT(Mat& image) {
    Mat tmp, A, B, C, D;

    // se a imagem tiver tamanho impar, recorta a regiao para
    // evitar cópias de tamanho desigual
    image = image(Rect(0, 0, image.cols & -2, image.rows & -2));
    int cx = image.cols / 2;
    int cy = image.rows / 2;

    // reorganiza os quadrantes da transformada
    // A B   ->  D C
    // C D       B A
    A = image(Rect(0, 0, cx, cy));
    B = image(Rect(cx, 0, cx, cy));
    C = image(Rect(0, cy, cx, cy));
    D = image(Rect(cx, cy, cx, cy));

    // A <-> D
    A.copyTo(tmp);
    D.copyTo(A);
    tmp.copyTo(D);

    // C <-> B
    C.copyTo(tmp);
    B.copyTo(C);
    tmp.copyTo(B);
}

//funcao que cria o filtro
Mat filtro(double gamaL, double gamaH, double c, double d0) {

    int dft_M = padded.rows, dft_N = padded.cols;
    Mat filter = Mat(padded.size(), CV_32FC2, Scalar(0)), tmp = Mat(padded.size(), CV_32F);

    for (int i = 0; i < dft_M; i++) {
        for (int j = 0; j < dft_N; j++) {
            tmp.at<float>(i, j) = (gamaH - gamaL) * (1 - exp(-c * (((i - dft_M / 2) * (i - dft_M / 2) + (j - dft_N / 2) * (j - dft_N / 2)) / (d0 * d0)))) + gamaL;
        }
    }

    Mat comps[] = { tmp,tmp };
    merge(comps, 2, filter);
    normalize(tmp, tmp, 0, 1, NORM_MINMAX);
    imshow("filtro", tmp);

    return filter;
}

//funcao para aplicar o filtro
Mat filtragem() {
    //filtro e imagem complexa  
    Mat filter, cplx_img;
    //parte real da imagem e imagem totalmente preta
    Mat_<float> real_img, zeros;
    //plano real e imaginario da imagem
    vector<Mat> planos;

    //dimensoes para a dft
    int dft_M = getOptimalDFTSize(original.rows);
    int dft_N = getOptimalDFTSize(original.cols);

    //realiza o padding da imagem
    copyMakeBorder(original, padded, 0,
        dft_M - original.rows, 0,
        dft_N - original.cols,
        BORDER_CONSTANT, Scalar::all(0));

    //parte imaginaria da matriz complexa (preenchida com zeros)
    zeros = Mat_<float>::zeros(padded.size());
    //prepara a matriz complexa para ser preenchida
    cplx_img = Mat(padded.size(), CV_32FC2, Scalar(0));
    //o filtro deve ter mesma dimensao e tipo que a matriz complexa
    filter = cplx_img.clone();
    //cria a matriz com as componentes do filtro e junta ambas em uma matriz multicanal complexa
    Mat comps[] = { Mat(dft_M, dft_N, CV_32F), Mat(dft_M, dft_N, CV_32F) };
    merge(comps, 2, filter);
    //limpa o array de matrizes que vao compor a imagem complexa
    planos.clear();
    //cria a parte real
    real_img = Mat_<float>(padded);
    real_img += Scalar::all(1);
    log(real_img, real_img);
    //insere as duas partes (real e complexa) no array de matrizes
    planos.push_back(real_img);
    planos.push_back(zeros);
    //transforma as duas partes em uma unica componente complexa
    merge(planos, cplx_img);
    //dft
    dft(cplx_img, cplx_img);
    //troca os quadrantes
    deslocaDFT(cplx_img);
    resize(cplx_img, cplx_img, padded.size());
    normalize(cplx_img, cplx_img, 0, 1, NORM_MINMAX);

    //criamos o filtro
    filter = filtro(gamaH, gamaL, c, d0);
    //filtragem
    mulSpectrums(cplx_img, filter, cplx_img, 0);
    //reorganiza os quadrantes
    deslocaDFT(cplx_img);
    //transformada inversa
    idft(cplx_img, cplx_img);
    //clear no array de planos
    planos.clear();
    //separa as partes real e imaginaria da imagem filtrada
    split(cplx_img, planos);
    exp(planos[0], planos[0]);
    planos[0] -= Scalar::all(1);
    //normaliza a parte real 
    normalize(planos[0], planos[0], 0, 1, NORM_MINMAX);

    return planos[0];
}


//controla os parametros do filtro
void trackbar_move(int, void*) {
    gamaH = (double)gamaH_slider / 100.0;
    gamaL = (double)gamaL_slider / 20.0;
    c = (double)c_slider;
    d0 = (double)d0_slider;
    output = filtragem();
    imshow("output", output);
    normalize(output, output, 0, 255, NORM_MINMAX);
    imwrite("saida.png", output);
}


int main(int argc, char** argv) {
    char TrackbarName[50];
    original = imread(argv[1], IMREAD_GRAYSCALE);
    resize(original, original, cv::Size(600, 480));
    imshow("original", original);

    namedWindow("output", WINDOW_FREERATIO);

    sprintf(TrackbarName, "gamaH - %d", gamaH_max);
    createTrackbar(TrackbarName, "output", &gamaH_slider, gamaH_max, trackbar_move);

    sprintf(TrackbarName, "gamaL - %d", gamaL_max);
    createTrackbar(TrackbarName, "output", &gamaL_slider, gamaL_max, trackbar_move);

    sprintf(TrackbarName, "c - %d", c_max);
    createTrackbar(TrackbarName, "output", &c_slider, c_max, trackbar_move);

    sprintf(TrackbarName, "d0 - %d", d0_max);
    createTrackbar(TrackbarName, "output", &d0_slider, d0_max, trackbar_move);
    waitKey();

    return 0;
}
~~~

**Resultados**

![Image](images/ex7/Screenshot_20210825_165922.png)

Na esquerda temos a imagem original, bastante escura e difícil de visualizar algumas regiões e na direita a imagem após aplicação do filtro homomórfico.

## Tutorial 8
### Exemplo 8.1

Utilizando os programas exemplos/canny.cpp e exemplos/pontilhismo.cpp como referência, implemente um programa cannypoints.cpp. A idéia é usar as bordas produzidas pelo algoritmo de Canny para melhorar a qualidade da imagem pontilhista gerada.

**Implementação**

~~~C++
#include <numeric>
#include <ctime>
#include <cstdlib>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define STEP 2
#define JITTER 1

int slider = 1;
int slider_max = 200;

char TrackbarName[50];

Mat original, cinza, border,output;
int width, height;
Vec3b colors;
int x, y;
vector<int> yrange;
vector<int> xrange;

void on_trackbar_canny(int, void*){

  Canny(cinza, border, slider, 3*slider);
  imshow("bordas", border);
  
  output = Mat(height, width, CV_8UC3, Scalar(255,255,255));
  for(int i = 0; i < width; i++){
    for(int j = 0; j < height; j++){
      if(border.at<uchar>(j,i) == 255)
      {
          x = i+rand()%(2*JITTER)-JITTER+1;
          y = j+rand()%(2*JITTER)-JITTER+1;
          colors = original.at<Vec3b>(y,x);
          circle(output,
                 cv::Point(x,y),
                 1,
                 CV_RGB(colors[2],colors[1],colors[0]),
                 -1,
                 LINE_AA);
      }
    }
  }
  imshow("canny",output);

}

int main(int argc, char**argv){

  original= imread(argv[1],IMREAD_COLOR);

  cvtColor(original,cinza, COLOR_BGR2GRAY);

  srand(time(0));

  width=original.size().width;
  height=original.size().height;

  xrange.resize(height/STEP);
  yrange.resize(width/STEP);

  iota(xrange.begin(), xrange.end(), 0);
  iota(yrange.begin(), yrange.end(), 0);

  for(uint i=0; i<xrange.size(); i++){
    xrange[i]= xrange[i]*STEP+STEP/2;
  }

  for(uint i=0; i<yrange.size(); i++){
    yrange[i]= yrange[i]*STEP+STEP/2;
  }

  
  sprintf(TrackbarName,"Threshold_inferior-%d",slider_max);
  namedWindow("canny",1);
  createTrackbar( TrackbarName, "canny",
                &slider,
                slider_max,
                on_trackbar_canny );

  on_trackbar_canny(slider, 0);

  waitKey();
  imwrite("bordas.png",border);
  imwrite("cannypontos.png",output);
  return 0;
}
~~~

**Resultado**

Imagem orignal

![Image](images/ex8/praia.jpg)

Imagem pontilhada gerada sobre o canny

![Image](images/ex8/cannypontos.png)

## Tutorial 9
### Exemplo 9.1
Utilizando o programa kmeans.cpp como exemplo prepare um programa exemplo onde a execução do código se dê usando o parâmetro nRodadas=1 e inciar os centros de forma aleatória usando o parâmetro KMEANS_RANDOM_CENTERS ao invés de KMEANS_PP_CENTERS. Realize 10 rodadas diferentes do algoritmo e compare as imagens produzidas. Explique porque elas podem diferir tanto.

**Implementação**

~~~C++
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv ){
	if(argc!=2){
		exit(0);
	}

	int nClusters = 15;
	Mat rotulos;
	int nRodadas = 1;
	Mat centros;
	
	for(int i = 0; i < 10; i++) {
		Mat img = imread( argv[1], IMREAD_COLOR);
		Mat samples(img.rows * img.cols, 3, CV_32F);

		for( int y = 0; y < img.rows; y++ ){
			for( int x = 0; x < img.cols; x++ ){
		    	for( int z = 0; z < 3; z++){
		      		samples.at<float>(y + x*img.rows, z) = img.at<Vec3b>(y,x)[z];
		 		}
			}
		}

		kmeans(samples,
		 		nClusters,
		 		rotulos,
		 		TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS, 10000, 0.0001),
		 		nRodadas,
		 		KMEANS_RANDOM_CENTERS,
		 		centros );


		Mat rotulada( img.size(), img.type() );
		for( int y = 0; y < img.rows; y++ ){
		 	for( int x = 0; x < img.cols; x++ ){
		 		int indice = rotulos.at<int>(y + x*img.rows,0);
		 		rotulada.at<Vec3b>(y,x)[0] = (uchar) centros.at<float>(indice, 0);
		 		rotulada.at<Vec3b>(y,x)[1] = (uchar) centros.at<float>(indice, 1);
		 		rotulada.at<Vec3b>(y,x)[2] = (uchar) centros.at<float>(indice, 2);
			}
		}

		cout << "Gerando imagem " << i << endl;
		string number_img = to_string(i);
		string name_img = string(argv[1]);

		string name = number_img + "_" + name_img;
		
		imwrite(name, rotulada);
	}
}
~~~

**Resultados**
![Imagem](images/ex9/flores.gif)

O gif acima é um loop entre as 10 imagens gerados pelo código impletado acima. Podemos perceber uma diferença sutil entre as imagens, principalmente com cores mais vibrantes, como é o caso das flores laranja. A diferença ocorre porque além da função *kmeans* do opencv executar apenas uma vez, ou seja, em cada loop a imagem será rotulada apenas uma vez, o uso da *flag* **KMEANS_RANDOM_CENTERS** faz com que os centros sejam aleatórios, gerando assim uma pequena diferença na clusterização de cada pixel de forma que o resultado final de cada imagem possui diferenças entre si.