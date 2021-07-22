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
