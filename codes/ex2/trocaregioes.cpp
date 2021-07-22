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
