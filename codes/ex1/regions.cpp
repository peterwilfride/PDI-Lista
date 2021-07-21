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
