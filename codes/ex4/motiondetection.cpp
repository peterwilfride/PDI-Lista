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

