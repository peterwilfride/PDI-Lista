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
