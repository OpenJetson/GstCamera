#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void send_mjpeg(cv::Mat mat, int port, int timeout, int quality);
