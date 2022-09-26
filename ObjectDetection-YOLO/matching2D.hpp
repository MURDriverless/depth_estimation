#ifndef matching2D_hpp
#define matching2D_hpp

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdio.h>
#include <vector>

#include <opencv2/core.hpp>
#include "opencv2/features2d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "dataStructures.h"


// typedef cv::xfeatures2d::SIFT SiftFeatureDetector;

enum class DetectorTypeIndex
{
    FAST = 0,
    BRISK = 1,
    ORB = 2,
    AKAZE = 3,
    SIFT = 4,
    SHITOMASI = 5,
    HARRIS = 6
};

enum class DescriptorTypeIndex
{
    BRISK = 0,
    BRIEF = 1,
    ORB = 2,
    FREAK = 3,
    AKAZE = 4,
    SIFT = 5,
};

static const std::vector<std::string> detectorTypeString{
    "FAST", "BRISK", "ORB", "AKAZE", "SIFT", "SHITOMASI", "HARRIS"};

static const std::vector<std::string> descriptorTypeString{"BRISK", "BRIEF", "ORB",
                                                           "FREAK", "AKAZE", "SIFT"};

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints,
                        cv::Mat &img,
                        bool bVis = false);

void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints,
                           cv::Mat &img,
                           bool bVis = false);

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints,
                        cv::Mat &img,
                        DetectorTypeIndex detectorTypeIndex,
                        bool bVis = false);

void descKeypoints(std::vector<cv::KeyPoint> &keypoints,
                   cv::Mat &img,
                   cv::Mat &descriptors,
                   DescriptorTypeIndex descriptorTypeIndex);

void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                      std::vector<cv::KeyPoint> &kPtsRef,
                      cv::Mat &descSource,
                      cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches,
                      std::string descriptorType,
                      std::string matcherType,
                      std::string selectorType);

DetectorTypeIndex getDetectorTypeIndex(std::string &detectorType);
const std::string &getDetectorTypeString(DetectorTypeIndex detectorTypeIndex);

DescriptorTypeIndex getDescriptorTypeIndex(std::string &descriptorType);
const std::string &getDescriptorTypeString(DescriptorTypeIndex descriptorTypeIndex);

void removeKeypointsOutsideBox(cv::Rect vehicleRect,
                               std::vector<cv::KeyPoint> &keypoints,
                               std::vector<cv::KeyPoint> &keypointsROI);

#endif /* matching2D_hpp */
