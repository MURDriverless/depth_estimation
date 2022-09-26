#pragma once

#include "ConeColorID.hpp"
#include <opencv2/core.hpp>

// cone region of interest contains the roiRect(?), it's (x,y) coordinates, width, w, and height, h, of the region of interest 
// a vector of it's keypoints in (x,y) coordinators
// and the colour of the cone 
struct ConeROI {
    cv::Rect roiRect;
    float x, y, w, h;
    std::vector<cv::Point2f> keypoints;

    ConeColorID colorID;
};