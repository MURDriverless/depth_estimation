#pragma once

#include "ConeColorID.hpp"
#include <opencv2/core.hpp>

struct ConeEst {
    cv::Point3f pos;

    ConeColorID colorID;
};