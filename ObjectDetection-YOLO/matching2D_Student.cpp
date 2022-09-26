#include "matching2D.hpp"
#include <numeric>

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                      std::vector<cv::KeyPoint> &kPtsRef,
                      cv::Mat &descSource,
                      cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches,
                      std::string descriptorType,
                      std::string matcherType,
                      std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {
            /*
             * OpenCV bug workaround : convert binary descriptors to floating point due to
             * a bug in current OpenCV implementation
             */
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    {
        // nearest neighbor (best match)
        matcher->match(descSource, descRef,
                       matches);  // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        // k nearest neighbors (k=2)
        int k = 2;

        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);

        // filter matches using descriptor distance ratio test 0.8
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints,
                   cv::Mat &img,
                   cv::Mat &descriptors,
                   DescriptorTypeIndex descriptorTypeIndex)
{
    cv::Ptr<cv::DescriptorExtractor> extractor;
    switch (descriptorTypeIndex)
    {
        case DescriptorTypeIndex::BRISK:
        {
            extractor = cv::BRISK::create();
            break;
        }
        case DescriptorTypeIndex::BRIEF:
        {
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
            break;
        }
        case DescriptorTypeIndex::ORB:
        {
            extractor = cv::ORB::create();
            break;
        }
        case DescriptorTypeIndex::FREAK:
        {
            extractor = cv::xfeatures2d::FREAK::create();
            break;
        }
        case DescriptorTypeIndex::AKAZE:
        {
            extractor = cv::AKAZE::create();
            break;
        }
        case DescriptorTypeIndex::SIFT:
        {
            extractor = cv::xfeatures2d::SIFT::create();
            break;
        }
        default:
        {
            throw invalid_argument("Invalid detector type");
        }
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << getDescriptorTypeString(descriptorTypeIndex) << " descriptor extraction in "
         << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;        //  size of an average block for computing a derivative
                              //  covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0;  // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners =
        img.rows * img.cols / max(1.0, minDistance);  // max. num. of keypoints

    double qualityLevel = 0.01;  // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance,
                            cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in "
         << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize =
        2;  // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse =
        100;          // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;  // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Look for prominent corners and instantiate keypoints
    double maxOverlap = 0.0;  // max. permissible overlap between two features in %, used
                              // during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            {  // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new
                // key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {  // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint;  // replace old key point with new one
                            break;              // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {  // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint);  // store new keypoint in dynamic
                                                       // list
                }
            }
        }  // eof loop over cols
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris corner detection with n=" << keypoints.size() << " keypoints in "
         << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 7);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints,
                        cv::Mat &img,
                        DetectorTypeIndex detectorTypeIndex,
                        bool bVis)
{
    double t = (double)cv::getTickCount();

    switch (detectorTypeIndex)
    {
        case DetectorTypeIndex::BRISK:
        {
            cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
            detector->detect(img, keypoints);
            break;
        }
        case DetectorTypeIndex::FAST:
        {
            cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect(img, keypoints);
            break;
        }
        case DetectorTypeIndex::ORB:
        {
            cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
            detector->detect(img, keypoints);
            break;
        }
        case DetectorTypeIndex::AKAZE:
        {
            cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
            detector->detect(img, keypoints);
            break;
        }
        case DetectorTypeIndex::SIFT:
        {
            cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
            detector->detect(img, keypoints);
            break;
        }
        case DetectorTypeIndex::SHITOMASI:
        case DetectorTypeIndex::HARRIS:
        default:
        {
            throw invalid_argument("Invalid detector type");
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << getDetectorTypeString(detectorTypeIndex) + " detection with n="
         << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName =
            getDetectorTypeString(detectorTypeIndex) + " Detector Results";
        cv::namedWindow(windowName, static_cast<int>(detectorTypeIndex) + 1);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

DetectorTypeIndex getDetectorTypeIndex(std::string &detectorType)
{
    if (detectorType.compare("FAST") == 0)
    {
        return DetectorTypeIndex::FAST;
    }

    if (detectorType.compare("BRISK") == 0)
    {
        return DetectorTypeIndex::BRISK;
    }

    if (detectorType.compare("ORB") == 0)
    {
        return DetectorTypeIndex::ORB;
    }

    if (detectorType.compare("AKAZE") == 0)
    {
        return DetectorTypeIndex::AKAZE;
    }

    if (detectorType.compare("SIFT") == 0)
    {
        return DetectorTypeIndex::SIFT;
    }

    if (detectorType.compare("SHITOMASI") == 0)
    {
        return DetectorTypeIndex::SHITOMASI;
    }

    if (detectorType.compare("HARRIS") == 0)
    {
        return DetectorTypeIndex::HARRIS;
    }

    throw invalid_argument("Invalid detector type");
}

const std::string &getDetectorTypeString(DetectorTypeIndex detectorTypeIndex)
{
    try
    {
        return detectorTypeString[static_cast<int>(detectorTypeIndex)];
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        throw invalid_argument("Invalid detector type index");
    }
}

DescriptorTypeIndex getDescriptorTypeIndex(std::string &descriptorType)
{
    if (descriptorType.compare("BRISK") == 0)
    {
        return DescriptorTypeIndex::BRISK;
    }

    if (descriptorType.compare("BRIEF") == 0)
    {
        return DescriptorTypeIndex::BRIEF;
    }

    if (descriptorType.compare("ORB") == 0)
    {
        return DescriptorTypeIndex::ORB;
    }

    if (descriptorType.compare("FREAK") == 0)
    {
        return DescriptorTypeIndex::FREAK;
    }

    if (descriptorType.compare("AKAZE") == 0)
    {
        return DescriptorTypeIndex::AKAZE;
    }

    if (descriptorType.compare("SIFT") == 0)
    {
        return DescriptorTypeIndex::SIFT;
    }

    throw invalid_argument("Invalid detector type");
}

const std::string &getDescriptorTypeString(DescriptorTypeIndex descriptorTypeIndex)
{
    try
    {
        return descriptorTypeString[static_cast<int>(descriptorTypeIndex)];
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        throw invalid_argument("Invalid descriptor type index");
    }
}

void removeKeypointsOutsideBox(cv::Rect vehicleRect,
                               std::vector<cv::KeyPoint> &keypoints,
                               std::vector<cv::KeyPoint> &keypointsROI)
{
    for (auto &keypoint : keypoints)
    {
        if (vehicleRect.contains(keypoint.pt))
        {
            keypointsROI.push_back(keypoint);
        }
    }
}