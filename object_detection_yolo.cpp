// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

// solvePnP is listed in this header file! 
#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core.hpp>

#include "ConeROI.hpp"
#include "ConeEst.hpp"

#include "dataStructures.h"
#include "matching2D.hpp"

// add custom keys if you want to use the parser to your liking 
const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
"{device d       |<cpu>| input device   }"
"{is_left l     |<none>| input int      }"
;
using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;
vector<string> classes_right; 

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, Mat& frame_right, 
const vector<Mat>& outs_right, int is_left, Mat cameraMatrix, Mat distCoeffs);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, float zEst, float xEst, float yEst);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

// void featureDetectorExperimentation()
// {
//     // source: https://stackoverflow.com/questions/39887357/featuredetectors-not-in-opencv-3-0-0 
//     cv::Mat image = imread("bird.jpg");

//     std::vector<KeyPoint> keypoints;
//     cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(40);
//     // Above line compiler error: "Error    1   error C2259: 'cv::FastFeatureDetector' : cannot instantiate abstract class"

//     fast->detect(image, keypoints);

//     drawKeypoints(image, keypoints, image, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);

//     imshow("Image", image);
//     waitKey(1000);
// }


int main(int argc, char** argv)
{   

    const std::string& calibrationFile = "left_calibration.xml";
    cv::FileStorage fs;
    cv::Size calibSize;
    cv::Mat cameraMatrix; 
    cv::Mat distCoeffs; 

    fs.open(calibrationFile, cv::FileStorage::READ);
    fs["cameraMatrix"]   >> cameraMatrix;
    fs["distCoeffs"]     >> distCoeffs;
    fs["calibImageSize"] >> calibSize;

    float imgCenter_x = cameraMatrix.at<double>(0, 2);
    float imgCenter_y = cameraMatrix.at<double>(1, 2);

    float focal_px_x = cameraMatrix.at<double>(0, 0);
    float focal_px_y = cameraMatrix.at<double>(1, 1);

    fs.release();

    // cout << imgCenter_x << endl;
    // cout << focal_px_y << endl;

    // cout << distCoeffs.at<double>(0, 4) << endl;

    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // Load names of classes
    // string classesFile = "coco.names";
    
    string classesFile = "cones.names"; // use the list of class names for our cones i.e., blue, orange, and yellow 
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    string device = "cpu";
    device = parser.get<String>("device");
    
    // Give the configuration and weight files for the model
    // String modelConfiguration = "yolov3.cfg";
    // String modelWeights = "yolov3.weights";

    String modelConfiguration = "yolov4-tiny.cfg";
    String modelWeights = "yolov4-tiny-best.weights";

    int left_cam_only = 0;

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

    Net net_right = readNetFromDarknet(modelConfiguration, modelWeights);

    if (device == "cpu")
    {
        cout << "Using CPU device" << endl;
        net.setPreferableBackend(DNN_TARGET_CPU);
    }
    // else if (device == "gpu")
    // {
    //     cout << "Using GPU device" << endl;
    //     // net.setPreferableBackend(DNN_BACKEND_CUDA);
    //     // net.setPreferableTarget(DNN_TARGET_CUDA);
    
    // // commented out because this part of the code doesn't work.
    //     net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
    //     net.setPreferableTarget(dnn::DNN_TARGET_CUDA);

    // }

    
    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;
    int FPS = 10;
    
    string str_right, outputFile_right;
    VideoCapture cap_right;
    VideoWriter video_right;
    Mat frame_right, blob_right;

    int is_left = 0;

    // if (parser.has("is_left")) {
    //     str = parser.get<String>("is_left");
    //     cout << str << endl;
    //     is_left = stoi(str);
    //     left_cam_only = is_left; 
    // }
    try {
        
        outputFile = "yolo_out_cpp.avi";
        if (parser.has("image"))
        {
            // Open the image file
            str = parser.get<String>("image");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.jpg");
            outputFile = str;
        }
        else if (parser.has("video"))
        {
            // Open the video file (left camera)
            str = parser.get<String>("video");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.avi");
            outputFile = str;

            // right camera
            if (left_cam_only == 0) {
                str_right = parser.get<String>("video");
                ifstream ifile_right(str_right);
                if (!ifile_right) throw("error");
                cap_right.open(str_right);
                str_right.replace(str_right.end()-4, str_right.end(), "_yolo_out_cpp.avi");
                outputFile_right = str_right;
            }
            
        }
        // Open the webcam
        else cap.open(parser.get<int>("device"));
        
    }
    catch(...) {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }
    
    // Get the video writer initialized to save the output video
    if (!parser.has("image")) {
        video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), FPS, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
        
        if (left_cam_only == 0) {
            video_right.open(outputFile_right, VideoWriter::fourcc('M','J','P','G'), FPS, Size(cap_right.get(CAP_PROP_FRAME_WIDTH), cap_right.get(CAP_PROP_FRAME_HEIGHT)));
        }
    }
    
    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Process frames.
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;

        if (left_cam_only == 0) {
            cap_right >> frame_right;
        }

        // Stop the program if reached end of video
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputFile << endl;
            waitKey(3000);
            break;
        }
        // Create a 4D blob from a frame.
        blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
                
        //Sets the input to the network 
        // anything to do with "net" means it's dealing with darknet 
        net.setInput(blob);
        
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        vector<Mat> outs_right;
        
        net.forward(outs, getOutputsNames(net));
        
        if (left_cam_only == 0) {
            blobFromImage(frame_right, blob_right, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
            net_right.setInput(blob_right);
            net_right.forward(outs_right, getOutputsNames(net_right));
        }
        // Remove the bounding boxes with low confidence
        cout << "before postprocessing" << endl;

        postprocess(frame, outs, frame_right, outs_right, is_left, cameraMatrix, distCoeffs);
        
        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        
        // Write the frame with the detection boxes
        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        if (parser.has("image")) imwrite(outputFile, detectedFrame);
        else video.write(detectedFrame);
        
        imshow(kWinName, frame);
        
    }
    
    cap.release();
    cap_right.release();

    if (!parser.has("image")) {
        video.release();
        video_right.release();
    } 

    return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, Mat& frame_right, 
const vector<Mat>& outs_right, int is_left, Mat cameraMatrix, Mat distCoeffs)
{   

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    cout << outs.size() << endl;
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    // added by Kelvin 
    double baseline = 550; // 550mm 
    double f2 = 1499.093; // focal length of right camera = 5mm (in the x axis)
    double f1 = 2724.847; // focal length of left camera in the x axis

    double lImgCenter_x = 548.607;
    double lImgCenter_y = 845.6168;

    double rImgCenter_x = 921.35846;
    double rImgCenter_y = 616.6407;

    cout << "after rImgCenter" << endl;

    std::vector<std::vector<cv::Point3f>> conePointsVec; // empty vector of vectors after all 

    std::vector<cv::Point3f> conePoints;
    conePoints.push_back(cv::Point3f(0, 0, 0));
    for (int i = 1; i <= 3; i++) {
        float x = -77.5/3.0f * i;
        float y = 300.0f/3.0f * i;

        conePoints.push_back(cv::Point3f( x, y, 0));
        conePoints.push_back(cv::Point3f(-x, y, 0));
    }
    std::vector<cv::Point3f> conePointsBig;
    conePointsBig.push_back(cv::Point3f(0, 0, 0));
    for (int i = 1; i <= 3; i++) {
        float x = -77.5/3.0f * i;
        float y = 505.0f/3.0f * i;

        conePointsBig.push_back(cv::Point3f( x, y, 0));
        conePointsBig.push_back(cv::Point3f(-x, y, 0));
    }

    conePointsVec.push_back(conePoints);
    conePointsVec.push_back(conePointsBig);

    // cv::Mat roi(frame, boxes[0]);

    cv::Mat imgGray;
    cv::cvtColor(frame, imgGray, cv::COLOR_BGR2GRAY);

    string detectorType = "SIFT";
    bool visDetector = true;
  
    vector<cv::KeyPoint> keypoints;

    DetectorTypeIndex detectorTypeIndex = getDetectorTypeIndex(detectorType);

    switch (detectorTypeIndex)
        {
            case DetectorTypeIndex::SHITOMASI:
            {
                detKeypointsShiTomasi(keypoints, imgGray, visDetector);
                break;
            }
            case DetectorTypeIndex::HARRIS:
            {
                detKeypointsHarris(keypoints, imgGray, visDetector);
                break;
            }
            case DetectorTypeIndex::FAST:
            case DetectorTypeIndex::BRISK:
            case DetectorTypeIndex::ORB:
            case DetectorTypeIndex::AKAZE:
            case DetectorTypeIndex::SIFT:
            {
                detKeypointsModern(keypoints, imgGray, detectorTypeIndex, visDetector);
                break;
            }
            default:
            {
                throw invalid_argument("Invalid detector type");
            }
        }
    
    // cv::Mat imgGray_right;

    // cv::Mat roi_right(frame_right, boxes[0]);

    // cv::Mat imgGray;
    // cv::cvtColor(roi, imgGray, cv::COLOR_BGR2GRAY);

    // cv::cvtColor(frame_right, imgGray_right, cv::COLOR_BGR2GRAY);

    // vector<cv::KeyPoint> keypoints_right;

    // switch (detectorTypeIndex)
    //     {
    //         case DetectorTypeIndex::SHITOMASI:
    //         {
    //             detKeypointsShiTomasi(keypoints_right, imgGray_right, visDetector);
    //             break;
    //         }
    //         case DetectorTypeIndex::HARRIS:
    //         {
    //             detKeypointsHarris(keypoints_right, imgGray_right, visDetector);
    //             break;
    //         }
    //         case DetectorTypeIndex::FAST:
    //         case DetectorTypeIndex::BRISK:
    //         case DetectorTypeIndex::ORB:
    //         case DetectorTypeIndex::AKAZE:
    //         case DetectorTypeIndex::SIFT:
    //         {
    //             detKeypointsModern(keypoints_right, imgGray_right, detectorTypeIndex, visDetector);
    //             break;
    //         }
    //         default:
    //         {
    //             throw invalid_argument("Invalid detector type");
    //         }
    //     }

    //*************************************************************************************************
    // keypoint regression 

    // generate a vector of image crops for keypoint detector
    // std::vector<cv::Mat> rois;

    // for (const auto &bbox: bboxs)
    // {
    //     ConeROI coneROI;

    //     int left    = std::max<float>(float(bbox.x), 0.0f);
    //     int right   = std::min<float>(float(bbox.x + bbox.w), (float) imageFrame.cols);
    //     int top     = std::max<float>(float(bbox.y), 0.0f);
    //     int bot     = std::min<float>(float(bbox.y + bbox.h), (float) imageFrame.rows);

    //     cv::Rect box(cv::Point(left, top), cv::Point(right, bot));
    //     cv::Mat roi = imageFrame(box);
    //     rois.push_back(roi);

    //     coneROI.roiRect = box;
    //     coneROI.x = bbox.x;
    //     coneROI.y = bbox.y;
    //     coneROI.w = bbox.w;
    //     coneROI.h = bbox.h;

    //     coneROIs.push_back(coneROI);
    // }

    // keypoint network inference
    // std::vector<std::vector<cv::Point2f>> keypoints = keypointDetector->doInference(rois);

    // if (previewArgs.valid) {
    //     detNN->draw(batch_frame);
    // }
    // for (int i = 0; i < bboxs.size(); i++) {
    //     for (int j = 0; j < keypoints[i].size(); j++) {
    //         cv::Point2f &keypoint = keypoints[i][j];
    //         keypoint.y += bboxs[i].y;
    //         keypoint.x += bboxs[i].x;

    //         if (previewArgs.valid) {
    //             cv::circle(batch_frame[0], keypoint, 3, cv::Scalar(0, 255, 0), -1, 8);
    //         }

    //         coneROIs[i].keypoints.push_back(keypoint);
    //         coneROIs[i].colorID = static_cast<ConeColorID>(bboxs[i].cl);
    //     }

    //****************************************************************************************************************
    // std::unique_ptr<KeypointDetector> keypointDetector;

    // keypointDetector.reset(new KeypointDetector("keypoint.onnx", "keypoint.trt", 80, 80, 100));
    // std::vector<std::vector<cv::Point2f>> keypoints = keypointDetector->doInference(boxes);
    // not going to work because I don't have tensorRT and it's a hassle to get last year's code to work. 

    // template code
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        // need to add keypoint regression to get the keypoints 
        // these keypoints help feature matching
        // need feature matching to find the corresponding object in the right image
        // then we can calculate the disparity error
        Mat tvec;
        Mat rvec;

        double est_depth = 0;

        conePoints = conePointsVec[0]; // 0 for blue and yellow cones <-- this could be the error. 

        const std::vector<cv::Point3f> &conePts = conePoints;

        // this is the problem. Need the keypoints from the keypoint regression code. 
        // const std::vector<cv::Point2f> keypoints; 

        cout << "before solvePNP" << endl;

        // ***************** edit from the SIFT code that I borrowed from Github ******************
        // cv::Mat imgGray;
        // cv::cvtColor(frame, imgGray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> keypoints_points;

        cv::KeyPoint::convert(keypoints, keypoints_points);

        bool ret = cv::solvePnP(conePts, keypoints_points, cameraMatrix, distCoeffs, rvec, tvec, false, SolvePnPMethod::SOLVEPNP_ITERATIVE);

         // source: https://stackoverflow.com/questions/45467927/camera-pose-estimation-using-opencv-c-solvepnp-function 
        // solvePnP(front_object_pts, front_image_pts, cameraMatrix_Front,
        //  Mat(4,1,CV_64FC1,Scalar(0)), rvec_front, tvec_front, false, CV_ITERATIVE);

        // this line won't work till the solvePnP line works. sigh. 
        // est_depth = tvec.at<double>(2, 0);

        cout << "after solvingPNP" << endl;
        // use the estimated area of where the right box might be to get the actual coordinates 
        // Rect box_right = ;

        // std::vector<cv::KeyPoint> featureKeypoints1;
        // std::vector<cv::KeyPoint> featureKeypoints2;
        // cv::Mat descriptors1;
        // cv::Mat descriptors2;

        // std::vector<cv::DMatch> matches;
        // std::vector<cv::DMatch> matchesFilt;

        // // cv::Ptr<cv::Feature2D> featureDetector;
        // cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;

        // // featureDetector->detectAndCompute(box, cv::noArray(), featureKeypoints1, descriptors1);
        
        // // new code since the above line doesn't work 
        // // Define features detector
        // // cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(10, true);
        // // // Detect the keypoints
        // // std::vector<cv::KeyPoint> keypoints1, keypoints2;
        // // detector->detect(box, keypoints1);
        // // detector->detect(im2, keypoints2);
        // // // Compute the keypoints descriptors
        // // cv::Mat descriptors1, descriptors2;
        // // cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create(32);
        // // brief->compute(im1, keypoints1, descriptors1);
        // // brief->compute(im2, keypoints2, descriptors2);

        // // featureDetector->detectAndCompute(box_right, cv::noArray(), featureKeypoints2, descriptors2);
        
        // cv::Ptr<cv::FastFeatureDetector> FeatureDetector = cv::FastFeatureDetector::create(40);

        // // frame = imread("bird.jpg");

        // FeatureDetector->detect(frame, featureKeypoints1);
        // FeatureDetector->detect(frame_right, featureKeypoints2);
        
        // cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create(32);

        // brief->compute(frame, featureKeypoints1, descriptors1);
        // brief->compute(frame_right, featureKeypoints2, descriptors2);


        // // drawKeypoints(frame, featureKeypoints1, frame, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);

        // // imshow("Image", frame);
        // // waitKey(1000);

        // // ********************************************************************************************************************
        // // /*
        // // No descriptor in left or right frame, either due to insufficient light.
        // // Or plain ground textures, expecially in synthetic data
        // // */
        // if (descriptors1.empty() || descriptors2.empty()) {
        //     continue;
        // }

        // descriptorMatcher->match(descriptors1, descriptors2, matches);

        // // *******************************************************************************
        // // estimated rectangle in the right image 
        // // box is the bounding box for the left image 
        // cv::Rect projRect(box);

        // int x_p = box.x - lImgCenter_x;
        // int x_pp = (f2/est_depth) * (est_depth/f1 * x_p - baseline);

        // projRect.x = x_pp + rImgCenter_x;

        // // Bounds checking
        // // Should we implement reshaping if out of bounds?
        // // if (!(0 <= projRect.x && projRect.x + projRect.width < rFrame.cols)) {
        // //     continue;
        // // }

        // // if (!(0 <= projRect.y && projRect.y + projRect.height < rFrame.rows)) {
        // //     continue;
        // // }


        // // // Filters for horizontal-ish matches only
        // uint32_t yDelta = projRect.height * 0.1;
        // for (const cv::DMatch &match : matches) {
        //     if (abs(featureKeypoints1[match.queryIdx].pt.y - featureKeypoints2[match.trainIdx].pt.y) < yDelta) {
        //         matchesFilt.push_back(match);
        //     }
        // }

        // // Check if no valid matches
        // if (matchesFilt.size() == 0) {
        //     continue;
        // }

        // std::vector<float> disparity;
        // for (const cv::DMatch &match : matchesFilt) {
        //     float x1 = featureKeypoints1[match.queryIdx].pt.x;
        //     x1 += box.x;
        //     x1 -= lImgCenter_x;

        //     float x2 = featureKeypoints2[match.trainIdx].pt.x;
        //     x2 += projRect.x;
        //     x2 -= rImgCenter_x;

        //     disparity.push_back(x1*f2/f1 - x2);
        // }

        // // // Performance loss from not using a sorted heap should be negligable
        // std::sort(disparity.begin(), disparity.end());

        // float medDisp = disparity[(int) disparity.size()/2];
        // float zEst = baseline*f2/medDisp;

        // added by Kelvin 
        // disparity = (left_box.x - lImgCenter_x)*f2/f1 - (right_box.x - rImgCenter_x);
        // float zEst = _baseline*f2/disparity;
        float zEst = 4400; // 4400mm
        float xEst = 0;
        float yEst = 0;

        // if (is_left == 1) {
            // left image 
            xEst = zEst*(box.x + box.width/2 - lImgCenter_x)/f1; // Andrew's code has a minus sign in front of zEst
            yEst = zEst*(box.y + box.height - lImgCenter_y)/f1; // Andrew's code has a minus sign in front of zEst
        // }
        // else {
            // right image 
            xEst = zEst*(box.x + box.width/2 - rImgCenter_x)/f2; // Andrew's code has a minus sign in front of zEst
            yEst = zEst*(box.y + box.height - rImgCenter_y)/f2; // Andrew's code has a minus sign in front of zEst
        // }
        // convert to m
        zEst = zEst/1000; 
        xEst = xEst/1000;
        yEst = yEst/1000;

        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame, zEst, xEst, yEst);
    }
}

//***************************************************************************
// notes 
    // fs.open(calibrationFile, cv::FileStorage::READ);
    // fs["cameraMatrix"]   >> cameraMatrix;
    // fs["distCoeffs"]     >> distCoeffs;
    // fs["calibImageSize"] >> calibSize;

    // imgCenter_x = cameraMatrix.at<double>(0, 2);
    // imgCenter_y = cameraMatrix.at<double>(1, 2);

    // focal_px_x = cameraMatrix.at<double>(0, 0);
    // focal_px_y = cameraMatrix.at<double>(1, 1);

// the intrinsic parameters of a camera is written as:
// [focal_px_x 0 img_center_x; 0 focal_px_y img_center_y; 0 0 1]
//******************************************************************************

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, float zEst, float xEst, float yEst)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    string conePoseEst = format("(%.2f, %.2f, %.2f)m", xEst, yEst, zEst);

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    Size conePoseEstSize = getTextSize(conePoseEst, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);

    // print cone pose estimation 
    rectangle(frame, Point(left, bottom + round(2*conePoseEstSize.height)), Point(left + round(1.5*conePoseEstSize.width), bottom + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, conePoseEst, Point(left, bottom + round(2*conePoseEstSize.height)), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }

    // for (int i = 0; names.size(); i++) {
    //     cout << names[i] << endl;
    // }
    
    return names;
}

//*****************************************************************************************************************
// Andrew's code for cone pose estimation 



// the cones' region of interest (ROI) are passed in as an input. So this method already knows the bounding boxes of the cones. 
// how does it detect the colour of the cones?

// void estConePos(const cv::Mat& lFrame, const cv::Mat& rFrame, 
//                 const std::vector<ConeROI>& coneROIs, std::vector<ConeEst>& coneEsts, 
//                 int lastFrame) {


//     // for all the detected cones
//     for (int i = 0; i < coneROIs.size(); i++) {
//         const ConeROI& coneROI = coneROIs[i];
//         cv::Mat tvec;
//         cv::Mat rvec;

//         double est_depth;

//         // this switch statement shows that the coneROI already detected the colour of the cones 
//         // Logic for switching pts, currently a STUB
//         // Kelvin: will need to upgrade this later. Just a temporary solution.
//         int conePtsID = -1;
//         switch (coneROI.colorID) {
//             case ConeColorID::Blue :
//                 conePtsID = 0;
//                 break;
//             case ConeColorID::Yellow :
//                 conePtsID = 0;
//                 break;
//             case ConeColorID::Orange :
//                 conePtsID = 1;
//                 break;
//         };

//         std::vector<std::vector<cv::Point3f>> conePointsVec; 
//         auto& conePoints = conePointsVec[conePtsID];

//         #ifdef CONE4
//             std::vector<cv::Point3f> conePts (conePoints.begin()+1, conePoints.end()-2);
//             std::vector<cv::Point2f> keyPts  (coneROI.keypoints.begin()+1, coneROI.keypoints.end()-2);
//         #else
//             const std::vector<cv::Point3f> &conePts = conePoints;
//             const std::vector<cv::Point2f> &keyPts  = coneROI.keypoints;
//         #endif

//         // TODO: Need to process the return
//         // bool ret = cv::solvePnP(conePts, keyPts, lCamParams.cameraMatrix, lCamParams.distCoeffs, rvec, tvec, false, cv::SolvePnPMethod::SOLVEPNP_IPPE);
        
//         if (true) {
//             est_depth = tvec.at<double>(2, 0);

//             // Using reference shouldnt cause performance degredation?
//             // const double &f1 = lCamParams.focal_px_x;
//             // const double &f2 = rCamParams.focal_px_x;
//             const double &f1 = 5.6;
//             const double &f2 = 5.6;

//             // const double &lImgCenter_x = lCamParams.imgCenter_x;
//             // const double &rImgCenter_x = rCamParams.imgCenter_x;
//             const double &lImgCenter_x = 960;
//             const double &rImgCenter_x = 960;

//             // const double &rImgCenter_y = rCamParams.imgCenter_y;
//             const double &rImgCenter_y = 600;

//             // TODO? : Enlarge borders around cones?
//             // float border = 0.0f;
//             // coneROI.roiRect -= cv::Point2i(border * coneROI.roiRect.width, border * coneROI.roiRect.height);
//             // coneROI.roiRect += cv::Size2i(2*border * coneROI.roiRect.width, 2*border * coneROI.roiRect.height);

//             cv::Rect projRect(coneROI.roiRect);

//             int x_p = coneROI.roiRect.x - lImgCenter_x;
//             int x_pp = (f2/est_depth) * (est_depth/f1 * x_p - _baseline);

//             projRect.x = x_pp + rImgCenter_x;

//             // projRect.x -= coneROI.roiRect.width * border;
//             // projRect.y -= coneROI.roiRect.height * border;
//             // projRect.width *= 1.0f + 2*border;
//             // projRect.height *= 1.0f + 2*border;

//             // Bounds checking
//             // Should we implement reshaping if out of bounds?
//             if (!(0 <= projRect.x && projRect.x + projRect.width < rFrame.cols)) {
//                 continue;
//             }

//             if (!(0 <= projRect.y && projRect.y + projRect.height < rFrame.rows)) {
//                 continue;
//             }

//             cv::Mat unDist1_cropped = lFrame(coneROI.roiRect);
//             cv::Mat unDist2_cropped = rFrame(projRect);

//             std::vector<cv::KeyPoint> featureKeypoints1;
//             std::vector<cv::KeyPoint> featureKeypoints2;
//             cv::Mat descriptors1;
//             cv::Mat descriptors2;

//             std::vector<cv::DMatch> matches;
//             std::vector<cv::DMatch> matchesFilt;

//             featureDetector->detectAndCompute(unDist1_cropped, cv::noArray(), featureKeypoints1, descriptors1);
//             featureDetector->detectAndCompute(unDist2_cropped, cv::noArray(), featureKeypoints2, descriptors2);

//             /*
//             No descriptor in left or right frame, either due to insufficient light.
//             Or plain ground textures, expecially in synthetic data
//             */
//             if (descriptors1.empty() || descriptors2.empty()) {
//                 continue;
//             }

//             descriptorMatcher->match(descriptors1, descriptors2, matches);

//             // Filters for horizontal-ish matches only
//             uint32_t yDelta = projRect.height * 0.1;
//             for (const cv::DMatch &match : matches) {
//                 if (abs(featureKeypoints1[match.queryIdx].pt.y - featureKeypoints2[match.trainIdx].pt.y) < yDelta) {
//                     matchesFilt.push_back(match);
//                 }
//             }

//             // Check if no valid matches
//             if (matchesFilt.size() == 0) {
//                 continue;
//             }

//             std::vector<float> disparity;
//             // taking the difference in x1 and x2 where disparity = x1 - x2 (when f2 == f1)
//             for (const cv::DMatch &match : matchesFilt) {
//                 float x1 = featureKeypoints1[match.queryIdx].pt.x;
//                 x1 += coneROI.roiRect.x;
//                 x1 -= lImgCenter_x;

//                 float x2 = featureKeypoints2[match.trainIdx].pt.x;
//                 x2 += projRect.x;
//                 x2 -= rImgCenter_x;

//                 disparity.push_back(x1*f2/f1 - x2);
//             }

//             // Performance loss from not using a sorted heap should be negligable
//             std::sort(disparity.begin(), disparity.end());

//             // median disparity 
//             float medDisp = disparity[(int) disparity.size()/2];
//             float zEst = _baseline*f2/medDisp;
//             float xEst = -zEst*(coneROI.roiRect.x + coneROI.roiRect.width/2 - rImgCenter_x)/f1;
//             float yEst = -zEst*(coneROI.roiRect.y + coneROI.roiRect.height - rImgCenter_y)/f1;

//             if (abs(est_depth - zEst) > 1500) {
//                 continue;
//             }

//             if (zEst < 0) {
//                 continue;
//             }

//             ConeEst coneEst;
//             coneEst.pos.x = xEst;
//             coneEst.pos.y = yEst;
//             coneEst.pos.z = zEst;
//             coneEst.colorID = coneROI.colorID;

//             coneEsts.push_back(coneEst);

//             if (previewArgs.valid) {
//                 cv::rectangle(*(previewArgs.rFrameBBoxMatPtr), projRect, cv::Scalar(255, 255, 255));

//                 if (i == 0) {
//                     cv::drawMatches(unDist1_cropped, featureKeypoints1, unDist2_cropped, featureKeypoints2, matchesFilt, *(previewArgs.matchesMatPtr));
//                 }
//             }

//             if (lastFrame >= 0) {
//                 std::cout << "Est Depth: " << est_depth << std::endl;
//                 std::cout << "Refined Pos (t, x, y, z, colorID): (" << lastFrame << ", " << xEst << ", "
//                 << yEst << ", " <<  zEst << ", " << static_cast<int>(coneEst.colorID) << ")" << std::endl;
//             }
//         }
//     }

//     // StereoBenchAddTime(SB_TIME_IDX::FRAME_SIFT);
// }