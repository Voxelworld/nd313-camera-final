
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf_s(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf_s(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


inline double sqr(double x) { return x*x; }

void ComputeMeanStd(const std::vector<double> &x, double &mean, double &std)
{
    mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    std = std::accumulate(x.begin(), x.end(), 0.0,
                          [mean](double sum, double x) { return sum + sqr(x-mean); }
    ) / x.size();
    std = sqrt(std);
    cout << "   Data mean=" << mean << " std=" << std << endl;
}

void ComputeMedianAbsoluteDeviation(std::vector<double> &x, double &median, double &mad)
{
    // https://hausetutorials.netlify.app/posts/2019-10-07-outlier-detection-with-median-absolute-deviation/
    std::sort(x.begin(), x.end());
    int n_2 = int(x.size() / 2);
    median = (x.size() % 2 == 1) ? x[n_2] : (x[n_2 - 1] + x[n_2]) / 2.0;

    // abs = |x_i - median(x)|
    std::vector<double> abs(x.size());
    std::transform(x.begin(), x.end(), abs.begin(), [median](double x) { return fabs(x - median); });
    std::sort(abs.begin(), abs.end());
    mad = (abs.size() % 2 == 1) ? abs[n_2] : (abs[n_2 - 1] + abs[n_2]) / 2.0;
    mad *= 1.4826;
    cout << "   Data median=" << median << " mad=" << mad << endl;
}

// FP.3 : Associate Keypoint Correspondences with Bounding Box it contains.
//
// Before a TTC estimate can be computed in the next exercise, you need to find all keypoint matches that belong to each 3D object.
// You can do this by simply checking whether the corresponding keypoints are within the region of interest in the camera image.
// All matches which satisfy this condition should be added to a vector.
// The problem you will find is that there will be outliers among your matches. To eliminate those,
// I recommend that you compute a robust mean of all the euclidean distances between keypoint matches
// and then remove those that are too far away from the mean.
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // filter keypoint candidates in current bounding box and compute distances
    // Find all keypoint matches which belong to each 3D object (inside bounding box).
    vector<double> distances;
    for (auto m : kptMatches)
    {
        if (boundingBox.roi.contains(kptsCurr[m.trainIdx].pt))
        {
            double dist = cv::norm(kptsCurr[m.trainIdx].pt - kptsPrev[m.queryIdx].pt); // L2 norm
            distances.push_back(dist);
        }
    }

#define Z_SCORE_TEST
#ifdef Z_SCORE_TEST
    // Simple outlier removal by z-score, based on data mean/variance.
    double mean, std;
    ComputeMeanStd(distances, mean, std);
    for (auto m : kptMatches)
    {
        if (boundingBox.roi.contains(kptsCurr[m.trainIdx].pt))
        {
            double dist = cv::norm(kptsCurr[m.trainIdx].pt - kptsPrev[m.queryIdx].pt); // L2 norm
            if (fabs(dist - mean) <= 3 * std) // 99.7% of normal distribution
            {
                boundingBox.keypoints.push_back(kptsCurr[m.trainIdx]);
                boundingBox.kptMatches.push_back(m);
            }
        }
    }
    cout << "Clustered " << boundingBox.keypoints.size() << "/" << distances.size() << " points by z-score." << endl;
#else    
    // Outlier removal by Median Absolute Deviation (MAD).
    // This is more robust than filtering using mean/variance of the data.
    // https://hausetutorials.netlify.app/posts/2019-10-07-outlier-detection-with-median-absolute-deviation/
    double median, mad;
    ComputeMedianAbsoluteDeviation(distances, median, mad);

    for (auto m : kptMatches)
    {
        if (boundingBox.roi.contains(kptsCurr[m.trainIdx].pt))
        {
            double dist = cv::norm(kptsCurr[m.trainIdx].pt - kptsPrev[m.queryIdx].pt); // L2 norm
            if (fabs(dist - median) <= 5 * mad)
            {
                boundingBox.keypoints.push_back(kptsCurr[m.trainIdx]);
                boundingBox.kptMatches.push_back(m);
            }
        }
    }
    cout << "Clustered " << boundingBox.keypoints.size() << "/" << distances.size() << " points by MAD." << endl;
#endif
}

// FP.4 : Compute Camera-based time-to-collision (TTC) based on keypoint correspondences in successive images.
//
// Once keypoint matches have been added to the bounding boxes, the next step is to compute the TTC estimate.
// As with Lidar, we already looked into this in the second lesson of this course, so you please revisit the respective section
// and use the code sample there as a starting point for this task here.
// Once you have your estimate of the TTC, please return it to the main function at the end of computeTTCCamera.
//
// The task is complete once the code is functional and returns the specified output.
// Also, the code must be able to deal with outlier correspondences in a statistically robust way to avoid severe estimation errors.
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // See lesson 3.3: Estimating TTC with Camera
    // https://classroom.udacity.com/nanodegrees/nd313/parts/1971021c-523b-414c-93a3-2c6297cf4771/modules/3eb3ecc3-b73d-43bb-b565-dcdd5d7a2635/lessons/dfe71db5-4233-4e4f-b33f-40cb9899dc13/concepts/daceaff3-1519-4f4c-82ff-16e02b5c2e8f

    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    double dT = 1 / frameRate;

    // a) Mean distance ratio
    //double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();
    //TTC = -dT / (1 - meanDistRatio);
    //cout << "TTC camera: " << distRatios.size() << " ratios. meanDistRatio=" << meanDistRatio << ", TTC=" << TTC << endl;

    // b) Median distance ratio (statistically more robust)
    std::sort(distRatios.begin(), distRatios.end());
    int n_2 = int(distRatios.size() / 2);
    double medianDistRatio = (distRatios.size() % 2 == 1) ? distRatios[n_2] : (distRatios[n_2 - 1] + distRatios[n_2]) / 2.0;
    TTC = -dT / (1 - medianDistRatio);
    cout << "TTC camera: " << distRatios.size() << " ratios. medianDistRatio=" << medianDistRatio << ", TTC=" << TTC << endl;
}

// FP.2 Helper
static double computeMeanInQuantile(std::vector<LidarPoint> & pts, double quantileMin, double quantileMax)
{
    // sort data and use quantileMin% - quantileMax% of the data to compute a stable mean X_min value.
    std::sort(pts.begin(), pts.end(), [](const LidarPoint &a, const LidarPoint &b) { return a.x < b.x; });
    auto start = int(pts.size() * quantileMin);
    auto end = int(pts.size() * quantileMax);
    double minX = std::accumulate(pts.begin() + start, pts.begin() + end, 0.0, [](const double sum, const LidarPoint &p) { return sum + p.x; });
    minX /= end - start + 1;
    cout << "X_min=" << minX << " based on X-range " << pts[0].x << "-" << pts[pts.size() - 1].x
            << " and index range " << start << "-" << end << " of " << pts.size() << endl;
    return minX;
}

// FP.2 : Compute Lidar-based TTC (x,y,z,r)
void computeTTCLidar(std::vector<LidarPoint> & lidarPointsPrev, std::vector<LidarPoint> & lidarPointsCurr, double frameRate, double &TTC)
{
    // Lesson 3.2: Estimating TTC with Lidar
    // https://classroom.udacity.com/nanodegrees/nd313/parts/1971021c-523b-414c-93a3-2c6297cf4771/modules/3eb3ecc3-b73d-43bb-b565-dcdd5d7a2635/lessons/dfe71db5-4233-4e4f-b33f-40cb9899dc13/concepts/c78c2068-ff3b-4146-9f1f-77ea44188ef2

#define ROBUST_LIDAR
#ifdef ROBUST_LIDAR
    // More robust x-min point selection by using 5%-30% quantile to remove single noisy points (see top view)
    double quantileMin = 0.05;
    double quantileMax = 0.30;
    double minXPrev = computeMeanInQuantile(lidarPointsPrev, quantileMin, quantileMax);
    double minXCurr = computeMeanInQuantile(lidarPointsCurr, quantileMin, quantileMax);
#else
    // Naive selection of nearest point
    double minXPrev = std::min_element(lidarPointsPrev.begin(), lidarPointsPrev.end(), [&](const LidarPoint &a, const LidarPoint &b) { return a.x < b.x; })->x;
    double minXCurr = std::min_element(lidarPointsCurr.begin(), lidarPointsCurr.end(), [&](const LidarPoint &a, const LidarPoint &b) { return a.x < b.x; })->x;
#endif

    // compute TTC from both measurements
    double dT = 1.0 / frameRate; // time between two measurements in seconds
    TTC = minXCurr * dT / (minXPrev - minXCurr);
    cout << "TTC lidar: delta x scale factor is " << (1.0 / (minXPrev - minXCurr)) << ", TTC=" << TTC << endl;
}

// FP.1 : Match 3D Objects
void matchBoundingBoxes(std::vector<cv::DMatch> & matches, std::map<int, int> & bbBestMatches, DataFrame & prevFrame, DataFrame & currFrame)
{
    auto &boxes_prev = prevFrame.boundingBoxes; // shorter alias
    auto &boxes_curr = currFrame.boundingBoxes;

    // Build voting matrix (see Report.md)
    // Each match between points votes for the corresponding bounding box pair which are hit by the point coordinates.
    cv::Mat votes = cv::Mat::zeros(int(boxes_prev.size()), int(boxes_curr.size()), cv::DataType<int>::type);

    // Let each match vote for it's bounding box pair (box_prev, box_curr) in matrix.
    for (auto m : matches)
    {
        for (auto y = 0; y < boxes_prev.size(); ++y)
        {
            for (auto x = 0; x < boxes_curr.size(); ++x)
            {
                if (boxes_prev[y].roi.contains(prevFrame.keypoints[m.queryIdx].pt) &&
                    boxes_curr[x].roi.contains(currFrame.keypoints[m.trainIdx].pt))
                {
                    votes.at<int>(y, x)++;
                }
            }
        }
    }

    // {
    //     string windowName = "Vote Matrix of Bounding-Box-Matches";
    //     cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    //     cv::imshow(windowName, votes);
    //     cv::waitKey(0);
    // }

    // Vote counting strategy:
    // Start with highest voted pair and remove the associated boxes from
    // the list of available boxes (one-to-one correspondence aka. bijection).
    double voteCount = 0.0;
    for (;;)
    {
        // find maximum vote cout in votes matrix
        cv::Point2i votePair;
        cv::minMaxLoc(votes, nullptr, &voteCount, nullptr, &votePair);

        if (voteCount < 1.0) // stop when no more box-pair votes are found
            break;

        //cout << "Vote for (box_" << votePair.y << ", box_" << votePair.x << ") is " << int(voteCount) << endl;
        auto boxId_prev = boxes_prev[votePair.y].boxID;
        auto boxId_curr = boxes_curr[votePair.x].boxID;
        bbBestMatches[boxId_prev] = boxId_curr;

        // Remove both boxes by resetting their counts to zero.
        votes.row(votePair.y) = 0;
        votes.col(votePair.x) = 0;
    }
    cout << "Number of matched bounding boxes: " << bbBestMatches.size() << endl;
}
