#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

int main()
{
    // Read image
    const cv::Mat source = cv::imread("../data/signs2.png", cv::IMREAD_COLOR);
    if (source.empty()) { std::cerr << "Could not open input image\n"; return 1; }

    // Convert colorspace
    cv::Mat grayscale;
    cv::cvtColor(source, grayscale, cv::COLOR_BGR2GRAY);

    int adaptiveMethod = 0;  // 0 = MEAN, 1 = GAUSSIAN
    int blockSize = 13;
    int cSlider = 51;        // Actual C = cSlider - 50
    int minArea = 600;
    int maxArea = 10000;
    int minRatioX100 = 58;   // 0.70
    int maxRatioX100 = 160;  // 1.30
    int approxEpsilon = 8;

    cv::namedWindow("Controls", cv::WINDOW_NORMAL);
    cv::imshow("Original", source);
    cv::imshow("Grayscale", grayscale);

    cv::createTrackbar("adaptive_method", "Controls", &adaptiveMethod, 1);
    cv::createTrackbar("block_size", "Controls", &blockSize, 101);
    cv::createTrackbar("C_slider", "Controls", &cSlider, 100);
    cv::createTrackbar("min_area", "Controls", &minArea, 50000);
    cv::createTrackbar("max_area", "Controls", &maxArea, 50000);
    cv::createTrackbar("min_ratio_x100", "Controls", &minRatioX100, 300);
    cv::createTrackbar("max_ratio_x100", "Controls", &maxRatioX100, 300);
    cv::createTrackbar("approx_eps", "Controls", &approxEpsilon, 100);

    while (true)
    {
        blockSize = std::max(blockSize, 3);
        if (blockSize % 2 == 0)
        {
            ++blockSize;
        }

        minArea = std::min(minArea, maxArea);
        minRatioX100 = std::min(minRatioX100, maxRatioX100);
        approxEpsilon = std::max(approxEpsilon, 1);

        cv::setTrackbarPos("block_size", "Controls", blockSize);
        cv::setTrackbarPos("min_area", "Controls", minArea);
        cv::setTrackbarPos("min_ratio_x100", "Controls", minRatioX100);
        cv::setTrackbarPos("approx_eps", "Controls", approxEpsilon);

        cv::Mat thresholded;
        const int actualC = cSlider - 50;
        const int adaptiveMethodType =
            (adaptiveMethod == 0) ? cv::ADAPTIVE_THRESH_MEAN_C : cv::ADAPTIVE_THRESH_GAUSSIAN_C;
        cv::adaptiveThreshold(
            grayscale,
            thresholded,
            255,
            adaptiveMethodType,
            cv::THRESH_BINARY,
            blockSize,
            actualC);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresholded.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat contourImage = cv::Mat::zeros(source.size(), CV_8UC3);
        cv::Mat annotated = source.clone();

        for (std::size_t i = 0; i < contours.size(); ++i)
        {
            const double area = cv::contourArea(contours[i]);
            const cv::Rect bounds = cv::boundingRect(contours[i]);
            if (bounds.height <= 0)
            {
                continue;
            }

            const double ratio = static_cast<double>(bounds.width) / bounds.height;
            const double minRatio = static_cast<double>(minRatioX100) / 100.0;
            const double maxRatio = static_cast<double>(maxRatioX100) / 100.0;

            // Filter contours based on current slider settings.
            if (area < static_cast<double>(minArea) ||
                area > static_cast<double>(maxArea) ||
                ratio < minRatio ||
                ratio > maxRatio)
            {
                continue;
            }

            const cv::Scalar color(
                (37 * static_cast<int>(i) + 80) % 256,
                (83 * static_cast<int>(i) + 120) % 256,
                (149 * static_cast<int>(i) + 160) % 256);

            std::vector<cv::Point> approximation;
            cv::approxPolyDP(contours[i], approximation, static_cast<double>(approxEpsilon), true);

            cv::drawContours(contourImage, contours, static_cast<int>(i), color, 2);
            cv::drawContours(annotated, std::vector<std::vector<cv::Point>>{approximation}, -1, color, 2);

            if (approximation.size() > 2)
            {
                cv::putText(
                    annotated,
                    "Edges: " + std::to_string(approximation.size()),
                    cv::Point(bounds.x, std::max(20, bounds.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2);
                cv::putText(
                    annotated,
                    "Area: " + std::to_string(static_cast<int>(area)),
                    cv::Point(bounds.x, std::max(40, bounds.y + 20)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2);
            }
        }

        cv::imshow("Adaptive Threshold", thresholded);
        cv::imshow("Contours", contourImage);
        cv::imshow("Approximated Edges", annotated);

        const int key = cv::waitKey(30);
        if (key == 27)
        {
            break;
        }
    }

    return 0;
}
