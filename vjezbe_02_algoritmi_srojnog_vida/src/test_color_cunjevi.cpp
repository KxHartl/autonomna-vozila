#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main()
{
    // Read image
    cv::Mat image = cv::imread("../data/c1.png", cv::IMREAD_COLOR);
    if (image.empty()) { std::cerr << "Unable to load the image!\n"; return 1; }

    // Convert colorspace
    cv::Mat hsv, redMask, greenMask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    int redLowH = 171;
    int redLowS = 100;
    int redLowV = 100;
    int redHighH = 180;
    int redHighS = 245;
    int redHighV = 245;

    int greenLowH = 35;
    int greenLowS = 60;
    int greenLowV = 60;
    int greenHighH = 90;
    int greenHighS = 255;
    int greenHighV = 255;

    int redMinArea = 200;
    int redMaxArea = 30000;
    int greenMinArea = 200;
    int greenMaxArea = 30000;

    cv::namedWindow("Detected cones", cv::WINDOW_NORMAL);
    cv::namedWindow("Red mask", cv::WINDOW_NORMAL);
    cv::namedWindow("Green mask", cv::WINDOW_NORMAL);
    cv::namedWindow("Combined mask", cv::WINDOW_NORMAL);
    cv::namedWindow("Threshold controls", cv::WINDOW_NORMAL);

    cv::createTrackbar("R low H", "Threshold controls", &redLowH, 180);
    cv::createTrackbar("R low S", "Threshold controls", &redLowS, 255);
    cv::createTrackbar("R low V", "Threshold controls", &redLowV, 255);
    cv::createTrackbar("R high H", "Threshold controls", &redHighH, 180);
    cv::createTrackbar("R high S", "Threshold controls", &redHighS, 255);
    cv::createTrackbar("R high V", "Threshold controls", &redHighV, 255);

    cv::createTrackbar("G low H", "Threshold controls", &greenLowH, 180);
    cv::createTrackbar("G low S", "Threshold controls", &greenLowS, 255);
    cv::createTrackbar("G low V", "Threshold controls", &greenLowV, 255);
    cv::createTrackbar("G high H", "Threshold controls", &greenHighH, 180);
    cv::createTrackbar("G high S", "Threshold controls", &greenHighS, 255);
    cv::createTrackbar("G high V", "Threshold controls", &greenHighV, 255);

    cv::createTrackbar("R min area", "Threshold controls", &redMinArea, 10000);
    cv::createTrackbar("R max area", "Threshold controls", &redMaxArea, 10000);
    cv::createTrackbar("G min area", "Threshold controls", &greenMinArea, 10000);
    cv::createTrackbar("G max area", "Threshold controls", &greenMaxArea, 10000);

    auto drawDetectedCones = [&](cv::Mat& output,
                                 const cv::Mat& mask,
                                 const cv::Scalar& color,
                                 const std::string& label,
                                 int minArea,
                                 int maxArea) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (std::size_t i = 0; i < contours.size(); ++i)
        {
            double area = cv::contourArea(contours[i]);
            if (area < static_cast<double>(minArea) || area > static_cast<double>(maxArea)) continue;

            cv::Rect box = cv::boundingRect(contours[i]);
            cv::rectangle(output, box, color, 2);
            cv::putText(
                output,
                label + " Area: " + std::to_string(static_cast<int>(area)),
                cv::Point(box.x, std::max(20, box.y - 5)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1);
        }
    };

    while (true)
    {
        redLowH = std::min(redLowH, redHighH);
        redLowS = std::min(redLowS, redHighS);
        redLowV = std::min(redLowV, redHighV);
        greenLowH = std::min(greenLowH, greenHighH);
        greenLowS = std::min(greenLowS, greenHighS);
        greenLowV = std::min(greenLowV, greenHighV);
        redMinArea = std::min(redMinArea, redMaxArea);
        greenMinArea = std::min(greenMinArea, greenMaxArea);

        cv::setTrackbarPos("R low H", "Threshold controls", redLowH);
        cv::setTrackbarPos("R low S", "Threshold controls", redLowS);
        cv::setTrackbarPos("R low V", "Threshold controls", redLowV);
        cv::setTrackbarPos("G low H", "Threshold controls", greenLowH);
        cv::setTrackbarPos("G low S", "Threshold controls", greenLowS);
        cv::setTrackbarPos("G low V", "Threshold controls", greenLowV);
        cv::setTrackbarPos("R min area", "Threshold controls", redMinArea);
        cv::setTrackbarPos("G min area", "Threshold controls", greenMinArea);

        // Threshold red and green cones in HSV space.
        cv::inRange(
            hsv,
            cv::Scalar(redLowH, redLowS, redLowV),
            cv::Scalar(redHighH, redHighS, redHighV),
            redMask);
        cv::inRange(
            hsv,
            cv::Scalar(greenLowH, greenLowS, greenLowV),
            cv::Scalar(greenHighH, greenHighS, greenHighV),
            greenMask);

        cv::Mat output = image.clone();
        drawDetectedCones(output, redMask, cv::Scalar(0, 0, 255), "RED", redMinArea, redMaxArea);
        drawDetectedCones(output, greenMask, cv::Scalar(0, 255, 0), "GREEN", greenMinArea, greenMaxArea);

        cv::Mat combinedMask;
        cv::bitwise_or(redMask, greenMask, combinedMask);

        // Show result
        cv::imshow("Detected cones", output);
        cv::imshow("Red mask", redMask);
        cv::imshow("Green mask", greenMask);
        cv::imshow("Combined mask", combinedMask);

        const int key = cv::waitKey(30);
        if (key == 27)
        {
            break;
        }
    }

    return 0;
}
