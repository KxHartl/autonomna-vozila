#include <opencv2/opencv.hpp>

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

    // Threshold
    cv::Mat thresholded;
    cv::adaptiveThreshold(grayscale, thresholded, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 31, 2);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresholded.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw results
    cv::Mat contourImage = cv::Mat::zeros(source.size(), CV_8UC3), annotated = source.clone();
    for (std::size_t i = 0; i < contours.size(); ++i)
    {
        const double area = cv::contourArea(contours[i]);
        const cv::Rect bounds = cv::boundingRect(contours[i]);
        const double ratio = static_cast<double>(bounds.width) / bounds.height;

        //Filter contours based on their characteristics
        if (area < 1200.0 || area > 8000.0 || ratio < 0.7 || ratio > 1.3) continue;

        const cv::Scalar color((37 * static_cast<int>(i) + 80) % 256, (83 * static_cast<int>(i) + 120) % 256, (149 * static_cast<int>(i) + 160) % 256);
        std::vector<cv::Point> approximation;
        const double arcLengthValue = cv::arcLength(contours[i], true);
        cv::approxPolyDP(contours[i], approximation, 7, true);
        cv::drawContours(contourImage, contours, static_cast<int>(i), color, 2);
        cv::drawContours(annotated, std::vector<std::vector<cv::Point>>{approximation}, -1, color, 2);

        if (approximation.size() > 2)
        {
            cv::putText(annotated, "Edges: " + std::to_string(approximation.size()), cv::Point(bounds.x, std::max(20, bounds.y - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2);
            cv::putText(annotated, "Area: " + std::to_string(static_cast<int>(area)), cv::Point(bounds.x, std::max(40, bounds.y + 20)), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2);
            std::cout << "Contour " << i << ": edges=" << approximation.size() << ", area=" << area << ", arcLength=" << arcLengthValue << ", height=" << bounds.height << '\n';
        }
    }

    cv::imshow("Original", source);
    cv::imshow("Grayscale", grayscale);
    cv::imshow("Adaptive Threshold", thresholded);
    cv::imshow("Contours", contourImage);
    cv::imshow("Approximated Edges", annotated);
    cv::waitKey(0);
    return 0;
}
