#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
constexpr const char* kResultWindow = "Result";
constexpr const char* kOriginalWindow = "Original";
constexpr const char* kGrayWindow = "Grayscale";
constexpr const char* kBinaryWindow = "Binary";
constexpr const char* kDataDirectory = "../data";

struct AcceptedContour
{
    std::vector<cv::Point> contour;
    double area = 0.0;
    double perimeter = 0.0;
    cv::Point bottomLeft;
    cv::Point bottomRight;
};

std::vector<fs::path> g_images;

int g_imageId = 0;
int g_adaptiveMethod = 1;
int g_blockSize = 13;
int g_cSlider = 55;      // Actual C is slider - 50.
int g_blurKsize = 3;
int g_minArea = 500;
int g_maxArea = 10000;
int g_minLength = 400;
int g_maxLength = 4000;

bool g_isUpdating = false;

std::string toLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool isJpgFile(const fs::path& path)
{
    return toLower(path.extension().string()) == ".jpg";
}

std::vector<fs::path> collectImages(const fs::path& directory)
{
    std::vector<fs::path> images;

    for (const auto& entry : fs::directory_iterator(directory))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        if (isJpgFile(entry.path()))
        {
            images.push_back(entry.path());
        }
    }

    std::sort(images.begin(), images.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename().string() < b.filename().string();
    });

    return images;
}

int makeOddAtLeast(int value, int minimum)
{
    value = std::max(value, minimum);
    if (value % 2 == 0)
    {
        ++value;
    }
    return value;
}

int makeOddOrZero(int value)
{
    if (value <= 0)
    {
        return 0;
    }

    if (value % 2 == 0)
    {
        ++value;
    }
    return value;
}

int actualC()
{
    return g_cSlider - 50;
}

cv::Point findBottomLeftPoint(const std::vector<cv::Point>& contour)
{
    cv::Point best = contour.front();
    int maxY = std::numeric_limits<int>::min();
    int minXAtMaxY = std::numeric_limits<int>::max();

    for (const auto& point : contour)
    {
        if (point.y > maxY || (point.y == maxY && point.x < minXAtMaxY))
        {
            maxY = point.y;
            minXAtMaxY = point.x;
            best = point;
        }
    }

    return best;
}

cv::Point findBottomRightPoint(const std::vector<cv::Point>& contour)
{
    cv::Point best = contour.front();
    int maxY = std::numeric_limits<int>::min();
    int maxXAtMaxY = std::numeric_limits<int>::min();

    for (const auto& point : contour)
    {
        if (point.y > maxY || (point.y == maxY && point.x > maxXAtMaxY))
        {
            maxY = point.y;
            maxXAtMaxY = point.x;
            best = point;
        }
    }

    return best;
}

void drawText(cv::Mat& image, const std::string& text, cv::Point origin, const cv::Scalar& color)
{
    origin.x = std::clamp(origin.x, 5, std::max(5, image.cols - 5));
    origin.y = std::clamp(origin.y, 20, std::max(20, image.rows - 10));

    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
}

void syncTrackbars(int imageCount)
{
    const int clampedImageId = std::clamp(g_imageId, 0, std::max(0, imageCount - 1));
    if (clampedImageId != g_imageId)
    {
        g_imageId = clampedImageId;
        cv::setTrackbarPos("image_id", kResultWindow, g_imageId);
    }

    const int fixedBlockSize = makeOddAtLeast(g_blockSize, 3);
    if (fixedBlockSize != g_blockSize)
    {
        g_blockSize = fixedBlockSize;
        cv::setTrackbarPos("block_size", kResultWindow, g_blockSize);
    }

    const int fixedBlur = makeOddOrZero(g_blurKsize);
    if (fixedBlur != g_blurKsize)
    {
        g_blurKsize = fixedBlur;
        cv::setTrackbarPos("blur_ksize", kResultWindow, g_blurKsize);
    }
}

void update()
{
    if (g_isUpdating || g_images.empty())
    {
        return;
    }

    g_isUpdating = true;
    syncTrackbars(static_cast<int>(g_images.size()));

    const fs::path& imagePath = g_images[static_cast<std::size_t>(g_imageId)];
    const cv::Mat original = cv::imread(imagePath.string(), cv::IMREAD_COLOR);

    if (original.empty())
    {
        std::cerr << "Failed to load image: " << imagePath << '\n';
        g_isUpdating = false;
        return;
    }

    cv::Mat grayscale;
    cv::cvtColor(original, grayscale, cv::COLOR_BGR2GRAY);

    cv::Mat thresholdInput = grayscale.clone();
    if (g_blurKsize > 0)
    {
        cv::GaussianBlur(
            grayscale,
            thresholdInput,
            cv::Size(g_blurKsize, g_blurKsize),
            0.0);
    }

    cv::Mat binary;
    const int adaptiveMethod =
        (g_adaptiveMethod == 0) ? cv::ADAPTIVE_THRESH_MEAN_C : cv::ADAPTIVE_THRESH_GAUSSIAN_C;
    cv::adaptiveThreshold(
        thresholdInput,
        binary,
        255,
        adaptiveMethod,
        cv::THRESH_BINARY_INV,
        g_blockSize,
        actualC());

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double minArea = static_cast<double>(std::min(g_minArea, g_maxArea));
    const double maxArea = static_cast<double>(std::max(g_minArea, g_maxArea));
    const double minLength = static_cast<double>(std::min(g_minLength, g_maxLength));
    const double maxLength = static_cast<double>(std::max(g_minLength, g_maxLength));

    std::vector<AcceptedContour> accepted;
    accepted.reserve(contours.size());

    for (const auto& contour : contours)
    {
        if (contour.size() < 2)
        {
            continue;
        }

        const double area = cv::contourArea(contour);
        const double perimeter = cv::arcLength(contour, true);

        if (!std::isfinite(area) || !std::isfinite(perimeter) || area <= 0.0 || perimeter <= 0.0)
        {
            continue;
        }

        if (area < minArea || area > maxArea)
        {
            continue;
        }

        if (perimeter < minLength || perimeter > maxLength)
        {
            continue;
        }

        accepted.push_back({
            contour,
            area,
            perimeter,
            findBottomLeftPoint(contour),
            findBottomRightPoint(contour)
        });
    }

    cv::Mat result = original.clone();
    for (const auto& item : accepted)
    {
        const std::vector<std::vector<cv::Point>> contourGroup = {item.contour};
        cv::drawContours(result, contourGroup, -1, cv::Scalar(0, 220, 0), 2, cv::LINE_AA);
    }

    int leftIndex = -1;
    int rightIndex = -1;
    const int midX = result.cols / 2;
    const int lowerRegionStartY = result.rows / 3;

    if (!accepted.empty())
    {
        for (int i = 0; i < static_cast<int>(accepted.size()); ++i)
        {
            const bool validLeftCandidate =
                accepted[i].bottomLeft.y >= lowerRegionStartY && accepted[i].bottomLeft.x < midX;
            if (validLeftCandidate &&
                (leftIndex < 0 ||
                 accepted[i].bottomLeft.x < accepted[leftIndex].bottomLeft.x ||
                 (accepted[i].bottomLeft.x == accepted[leftIndex].bottomLeft.x &&
                  accepted[i].bottomLeft.y > accepted[leftIndex].bottomLeft.y)))
            {
                leftIndex = i;
            }

            const bool validRightCandidate =
                accepted[i].bottomRight.y >= lowerRegionStartY && accepted[i].bottomRight.x > midX;
            if (validRightCandidate &&
                (rightIndex < 0 ||
                 accepted[i].bottomRight.x > accepted[rightIndex].bottomRight.x ||
                 (accepted[i].bottomRight.x == accepted[rightIndex].bottomRight.x &&
                  accepted[i].bottomRight.y > accepted[rightIndex].bottomRight.y)))
            {
                rightIndex = i;
            }
        }
    }

    if (leftIndex >= 0)
    {
        const auto& item = accepted[static_cast<std::size_t>(leftIndex)];
        const std::vector<std::vector<cv::Point>> contourGroup = {item.contour};
        cv::drawContours(result, contourGroup, -1, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
        cv::circle(result, item.bottomLeft, 6, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);
        drawText(result, "LEFT", item.bottomLeft + cv::Point(10, -10), cv::Scalar(255, 0, 0));
    }

    if (rightIndex >= 0)
    {
        const auto& item = accepted[static_cast<std::size_t>(rightIndex)];
        const std::vector<std::vector<cv::Point>> contourGroup = {item.contour};
        cv::drawContours(result, contourGroup, -1, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        cv::circle(result, item.bottomRight, 6, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        drawText(result, "RIGHT", item.bottomRight + cv::Point(10, -10), cv::Scalar(0, 0, 255));
    }

    drawText(result, "Image: " + imagePath.filename().string(), cv::Point(10, 25), cv::Scalar(255, 255, 255));
    drawText(
        result,
        "Method: " + std::string(g_adaptiveMethod == 0 ? "MEAN" : "GAUSSIAN") +
            "  block: " + std::to_string(g_blockSize) +
            "  C: " + std::to_string(actualC()) +
            "  blur: " + std::to_string(g_blurKsize),
        cv::Point(10, 50),
        cv::Scalar(255, 255, 255));
    drawText(
        result,
        "Contours total: " + std::to_string(contours.size()) +
            "  accepted: " + std::to_string(accepted.size()),
        cv::Point(10, 75),
        cv::Scalar(255, 255, 255));
    drawText(
        result,
        "Area [" + std::to_string(static_cast<int>(minArea)) + ", " + std::to_string(static_cast<int>(maxArea)) +
            "]  Length [" + std::to_string(static_cast<int>(minLength)) + ", " +
            std::to_string(static_cast<int>(maxLength)) + "]",
        cv::Point(10, 100),
        cv::Scalar(255, 255, 255));

    cv::imshow(kOriginalWindow, original);
    cv::imshow(kGrayWindow, grayscale);
    cv::imshow(kBinaryWindow, binary);
    cv::imshow(kResultWindow, result);

    g_isUpdating = false;
}

void onTrackbar(int, void*)
{
    update();
}
} // namespace

int main()
{
    const fs::path dataDirectory(kDataDirectory);
    if (!fs::exists(dataDirectory) || !fs::is_directory(dataDirectory))
    {
        std::cerr << "Data directory does not exist: " << dataDirectory << '\n';
        return 1;
    }

    g_images = collectImages(dataDirectory);
    if (g_images.empty())
    {
        std::cerr << "No .jpg images found in: " << dataDirectory << '\n';
        return 1;
    }

    cv::namedWindow(kResultWindow, cv::WINDOW_NORMAL);
    cv::namedWindow(kOriginalWindow, cv::WINDOW_NORMAL);
    cv::namedWindow(kGrayWindow, cv::WINDOW_NORMAL);
    cv::namedWindow(kBinaryWindow, cv::WINDOW_NORMAL);

    cv::createTrackbar("image_id", kResultWindow, &g_imageId, static_cast<int>(g_images.size()) - 1, onTrackbar);
    cv::createTrackbar("adaptive_method", kResultWindow, &g_adaptiveMethod, 1, onTrackbar);
    cv::createTrackbar("block_size", kResultWindow, &g_blockSize, 101, onTrackbar);
    cv::createTrackbar("C", kResultWindow, &g_cSlider, 100, onTrackbar);
    cv::createTrackbar("blur_ksize", kResultWindow, &g_blurKsize, 15, onTrackbar);
    cv::createTrackbar("min_area", kResultWindow, &g_minArea, 100000, onTrackbar);
    cv::createTrackbar("max_area", kResultWindow, &g_maxArea, 100000, onTrackbar);
    cv::createTrackbar("min_length", kResultWindow, &g_minLength, 10000, onTrackbar);
    cv::createTrackbar("max_length", kResultWindow, &g_maxLength, 10000, onTrackbar);

    update();

    while (true)
    {
        const int key = cv::waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
    }

    return 0;
}
