#include <iostream>
#include <string>

#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/opencv.hpp>

static int th1 = 40, th2 = 65;

void cropAndRemoveColors(cv::Mat &image)
{
    constexpr auto limit = 100; // row range limit where the url bar exists
    const auto rows = image.rows > limit ? limit : image.rows;
    image = image.rowRange({0, rows});
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
}

void morphologyTransforms(cv::Mat &image)
{
    if (image.channels() != 1)
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // morphological gradient
    auto morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size{3, 3});
    cv::morphologyEx(image, image, cv::MORPH_GRADIENT, morphKernel);

    // connect horizontally oriented regions
    cv::threshold(image, image, th1, th2, cv::THRESH_BINARY | cv::THRESH_OTSU);
    morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size{9, 1});
    cv::morphologyEx(image, image, cv::MORPH_CLOSE, morphKernel);
}

void filterByCounting(cv::Mat &kek, cv::Mat &image)
{
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::findContours(image, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE, cv::Point{0, 0});

    cv::Rect result{0, 0, 0, 0};
    const auto contoursSize = static_cast<const int>(contours.size());
    const auto hierarchySize = static_cast<const int>(hierarchy.size());

    for (int idx = 0; idx >= 0 && hierarchySize > idx && contoursSize > idx; idx = hierarchy[idx][0])
    {
        auto rect = cv::boundingRect(contours[idx]);
        cv::Mat maskROI(mask, rect);
        maskROI = cv::Scalar::all(0);

        cv::drawContours(mask, contours, idx, cv::Scalar::all(255), -1);
        cv::rectangle(kek, rect, {255, 0, 0}, 2);
    }
}

bool isPair(const cv::Vec4i &a, const cv::Vec4i &b)
{
    constexpr auto hordiff = 10;
    constexpr auto verdiff = 50;

    bool res = std::abs(a[0] - b[0]) < hordiff;
    res &= std::abs(a[2] - b[2]) < hordiff;

    res &= std::abs(a[1] - b[2]) > verdiff;
    res &= std::abs(a[3] - b[3]) > verdiff;

    return res;
}

static std::pair<cv::Vec4i, cv::Vec4i> result;

bool makePair(const std::vector<cv::Vec4i> &lines, int center, unsigned int offset)
{
    for (auto line1 = lines.begin(); line1 != lines.end(); ++line1)
    {
        int center1 = (*line1)[0] + ((*line1)[2] - (*line1)[0]) / 2;
        if (std::abs(center1 - center) > offset)
            continue;

        for (auto line2 = line1 + 1; line2 != lines.end(); ++line2)
        {
            constexpr auto eps = 10;
            int center2 = (*line2)[0] + ((*line2)[2] - (*line2)[0]) / 2;
            if (std::abs(center2 - center1) < eps && std::abs((*line2)[0] - (*line1)[0]) < eps &&
                std::abs((*line2)[2] - (*line1)[2]) < eps)
            {
                result = std::make_pair(*line1, *line2);
                return true;
            }
        }
    }
    return false;
}

void blindLines(std::vector<cv::Vec4i> &lines, unsigned int horgap, unsigned int vergap)
{
    auto equal = [horgap, vergap](const cv::Vec4i &a, const cv::Vec4i &b) {
        return std::abs(a[0] - b[0]) <= horgap && // x1
               std::abs(a[1] - b[1]) <= vergap && // y1
               std::abs(a[2] - b[2]) <= horgap && // x2
               std::abs(a[3] - b[3]) <= vergap;   // y2
    };

    for (auto line1 = lines.begin(); line1 != lines.end(); ++line1)
    {
        for (auto line2 = line1 + 1; line2 != lines.end();)
        {
            if (equal(*line1, *line2))
            {
                line2 = lines.erase(line2);
                continue;
            }
            ++line2;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: ./urlbar-detector <image_name>" << std::endl;
        return -1;
    }

    const auto filename = argv[1];
    auto image = cv::imread(filename, cv::IMREAD_COLOR);

    int cols = image.cols / 2;
    if (cols > 150)
        cols -= 150;
    cv::namedWindow("After");

    // cv::createTrackbar("1", "After", &th1, 255);
    // cv::createTrackbar("2", "After", &th2, 255);
    cv::createTrackbar("3", "After", &cols, image.cols);

    do
    {
        auto img = image.clone();
        auto tmp = image.clone();
        cropAndRemoveColors(tmp);

        cv::Canny(tmp, tmp, th1, th2);

        // morphologyTransforms(tmp);
        // filterByCounting(image, tmp);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(tmp, lines, 1, CV_PI / 180, 30, cols /*30*/);
        blindLines(lines, 10, 10);
        if (makePair(lines, image.cols / 2, 50))
        {
            cv::line(img, cv::Point(result.first[0], result.first[1]), cv::Point(result.first[2], result.first[3]), {0, 0, 255}, 1, 8);
            cv::line(img, cv::Point(result.second[0], result.second[1]), cv::Point(result.second[2], result.second[3]), {0, 0, 255}, 1, 8);
        }
        // auto end = std::remove_if(lines.begin(), lines.end(), [&image](const cv::Vec4i &line) {
        //     return std::abs(line[0] - (image.cols - line[2])) > 10;
        // });

        // lines.erase(end, lines.end());

        // for (size_t i = 0; i < lines.size(); i++)
        // {
        //     std::cout << lines[i][0] << ' ' << lines[i][1] << ' ' << lines[i][2] << ' ' << lines[i][3] << std::endl;
        //     std::cout << "line sz: " << lines[i][2] - lines[i][0] << std::endl;
        //     // std::cout << "left " << lines[i][0] << ", right: " << image.cols - lines[i][2] << std::endl;

        //     cv::line(img, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), {0, 0, 255}, 1, 8);
        // }
        // std::cout << "total lines: " << lines.size() << std::endl << std::endl;

        cv::imshow("Before", img);
        cv::imshow("After", tmp);
    } while (cv::waitKey(1000) != 27);

    return 0;
}
