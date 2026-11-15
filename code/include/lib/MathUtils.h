#pragma once

#include <vector>
#include <algorithm>
namespace dnn {

double dotProduct(const std::vector<double>& a, const std::vector<double>& b);

double sum(const std::vector<double>& values);

// Encodes value as a one-hot vector of numClasses elements: the element at index value is 1.0
// and every other element is 0.0. Throws std::out_of_range if value is negative or >= numClasses.
std::vector<double> encodeAsOneHot(double value, std::size_t numClasses);

template <typename T>
std::vector<double> normalize(const std::vector<T>& values) {
    std::vector<double> result;
    if (values.empty()) {
        return result;
    }
    T maxElement = *std::max_element(values.begin(), values.end());
    result.reserve(values.size());
    for (const T& v : values) {
        result.push_back(static_cast<double>(v) / static_cast<double>(maxElement));
    }
    return result;
}

template <typename T>
std::vector<double> normalize(const std::vector<T>& values, int maxValue) {
    std::vector<double> result;
    result.reserve(values.size());
    for (const T& v : values) {
        result.push_back(static_cast<double>(v) / static_cast<double>(maxValue));
    }
    return result;
}

}
