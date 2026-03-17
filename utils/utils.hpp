#pragma once
#include <vector>
#include <cmath>
#include <omp.h>

namespace vectorOperations {
    double scalar(const std::vector<double>& a, const std::vector<double>& b) {
    double ans = 0.;
    #pragma omp parallel for reduction(+:ans)
    for (size_t i = 0; i < a.size(); i++) {
        ans += a[i] * b[i];
    }
    return ans;
}

std::vector<double> minus_vectors(const std::vector<double>& a, const std::vector<double>& b) {
    size_t n = a.size();
    std::vector<double> ans(n);
    for (size_t i = 0; i < n; i++) {
        ans[i] = a[i] - b[i];
    }
    return ans;
}

std::vector<double> plus_vectors(const std::vector<double>& a, const std::vector<double>& b) {
    size_t n = a.size();
    std::vector<double> ans(n);
    for (size_t i = 0; i < n; i++) {
        ans[i] = a[i] + b[i];
    }
    return ans;
}

std::vector<double> multiply_vector(const std::vector<double>& a, double b) {
    size_t n = a.size();
    std::vector<double> ans(n);
    for (size_t i = 0; i < n; i++) {
        ans[i] = a[i] * b;
    }
    return ans;
}

double vector_norm(const std::vector<double>& vec) {
    double ans = 0.;
    #pragma omp parallel for reduction(+:ans)
    for (size_t i = 0; i < vec.size(); i++) {
        ans += vec[i] * vec[i];
    }
    return sqrt(ans);
}
}