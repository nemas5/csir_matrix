#pragma once
#include <vector>
#include <random>

class CSIRMatrix {
    private:
        std::vector<double> di;  // Диагональные элементы
        std::vector<double> elem;  // Все элементы левее диагонали
        std::vector<int> ia;  // Диапазоны строк
        std::vector<int> ja;  // С какого столбца в строке начинается ненулевой профиль
        std::mt19937 gen;
    public:
        CSIRMatrix(int n);
        void output();
        std::vector<double> cg(const std::vector<double>& b);
        std::vector<double> multiply_by_vector(const std::vector<double>& v);
};