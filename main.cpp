#include <vector>
#include <iostream>
#include "CSIRMatrix/CSIRMatrix.hpp"

using namespace std;

int main() {
    int n = 8000;

    // Инициализация и генерация симметричной положительно определённой матрицы в CSIR формате
    CSIRMatrix matrix(n);
    matrix.output();

    // Генерация правой части
    mt19937 gen(54321);
    uniform_real_distribution<double> dist_elem(-1e+2, 1e+2);
    vector<double> b(n);
    cout << '\n' << "Правая часть:" << '\n';
    for (int i = 0; i < n; i++) {
        b[i] = dist_elem(gen);
        cout << b[i] << " ";
    }
    cout << '\n';
    // Получение решения СЛАУ ментодом сопряжённого градиента
    vector<double> x = matrix.cg(b);
    cout << '\n' << "Решение СЛАУ:" << '\n';
    for (int i = 0; i < n; i++) {
        cout << x[i] << " ";
    }
    cout << '\n';
    
    return 0;
}