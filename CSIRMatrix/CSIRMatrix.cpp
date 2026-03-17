#include "CSIRMatrix.hpp"
#include "utils/utils.hpp"
#include <iostream>
#include <omp.h>

std::vector<double> CSIRMatrix::multiply_by_vector(const std::vector<double>& v) {
    int n = v.size();
    std::vector<double> res(n, 0.0);
    #pragma omp parallel for
    for (int row = 0; row < n; row++) {
        res[row] += v[row] * di[row];  // При домножении на вектор один из компонентов вектора домножается на диагональный элемент
        for (int el_ptr = ia[row]; el_ptr < ia[row + 1]; el_ptr++) {  // Элементы левее диагонали в этой строке (хранимые по порядку)
            res[row] += elem[el_ptr] * v[(el_ptr - ia[row]) + ja[row]];  // Учитываем смещение при хранении строки
        }
        for (int row_ptr = row + 1; row_ptr < n; row_ptr++) {  // Перебираем все столбцы (фактические строки) правее диагонали и забираем из них элементы текущей строки row
            if (ja[row_ptr] <= row) {  // Если смещение столбца бошльше, чем номер строки, то пропускаем
                res[row] += elem[ia[row_ptr] + (row - ja[row_ptr])] * v[row_ptr];
            }
        }
    }
    return res;
}

std::vector<double> CSIRMatrix::cg(const std::vector<double>& b) {
    double eps = 0.001;
    std::vector<double> x(b.size(), 0.);  // Инициализируем вектор нулями
    std::vector<double> r_prev = vectorOperations::minus_vectors(b, multiply_by_vector(x));
    std::vector<double> r_new;
    std::vector<double> v(r_prev);
    double t, s;
    while (vectorOperations::vector_norm(r_prev) > eps) {
        t = vectorOperations::scalar(r_prev, r_prev) / vectorOperations::scalar(v, multiply_by_vector(v));
        x = vectorOperations::plus_vectors(x, vectorOperations::multiply_vector(v, t));
        r_new = vectorOperations::minus_vectors(r_prev, vectorOperations::multiply_vector(multiply_by_vector(v), t));
        s = vectorOperations::scalar(r_new, r_new) / vectorOperations::scalar(r_prev, r_prev);
        v = vectorOperations::plus_vectors(r_new, vectorOperations::multiply_vector(v, s));
        r_prev = r_new;
    }
    return x;
}

void CSIRMatrix::output() {
    std::cout << '\n' << "Симметричная положительно определённая матрица:" << '\n';
    std::cout << "Диагонали: ";
    for (int i = 0; i < di.size(); i++) {
        std::cout << di[i] << " ";
    }
    std::cout << '\n';
    std::cout << "Недиагональные элементы: ";
    for (int i = 0; i < elem.size(); i++) {
        std::cout << elem[i] << " ";
    }
    std::cout << '\n';
    std::cout << "ia: ";
    for (int i = 0; i < ia.size(); i++) {
        std::cout << ia[i] << " ";
    }
    std::cout << '\n';
        std::cout << "ja: ";
    for (int i = 0; i < ja.size(); i++) {
        std::cout << ja[i] << " ";
    }
    std::cout << '\n';
}

CSIRMatrix::CSIRMatrix(int n) : gen(12345) {
    std::vector<double> l_di(n);
    std::vector<double> l_elem;
    std::vector<int> l_ia, l_ja;
    l_ia.push_back(0);
    std::uniform_real_distribution<double> dist_diag(1e-2, 1e+2);
    std::uniform_real_distribution<double> dist_elem(-1e+3, 1e+2);
    std::bernoulli_distribution bern(0.4);
    bool flag;
    for (int i = 0; i < n; i++) {  // В цикле генерируется L-матрица (нижнедиагональная)
        l_di[i] = dist_diag(gen);
        flag = true;
        for (int j = 0; j < i; j++) {
            if (bern(gen) > 0 && flag) {
                l_ja.push_back(j);  // Записываем откуда начинается строка после первого ненулевого элемента
                flag = false;
            }
            if (flag == false) {
                l_elem.push_back(dist_elem(gen));  // Записываем сам элемент строки
            }
        }
        l_ia.push_back(l_elem.size());  // Где строка начинается (включительно) и где заканчивается (невключительно)
        if (flag) {
            l_ja.push_back(i);
        }
    }

    std::cout << '\n' << "Диагональная матрица L:" << '\n';
    std::cout << "Диагонали: ";
    for (int i = 0; i < l_di.size(); i++) {
        std::cout << l_di[i] << " ";
    }
    std::cout << '\n';
    std::cout << "Недиагональные элементы: ";
    for (int i = 0; i < l_elem.size(); i++) {
        std::cout << l_elem[i] << " ";
    }
    std::cout << '\n';
    std::cout << "l_ia: ";
    for (int i = 0; i < l_ia.size(); i++) {
        std::cout << l_ia[i] << " ";
    }
    std::cout << '\n';
        std::cout << "l_ja: ";
    for (int i = 0; i < l_ja.size(); i++) {
        std::cout << l_ja[i] << " ";
    }
    std::cout << '\n';

    // Получение симметричной положительно определённой матрицы
    di.resize(n);
    for (int i = 0; i < n; i++) {  // Квадраты диагоналей L добавляются к итоговой матрице
        di[i] += l_di[i] * l_di[i];
    }
    ia.push_back(0);
    int row_offset, col_offset, col;
    double new_element;
    for (int row = 0; row < n; row++) {  // Идём по строкам без диагоналей
        row_offset = l_ja[row];  // С этой позиции (включительно) начинаются элементы в строке
        // std::cout << "row: " << row << " row_offset: " << row_offset << '\n';
        flag = true;
        for (int el_row_ptr = l_ia[row]; el_row_ptr < l_ia[row + 1]; el_row_ptr++) {  // Реальные индексы элементов строки (уже сразу ненулевые со смещением)
            // std::cout << "el_row_ptr: " << el_row_ptr << '\n';
            di[row] += l_elem[el_row_ptr] * l_elem[el_row_ptr];  // Каждый элемент строки вносит вклад в диагональ своей строки своим квадратом
            col = el_row_ptr - l_ia[row] + row_offset;  // Индекс столбца, на который домножается текущая строка
            new_element = l_di[col] * l_elem[el_row_ptr];  // Каждая диагональ вносит вклад в итоговую матрицу прибавлением произведения текущего элемента и диагонали столбца
            col_offset = l_ja[col];  // С этой позиции (включительно) начинаются элементы в "столбце" (который фактически строка)
            if (row_offset >= col_offset) {  // Разные циклы в зависимости от величин сдвигов
                for (int el_col_ptr = l_ia[col] + row_offset - col_offset; el_col_ptr < l_ia[col + 1]; el_col_ptr++) {
                    new_element += l_elem[el_col_ptr] * l_elem[l_ia[row] + el_col_ptr - (l_ia[col] + row_offset - col_offset)];
                }
            }
            else {
                for (int el_col_ptr = l_ia[col]; el_col_ptr < l_ia[col + 1]; el_col_ptr++) {
                    new_element += l_elem[el_col_ptr] * l_elem[l_ia[row] + (col_offset - row_offset) + (el_col_ptr - l_ia[col])];
                }
            }
            std::cout << new_element << '\n';
            if (new_element != 0 && flag) {
                flag = false;
                ja.push_back(col);
            }
            if (flag == false)
                elem.push_back(new_element);
        }
        if (flag == true)
            ja.push_back(row);
        ia.push_back(elem.size());
    }
}