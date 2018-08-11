#include <iostream>
#include <vector>

template <typename T>
void l_triangle(std::vector<std::vector<T>> &A, std::vector<T> &x, std::vector<T> &b){
    for(auto i = 0; i < A.size(); ++i){
        T sum {};
        for(auto j = 0; j < i; ++j)
            sum += A[i][j] * x[j];

        x[i] = (b[i] - sum)/A[i][i];
    }
}

template <typename T>
void u_triangle(std::vector<std::vector<T>> &A, std::vector<T> &x, std::vector<T> &b){
    for(int i = int(A.size()) - 1; i >= 0; --i){
        T sum {};
        for(int j = int(A.size()) - 1; j > i; --j)
            sum += A[i][j] * x[j];

        x[i] = (b[i] - sum)/A[i][i];
    }
}

template <typename T>
void pl_triangle(std::vector<std::vector<T>> &A, std::vector<T> &x, std::vector<T> &b, std::vector<T> &p){
    for(auto i = 0; i < A.size(); ++i){
        T sum {};
        for(auto j = 0; j < i; ++j)
            sum += A[p[i]][j] * x[j];

        x[i] = (b[p[i]] - sum)/A[i][i];
    }
}

template <typename T>
void pu_triangle(std::vector<std::vector<T>> &A, std::vector<T> &x, std::vector<T> &b, std::vector<T> &p){
    for(int i = int(A.size()) - 1; i >= 0; --i){
        T sum {};
        for(int j = int(A.size()) - 1; j > i; --j)
            sum += A[p[i]][j] * x[j];

        x[i] = (b[p[i]] - sum)/A[i][i];
    }
}

template <typename T>
void gauss_jordan(std::vector<std::vector<T>> &A, std::vector<T> &x, std::vector<T> &b){
    std::vector<int> rows;
    std::vector<int> columns;

    for(auto i = 0; i < A.size(); ++i){
        int mrow = i;
        int mcolumn = i;
        for(int j = i; j < A.size(); ++j)
            for(int k = i; k < A.size(); ++k){
                if(A[mrow][mcolumn] < A[j][k]){
                    mrow = j;
                    mcolumn = k;
                }
            }

        for(auto j = i + 1; j < A.size(); ++j){
            T l {A[j][i] / A[i][i]};
            A[j][i] = {};
            for(auto k = i + 1; k < A.size(); ++k)
                A[j][k] -= l * A[i][k];
            b[j] -= l * b[i];
        }
        for(auto t = 0; t < i; ++t){
            T l {A[t][i] / A[i][i]};
            A[t][i] = {};
            for(auto e = i + 1; e < A.size(); ++e)
                A[t][e] -= l * A[i][e];
            b[t] -= l * b[i];
        }
    }
    //u_triangle(A, x, b);
    for(auto i = 0; i < x.size(); ++i)
        x[i] = b[i] / A[i][i];
}

int main() {
    std::vector<std::vector<double>> A {std::vector<double>{1, 2, 3},
                                        std::vector<double>{4, 1, 2},
                                        std::vector<double>{2, 8, 7}};
    std::vector<double> x (3);
    std::vector<double> b {9, 3, 13};
    gauss_jordan(A, x, b);
    for(int i = 0; i < x.size(); ++i)
        std::cout << x[i] << " ";

    return 0;
}