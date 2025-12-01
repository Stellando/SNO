#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>
#include <string>

// CEC21 測試函數介面
extern "C" {
    void cec21_basic_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_bias_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_bias_rot_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_bias_shift_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_bias_shift_rot_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_rot_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_shift_func(double *x, double *f, int nx, int mx, int func_num);
    void cec21_shift_rot_func(double *x, double *f, int nx, int mx, int func_num);
}

// 函數名稱對應
class TestFunction {
public:
    static double evaluate(const std::string& funcName, const std::vector<double>& x, int dimension,
                          bool useBias = true, bool useShift = true, bool useRotation = true);
    static void getBounds(const std::string& funcName, int dimension, std::vector<double>& lb, std::vector<double>& ub);
};

#endif // FUNCTIONS_H
