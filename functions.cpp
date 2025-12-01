#include "functions.h"
#include <cstring>
#include <algorithm>

// CEC21 函數名稱映射
int getFunctionNumber(const std::string& funcName) {
    if (funcName == "BentCigar") return 1;
    if (funcName == "Schwefel") return 2;
    if (funcName == "BiRastrigin") return 3;
    if (funcName == "GrieRosen") return 4;
    if (funcName == "Hybrid1") return 5;
    if (funcName == "Hybrid2") return 6;
    if (funcName == "Hybrid3") return 7;
    if (funcName == "Composition1") return 8;
    if (funcName == "Composition2") return 9;
    if (funcName == "Composition3") return 10;
    return 1; // 預設
}

double TestFunction::evaluate(const std::string& funcName, const std::vector<double>& x, int dimension,
                             bool useBias, bool useShift, bool useRotation) {
    int funcNum = getFunctionNumber(funcName);
    
    // 將 vector 轉換為陣列
    double* xArray = new double[dimension];
    for (int i = 0; i < dimension; i++) {
        xArray[i] = x[i];
    }
    
    double fitness;
    
    // 根據三位元控制選擇對應的 CEC21 變形函數
    if (!useBias && !useShift && !useRotation) {
        // 000: Basic (無變形)
        cec21_basic_func(xArray, &fitness, dimension, 1, funcNum);
    }
    else if (useBias && !useShift && !useRotation) {
        // 100: Bias only
        cec21_bias_func(xArray, &fitness, dimension, 1, funcNum);
    }
    else if (!useBias && useShift && !useRotation) {
        // 010: Shift only
        cec21_shift_func(xArray, &fitness, dimension, 1, funcNum);
    }
    else if (!useBias && !useShift && useRotation) {
        // 001: Rotation only
        cec21_rot_func(xArray, &fitness, dimension, 1, funcNum);
    }
    else if (useBias && useShift && !useRotation) {
        // 110: Bias + Shift
        cec21_bias_shift_func(xArray, &fitness, dimension, 1, funcNum);
    }
    else if (useBias && !useShift && useRotation) {
        // 101: Bias + Rotation
        cec21_bias_rot_func(xArray, &fitness, dimension, 1, funcNum);
    }
    else if (!useBias && useShift && useRotation) {
        // 011: Shift + Rotation
        cec21_shift_rot_func(xArray, &fitness, dimension, 1, funcNum);
    }
    else {
        // 111: Bias + Shift + Rotation (全部)
        cec21_bias_shift_rot_func(xArray, &fitness, dimension, 1, funcNum);
    }
    
    delete[] xArray;
    return fitness;
}

void TestFunction::getBounds(const std::string& funcName, int dimension, 
                             std::vector<double>& lb, std::vector<double>& ub) {
    // CEC21 的標準搜尋範圍是 [-100, 100]
    lb.assign(dimension, -100.0);
    ub.assign(dimension, 100.0);
}
