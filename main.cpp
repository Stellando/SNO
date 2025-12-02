//g++ -g main.cpp algorithm.cpp functions.cpp cec21_test_func.cpp -o main.exe 
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <windows.h>  // Windows 控制台設定
#include "algorithm.h"
#include "functions.h"  // 加入 functions.h 以使用 getFunctionName

using namespace std;
// ==================== 測試設定區域 ====================
// CEC21 測試函數編號 (1-10):
//  1: BentCigar      2: Schwefel       3: BiRastrigin    4: GrieRosen      5: Hybrid1
//  6: Hybrid2        7: Hybrid3        8: Composition1   9: Composition2  10: Composition3
const int FUNCTION_START = 1;             // 起始函數編號 (1-10)
const int FUNCTION_END = 10;              // 結束函數編號 (1-10)
const int DIMENSION = 10;                 // 問題維度 (CEC21 支援 2, 10, 20)
const int MAX_EVALUATIONS = DIMENSION * 20000;  // 最大評估次數
const int RUN_TIMES = 1;                 // 每個函數執行次數

// ==================== CEC21 變形控制區 (三位元控制) ====================
// Bit 0: Bias   (0=關閉, 1=開啟)
// Bit 1: Shift  (0=關閉, 1=開啟)
// Bit 2: Rotation (0=關閉, 1=開啟)
const bool USE_BIAS = false;               // Bias 變形
const bool USE_SHIFT = false;              // Shift 變形
const bool USE_ROTATION = false;           // Rotation 變形
// 組合範例:
// false, false, false -> 無變形 (Basic)
// true,  false, false -> 只有 Bias
// false, true,  false -> 只有 Shift
// false, false, true  -> 只有 Rotation
// true,  true,  false -> Bias + Shift
// true,  false, true  -> Bias + Rotation
// false, true,  true  -> Shift + Rotation
// true,  true,  true  -> 全部變形 (Bias + Shift + Rotation)

// ==================== SNO 參數設定 ====================
const int N_S_INIT = 200;                 // 探索者初始數量 N(s)^init
const double N_X_RATIO = 0.1;             // 開發者比例 N(x)^init = N_S_INIT * N_X_RATIO
const double N_X_FINAL_RATIO = 0.2;       // 最終開發者比例 N(x)^final = N_S_INIT * N_X_FINAL_RATIO
const int N_NET = 13 * 13;                // 空間網大小 N(N) = 13^D (D=2時為169)
const int N_REGION = 13 * 13;             // 外部檔案大小 N(A) = 13^2
const double M_F = 0.3;                   // 突變參數 m^f (柯西分布)
const double M_C = 0.5;                   // 交配參數 m^c (常態分布)
const double RHO_MAX = 1.0;               // 最大比例 ρ_max
const double ALPHA = 1.0;                 // 參數 α
const double BETA = 3.0;                  // 參數 β
const int N_NA = 10;                      // 最大調整次數 N(N_a)

// ==================== 輸出控制 ====================
bool SAVE_TO_FILE = true;                 // true=輸出TXT檔案, false=不輸出

// ==================== 主程式 ====================
int main() {
    
    // 設定 Windows 控制台為 UTF-8 編碼
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    // 顯示測試配置
    cout << "=======================================================" << endl;
    cout << "  SNO 演算法 - CEC21 批次測試" << endl;
    cout << "=======================================================" << endl;
    cout << "測試函數: F" << FUNCTION_START << " - F" << FUNCTION_END << endl;
    cout << "維度: " << DIMENSION << " | 評估次數: " << MAX_EVALUATIONS << endl;
    cout << "每函數執行: " << RUN_TIMES << " 次 | 檔案輸出: " << (SAVE_TO_FILE ? "ON" : "OFF") << endl;
    cout << "變形設定: Bias=" << (USE_BIAS ? "ON" : "OFF") 
         << " | Shift=" << (USE_SHIFT ? "ON" : "OFF") 
         << " | Rotation=" << (USE_ROTATION ? "ON" : "OFF") << endl;
    cout << "-------------------------------------------------------" << endl;
    cout << "SNO 參數: N(s)=" << N_S_INIT << " | N(x)=" << int(N_S_INIT*N_X_RATIO) 
         << "→" << int(N_S_INIT*N_X_FINAL_RATIO)
         << " | m^f=" << M_F << " | m^c=" << M_C << endl;
    cout << "=======================================================" << endl << endl;
    
    // 對每個測試函數執行
    for(int funcNum = FUNCTION_START; funcNum <= FUNCTION_END; funcNum++) {
        string funcName = getFunctionName(funcNum);
        
        cout << "\n┌─────────────────────────────────────────────────────┐" << endl;
        cout << "│ F" << setw(2) << funcNum << ": " << left << setw(44) << funcName << "│" << endl;
        cout << "└─────────────────────────────────────────────────────┘" << endl;
        
        // 儲存結果
        vector<double> results;
        vector<double> times;
        
        // 執行測試
        for(int run = 1; run <= RUN_TIMES; run++) {
            cout << "  Run " << setw(2) << run << "/" << RUN_TIMES << "..." << flush;
            
            auto start = chrono::high_resolution_clock::now();
            
            // 建立並執行 SNO
            SNO sno(funcName, DIMENSION, MAX_EVALUATIONS, 
                    N_S_INIT, N_X_RATIO, N_X_FINAL_RATIO, N_NET, N_REGION,
                    M_F, M_C, RHO_MAX, N_NA, ALPHA, BETA,
                    USE_BIAS, USE_SHIFT, USE_ROTATION);
            sno.run();
            
            auto end = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(end - start).count();
            
            double fitness = sno.getBestFitness();
            results.push_back(fitness);
            times.push_back(elapsed);
            
            cout << " ✓ " << scientific << setprecision(4) << fitness 
                 << " (" << fixed << setprecision(2) << elapsed << "s)" << endl;
        }
        
        // 統計分析
        double mean = accumulate(results.begin(), results.end(), 0.0) / RUN_TIMES;
        double variance = 0.0;
        for(double val : results) variance += (val - mean) * (val - mean);
        double stdDev = sqrt(variance / RUN_TIMES);
        double best = *min_element(results.begin(), results.end());
        double worst = *max_element(results.begin(), results.end());
        double avgTime = accumulate(times.begin(), times.end(), 0.0) / RUN_TIMES;
        
        // 顯示統計結果
        cout << "\n  " << string(49, '-') << endl;
        cout << "  統計結果:" << endl;
        cout << "  " << string(49, '-') << endl;
        cout << scientific << setprecision(6);
        cout << "  Best:  " << best << endl;
        cout << "  Worst: " << worst << endl;
        cout << "  Mean:  " << mean << endl;
        cout << "  Std:   " << stdDev << endl;
        cout << fixed << setprecision(3);
        cout << "  Time:  " << avgTime << " s" << endl;
        cout << "  " << string(49, '-') << endl;
        
        // 檔案輸出
        if(SAVE_TO_FILE) {
            string filename = "SNO_F" + to_string(funcNum) + "_" + funcName + "_D" + to_string(DIMENSION) + ".txt";
            ofstream file(filename);
            
            if(file.is_open()) {
                file << "SNO Algorithm Test Results" << endl;
                file << "Function: F" << funcNum << " (" << funcName << ") | Dimension: " << DIMENSION << endl;
                file << "Evaluations: " << MAX_EVALUATIONS << " | Runs: " << RUN_TIMES << endl;
                file << "Transformation: Bias=" << (USE_BIAS ? "ON" : "OFF")
                     << " | Shift=" << (USE_SHIFT ? "ON" : "OFF")
                     << " | Rotation=" << (USE_ROTATION ? "ON" : "OFF") << endl;
                file << "========================================" << endl << endl;
                
                file << scientific << setprecision(6);
                file << "Best:  " << best << endl;
                file << "Worst: " << worst << endl;
                file << "Mean:  " << mean << endl;
                file << "Std:   " << stdDev << endl;
                file << fixed << setprecision(3);
                file << "Time:  " << avgTime << " s" << endl << endl;
                
                file << "All Results:" << endl;
                file << scientific << setprecision(6);
                for(int i = 0; i < RUN_TIMES; i++) {
                    file << "Run " << setw(2) << (i+1) << ": " << results[i] << endl;
                }
                
                file.close();
                cout << "\n  ✓ 結果已儲存: " << filename << endl;
            }
        }
    }
    
    cout << "\n=======================================================" << endl;
    cout << "  所有測試完成！" << endl;
    cout << "=======================================================" << endl;
    system("pause");
    return 0;
}



