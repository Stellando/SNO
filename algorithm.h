#include <vector>
#include <functional>
#include <random>
#include <ctime>
#include <string>

#ifndef ALGORITHM_H
#define ALGORITHM_H

using namespace std;

// ==================== 資料結構定義 ====================

// 解的結構
struct Solution {
    vector<double> position;      // 位置向量
    double fitness;               // 適應值
    
    Solution() : fitness(1e100) {}
    Solution(int dim) : position(dim, 0.0), fitness(1e100) {}
};

// 空間網點的結構
struct NetPoint {
    vector<double> position;      // 網格點位置
    double fitness;               // 目標函數值
    int regionID;                 // 所屬區域ID
    
    NetPoint() : fitness(1e100), regionID(-1) {}
    NetPoint(int dim) : position(dim, 0.0), fitness(1e100), regionID(-1) {}
};

// 區域資訊結構
struct Region {
    int id;                       // 區域ID
    vector<int> netPointIndices;  // 所屬的網格點索引
    
    // 搜尋期望值相關
    double expectedValue;         // 搜尋期望值 e_i
    int visitCount;               // v_a
    int unvisitedCount;           // v_b
    bool visitedThisRound;        // 輔助標記
    vector<double> recentImprovements;  // 最近的改進值 (用於計算sum)
    double bestFitness;           // 區域最佳值 f(R_{i,p}^t)
    
    // 正規化前的原始值
    double rawVisitRatio;         // 原始訪問比例
    double rawImprovementSum;     // 原始改進總和
    double rawBestValue;          // 原始最佳值項
    
    // 參數歷史紀錄 (用於自適應)
    vector<double> successfulF;   // 成功的突變因子 f
    vector<double> successfulC;   // 成功的交配率 c
    double meanF;                 // 參數均值 m_i^f (Eq. 3.9)
    double meanC;                 // 參數均值 m_i^c (Eq. 3.9)
    
    Region() : id(-1), expectedValue(0.0), visitCount(1), unvisitedCount(1),
               visitedThisRound(false), bestFitness(1e100), rawVisitRatio(0.0), 
               rawImprovementSum(0.0), rawBestValue(0.0), meanF(0.5), meanC(0.5) {}
};

// 外部檔案 (Archive) 結構
struct ArchiveItem {
    vector<double> position;      // 被拋棄的解的位置
    double fitness;               // 適應值
    int iteration;                // 被拋棄時的迭代次數
    
    ArchiveItem() : fitness(1e100), iteration(0) {}
    ArchiveItem(const Solution& sol, int iter) 
        : position(sol.position), fitness(sol.fitness), iteration(iter) {}
};

// ==================== SNO 演算法主類別 ====================
class SNO
{
private:
    // 測試函數相關
    string functionName;
    int dimension;
    int maxEvaluations;
    int currentEvaluations;
    int currentIteration;         // 當前迭代次數
    
    // SNO 參數
    int nSInit;                   // N(s)^init: 探索者初始數量
    double nXRatio;               // N(x)^init 比例
    double nXFinalRatio;          // N(x)^final 比例
    int nNet;                     // N(N): 空間網大小
    int nRegion;                  // N(A): 區域數量
    double mF;                    // m^f: 突變參數
    double mC;                    // m^c: 交配參數
    double rhoMax;                // ρ_max: 最大比例
    int nNA;                      // N(N_a): 最大調整次數
    double alpha;                 // α 參數
    double beta;                  // β 參數
    
    // CEC21 變形選項
    bool useBias;                 // 是否使用 Bias 變形
    bool useShift;                // 是否使用 Shift 變形
    bool useRotation;             // 是否使用 Rotation 變形
    
    // 群體
    vector<Solution> explorers;   // 探索者 s
    vector<Solution> miners;      // 開發者 x
    vector<NetPoint> spaceNet;    // 空間網 N
    vector<Region> regions;       // 區域集合 R (原虛擬碼中的 R)
    vector<ArchiveItem> archive;  // 外部檔案 A
    
    // 最佳解
    Solution bestSolution;
    
    // 搜尋範圍
    vector<double> lowerBound;
    vector<double> upperBound;
    
    // 隨機數生成器
    mt19937 rng;
    
public:
    // 建構子
    SNO(string funcName, int dim, int maxEval,
        int ns_init, double nx_ratio, double nx_final, int n_net, int n_region,
        double m_f, double m_c, double rho_max, int n_na, double a, double b,
        bool use_bias = true, bool use_shift = true, bool use_rotation = true);
    
    // 主要執行函數
    void run();
    
    // 獲取結果
    double getBestFitness() const { return bestSolution.fitness; }
    vector<double> getBestPosition() const { return bestSolution.position; }
    
private:
    // ==================== 核心函數 ====================
    
    // 初始化
    void Initialization();
    
    // 搜尋期望值評估
    void ExpectedValue();
    
    // 潛力區域探索 (探索者)
    void RegionSearch();
    
    // 潛力點搜尋 (開發者)
    void PointSearch();
    
    // 空間網調整
    void NetAdjustment(const Solution& newSolution);
    
    // 群體數量調整
    void PopulationAdjustment();
    
    // ==================== 輔助函數 ====================
    
    // 評估適應值
    double evaluate(const vector<double>& position);
    
    // 更新最佳解
    void updateBest(const Solution& sol);
    
    // 邊界處理
    void boundCheck(vector<double>& position);
    
    // 產生隨機數
    double randDouble(double min, double max);
    int randInt(int min, int max);
    
    // 柯西分布
    double cauchyRandom(double location, double scale);
    
    // 常態分布
    double normalRandom(double mean, double stddev);
    
    // 輪盤法選擇
    int rouletteWheelSelection(const vector<double>& weights);
};

#endif // ALGORITHM_H