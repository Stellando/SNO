
#include <ctime>
#include <cstdlib>
#include "algorithm.h"
#include "functions.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <cmath>

using namespace std;

// ==================== SNO 建構子 ====================
SNO::SNO(string funcName, int dim, int maxEval,
         int ns_init, double nx_ratio, double nx_final, int n_net, int n_region,
         double m_f, double m_c, double rho_max, int n_na, double a, double b,
         bool use_bias, bool use_shift, bool use_rotation)
    : functionName(funcName), dimension(dim), maxEvaluations(maxEval),
      nSInit(ns_init), nXRatio(nx_ratio), nXFinalRatio(nx_final),
      nNet(n_net), nRegion(n_region), mF(m_f), mC(m_c), rhoMax(rho_max),
      nNA(n_na), alpha(a), beta(b), currentEvaluations(0), currentIteration(0),
      useBias(use_bias), useShift(use_shift), useRotation(use_rotation)
{
    // 初始化隨機數生成器
    rng.seed(static_cast<unsigned int>(time(nullptr)));
    
    // 初始化最佳解
    bestSolution = Solution(dimension);
    
    // 設定搜尋範圍 (使用 CEC21 標準範圍)
    TestFunction::getBounds(functionName, dimension, lowerBound, upperBound);
}

// ==================== 主要執行函數 ====================
void SNO::run() {
    
    // Step 1: 初始化
    Initialization();
    
    // 主要迭代循環
    while (currentEvaluations < maxEvaluations) {
        
        currentIteration++;  // 迭代計數
        
        // Step 2: 搜尋期望值評估
        ExpectedValue();
        
        // Step 3: 潛力區域探索 (探索者)
        RegionSearch();
        if (currentEvaluations >= maxEvaluations) break;
        
        // Step 4: 潛力點搜尋 (開發者)
        PointSearch();
        if (currentEvaluations >= maxEvaluations) break;
        
        // Step 5: 群體數量調整
        PopulationAdjustment();
        
        // 檢查是否達到最大評估次數
        if (currentEvaluations >= maxEvaluations) break;
    }
    
    // 換行輸出最終結果
    cout << endl;
}

// ==================== Step 1: 初始化 ====================
// 根據 Algorithm 14: Space Net Optimization: Initialization
void SNO::Initialization() {
    
    // Step 1: Set N(s) = N(s)^init, N(x) = N(x)^init
    int nExplorers = nSInit;
    int nMiners = static_cast<int>(nSInit * nXRatio);
    
    // Step 2: Randomly initialize population s and x by using Eq. (3.1)
    //S=探索者 X=開發者
    explorers.resize(nExplorers);
    for (int i = 0; i < nExplorers; i++) {
        explorers[i] = Solution(dimension);
        for (int d = 0; d < dimension; d++) {
            explorers[i].position[d] = randDouble(lowerBound[d], upperBound[d]);
        }
        explorers[i].fitness = evaluate(explorers[i].position);
        updateBest(explorers[i]);
    }
    
    miners.resize(nMiners);
    for (int i = 0; i < nMiners; i++) {
        miners[i] = Solution(dimension);
        for (int d = 0; d < dimension; d++) {
            miners[i].position[d] = randDouble(lowerBound[d], upperBound[d]);
        }
        miners[i].fitness = evaluate(miners[i].position);
        updateBest(miners[i]);
    }
    
    // Step 3: Set N(N) = N(N)^init
    // Step 4: Randomly initialize space net N by using Eq. (3.2)
    spaceNet.resize(nNet);
    for (int i = 0; i < nNet; i++) {
        spaceNet[i] = NetPoint(dimension);
        for (int d = 0; d < dimension; d++) {
            spaceNet[i].position[d] = randDouble(lowerBound[d], upperBound[d]);
        }
        spaceNet[i].fitness = evaluate(spaceNet[i].position);
    }
    
    // Step 5: Set N(R) = (√N(N) - 1)^2
    // 空間網是一個 W×W 的 2D 網格結構，其中 W = √N(N)
    int gridWidth = static_cast<int>(sqrt(nNet));
    
    // 檢查 nNet 是否為完全平方數
    if (gridWidth * gridWidth != nNet) {
        cout << "警告: N(N)=" << nNet << " 不是完全平方數！調整為 " << gridWidth*gridWidth << endl;
        nNet = gridWidth * gridWidth;
        spaceNet.resize(nNet);
        for (int i = 0; i < nNet; i++) {
            spaceNet[i] = NetPoint(dimension);
            for (int d = 0; d < dimension; d++) {
                spaceNet[i].position[d] = randDouble(lowerBound[d], upperBound[d]);
            }
            spaceNet[i].fitness = evaluate(spaceNet[i].position);
        }
    }
    
    // 計算正確的區域數量：(W-1)²
    // 每個區域由 4 個相鄰網點組成（2×2 的小格子）
    int correctNRegion = (gridWidth - 1) * (gridWidth - 1);
    
    // Step 6-9: for i = 1 to N(R) do
    //   Set v_a = 1, v_b = 1
    //   Set m^c_i = m^c, m^f_i = m^f
    regions.resize(correctNRegion);
    for (int i = 0; i < correctNRegion; i++) {
        regions[i].id = i;
        regions[i].expectedValue = 1.0;      // 初始期望值
        regions[i].visitCount = 1;           // v_a = 1 (避免除以0)
        regions[i].unvisitedCount = 1;       // v_b = 1
        regions[i].visitedThisRound = false; // 本輪未被訪問
        regions[i].bestFitness = 1e100;
        regions[i].recentImprovements.clear();
        regions[i].meanF = mF;               // 初始化參數均值
        regions[i].meanC = mC;
    }
    
    // 建立 2D 網格結構：每個區域包含 4 個相鄰網點
    // 網格點索引 k 映射到座標 (row, col) = (k / W, k % W)
    // 區域 r 對應其左上角網點的座標
    for (int r = 0; r < correctNRegion; r++) {
        int row = r / (gridWidth - 1);
        int col = r % (gridWidth - 1);
        
        // 一個區域包含 4 個點：左上、右上、左下、右下
        int p1 = row * gridWidth + col;              // Top-Left
        int p2 = row * gridWidth + (col + 1);        // Top-Right
        int p3 = (row + 1) * gridWidth + col;        // Bottom-Left
        int p4 = (row + 1) * gridWidth + (col + 1);  // Bottom-Right
        
        regions[r].netPointIndices = {p1, p2, p3, p4};
        
        // 注意：邊界網點會被多個區域共享
        // 例如：網點 (1,1) 會同時屬於區域 (0,0), (0,1), (1,0), (1,1)
    }
    
    // 更新 nRegion 為正確值
    nRegion = correctNRegion;
    
    // Step 10: Create an empty external archive A
    archive.clear();
    
    // Step 11: Evaluate s, x, N by using objective function
    // 已在上面的初始化過程中完成
    
    // Step 12: Update the best solution x*
    // 已在 evaluate 時通過 updateBest 完成
    
    // Step 13: F = F + N(s) + N(x) + N(N)
    // currentEvaluations 已在 evaluate 函數中累加
}

// ==================== Step 2: 搜尋期望值評估 ====================
// 根據 Algorithm 15: Space Net Optimization: Expected Value
void SNO::ExpectedValue() {
    
    // Step 1-4: Calculate raw values for each region
    for (int i = 0; i < nRegion; i++) {
        Region& region = regions[i];
        
        // Step 2: Calculate the visited ratio of region i
        // v_b / v_a = 區域多久沒被搜尋了 / 區域被選中進行搜尋的次數
        region.rawVisitRatio = static_cast<double>(region.unvisitedCount) / 
                               static_cast<double>(region.visitCount);
        
        // Step 3: Calculate the sum of improvement of region i
        // Sum of recent 4 improvements: Σ(f(R_{i,j}^{t-1}) - f(R_{i,j}^t))
        region.rawImprovementSum = 0.0;
        int count = min(4, static_cast<int>(region.recentImprovements.size()));
        for (int j = 0; j < count; j++) {
            region.rawImprovementSum += region.recentImprovements[j];
        }
        
        // Step 4: Find the best objective value of region i
        // Already stored in region.bestFitness
        // 計算 1.0 - c̃(f(R_{i,p}^t)) 其中 c̃ 是正規化函數
        // 這裡先暫存原始最佳值，稍後統一正規化
        region.rawBestValue = region.bestFitness;
    }
    
    // Step 6: Normalize each value using Eq. (3.4)
    // 找出每個指標的最大最小值進行正規化
    double minVisit = 1e100, maxVisit = -1e100;
    double minImprovement = 1e100, maxImprovement = -1e100;
    double minBest = 1e100, maxBest = -1e100;
    
    for (int i = 0; i < nRegion; i++) {
        minVisit = min(minVisit, regions[i].rawVisitRatio);
        maxVisit = max(maxVisit, regions[i].rawVisitRatio);
        minImprovement = min(minImprovement, regions[i].rawImprovementSum);
        maxImprovement = max(maxImprovement, regions[i].rawImprovementSum);
        minBest = min(minBest, regions[i].rawBestValue);
        maxBest = max(maxBest, regions[i].rawBestValue);
    }
    
    // 正規化函數 c̃(x) = (x - min) / (max - min + ε)
    auto normalize = [](double x, double minVal, double maxVal) -> double {
        double range = maxVal - minVal;
        if (range < 1e-10) return 0.5;  // 如果範圍太小，返回中間值
        return (x - minVal) / range;
    };
    
    // Step 7-8: Calculate the expected value e_i by using Eq. (3.3)
    // 計算動態權重 δ：從 2 線性遞減至 1
    double progress = static_cast<double>(currentEvaluations) / maxEvaluations;
    double weight_delta = 2.0 - 1.0 * progress;  // 2.0 → 1.0 線性遞減
    
    for (int i = 0; i < nRegion; i++) {
        Region& region = regions[i];
        
        // 正規化各項指標
        double normVisit = normalize(region.rawVisitRatio, minVisit, maxVisit);
        double normImprovement = normalize(region.rawImprovementSum, minImprovement, maxImprovement);
        double normBest = normalize(region.rawBestValue, minBest, maxBest);
        
        // 根據公式 (3.3):
        // e_i = c̃(v_{b,i}/n_{a,i}) + c̃(Σ(improvements)) + δ · (1.0 - c̃(f(R_{i,p}^t)))
        // 其中 δ 是動態權重，從 2 遞減至 1（論文第44頁）
        
        region.expectedValue = normVisit + normImprovement + weight_delta * (1.0 - normBest);
        
        // 確保期望值為正
        if (region.expectedValue < 0) region.expectedValue = 0.0;
    }
}

// ==================== Step 3: 潛力區域探索 (探索者) ====================
// 根據 Algorithm 16: Space Net Optimization: Region Search
void SNO::RegionSearch() {

    // 重置區域的輔助標記
    for(auto& r : regions) r.visitedThisRound = false;

    // Step 1: Rank all regions according to the expected value, then construct a roulette wheel
    vector<pair<double, int>> rankedRegions;  // <expectedValue, regionID>
    for (int i = 0; i < nRegion; i++) {
        rankedRegions.push_back({regions[i].expectedValue, i});
    }
    sort(rankedRegions.begin(), rankedRegions.end(), greater<pair<double, int>>());
    
    // 計算本輪參與輪盤的區域數量（隨時間遞減）
    // 論文第5頁：「每個迭代建立輪盤的區域數量會隨耗用評估次數增加而逐漸遞減」
    // 採用非線性遞減：從 100% 緩慢遞減到 80%，確保前期充分探索
    double progress = static_cast<double>(currentEvaluations) / maxEvaluations;
    // 使用平方根函數實現緩慢遞減：sqrt(progress) 比 progress 增長更慢
    int nActiveRegions = static_cast<int>(nRegion * (1.0 - 0.2 * sqrt(progress)));
    nActiveRegions = max(1, nActiveRegions);  // 至少保留1個區域
    
    // 只使用排名前 nActiveRegions 的區域建立輪盤
    // 準備輪盤法的權重（只包含活躍區域）
    vector<double> weights;
    vector<int> activeRegionIDs;
    for (int i = 0; i < nActiveRegions; i++) {
        int regionID = rankedRegions[i].second;
        weights.push_back(regions[regionID].expectedValue);
        activeRegionIDs.push_back(regionID);
    }
    
    // Step 9: Create the table of successful parameter S^f and S^c
    vector<double> Sf_table;  // 所有成功的 f 參數
    vector<double> Sc_table;  // 所有成功的 c 參數
    
    // Step 2: for i = 1 to N(s) do
    for (size_t i = 0; i < explorers.size(); i++) {
        // 檢查評估次數限制
        if (currentEvaluations >= maxEvaluations) break;
        
        // Step 3: Using roulette wheel selection to select the region R_s
        // 從活躍區域中選擇（輪盤返回的是 weights 中的索引）
        int selectedIdx = rouletteWheelSelection(weights);
        int selectedRegionID = activeRegionIDs[selectedIdx];  // 映射回真實區域ID
        Region& region = regions[selectedRegionID];
        
        // 標記該區域在本輪被訪問
        if (!region.visitedThisRound) {
            region.visitedThisRound = true;
            region.visitCount++;  // v_a 增加
            region.unvisitedCount = 1;  // v_b 重置為 1
        }
        
        // Step 5: Generate parameter f_i and c_i by using Eq. (3.5) and Eq. (3.6)
        // f_i = rand(m_s^f, 0.1, c)  - Cauchy distribution
        // c_i = rand(m_s^c, 0.1, n)  - Normal distribution
        double f = cauchyRandom(region.meanF, 0.1);
        double c = normalRandom(region.meanC, 0.1);
        
        // 確保參數在合理範圍內
        // 論文第6頁：「若 f_i 大於 1 時，將直接設為 1」
        f = max(0.0, min(1.0, f));
        c = max(0.0, min(1.0, c));
        
        // Step 6: Select the net point N_s by using Eq. (3.7)
        // N_s = { Tournament(R_s), if rand(0,1,u) < δ_{1,0}^{0,1}
        //       { R_{s,b},         otherwise
        NetPoint* selectedNet = nullptr;
        
        // 動態參數 δ_{1,0}^{0,1}：從 1.0 (前期) 線性遞減至 0.0 (後期)
        // 論文：「在搜尋前期有較高的機率使用競賽選擇法...後期則有較高機率直接選擇區域中最好的」
        double progress = static_cast<double>(currentEvaluations) / maxEvaluations;
        double delta_1_0 = 1.0 - progress;
        
        if (randDouble(0, 1) < delta_1_0) {
            // 前期機率高：使用競賽選擇 (維持多樣性)
            if (!region.netPointIndices.empty()) {
                int tournamentSize = min(3, static_cast<int>(region.netPointIndices.size()));
                int bestIdx = region.netPointIndices[randInt(0, region.netPointIndices.size() - 1)];
                double bestFit = spaceNet[bestIdx].fitness;
                
                for (int t = 1; t < tournamentSize; t++) {
                    int idx = region.netPointIndices[randInt(0, region.netPointIndices.size() - 1)];
                    if (spaceNet[idx].fitness < bestFit) {
                        bestIdx = idx;
                        bestFit = spaceNet[idx].fitness;
                    }
                }
                selectedNet = &spaceNet[bestIdx];
            }
        } else {
            // 後期機率高：直接選最好的 (加速收斂)
            if (!region.netPointIndices.empty()) {
                int bestIdx = region.netPointIndices[0];
                double bestFit = spaceNet[bestIdx].fitness;
                
                for (int idx : region.netPointIndices) {
                    if (spaceNet[idx].fitness < bestFit) {
                        bestIdx = idx;
                        bestFit = spaceNet[idx].fitness;
                    }
                }
                selectedNet = &spaceNet[bestIdx];
            }
        }
        
        // Step 7: Generate the new solution s_i' by using Eq. (3.8)
        Solution newSolution(dimension);
        double F_r = progress;  // 進度比例
        
        if (selectedNet != nullptr) {
            // 嚴格保證 r1 != r2 != i
            int r1, r2;
            do { r1 = randInt(0, explorers.size() - 1); } while (r1 == static_cast<int>(i) && explorers.size() > 1);
            do { r2 = randInt(0, explorers.size() - 1); } while ((r2 == static_cast<int>(i) || r2 == r1) && explorers.size() > 2);
            
            double threshold = pow(F_r, alpha);
            
            for (int d = 0; d < dimension; d++) {
                if (randDouble(0, 1) < threshold) {
                    // 後期策略（Fr^α 大時機率高）：以網點為中心
                    newSolution.position[d] = selectedNet->position[d] + 
                        f * (explorers[r1].position[d] - explorers[r2].position[d]);
                } else {
                    // 前期策略（Fr^α 小時機率高）：以當前解為中心
                    newSolution.position[d] = explorers[i].position[d] + 
                        f * (selectedNet->position[d] - explorers[r2].position[d]);
                }
            }
        } else {
            // 如果沒有可用的網點，使用隨機探索者
            int r1, r2;
            do { r1 = randInt(0, explorers.size() - 1); } while (r1 == static_cast<int>(i) && explorers.size() > 1);
            do { r2 = randInt(0, explorers.size() - 1); } while ((r2 == static_cast<int>(i) || r2 == r1) && explorers.size() > 2);
            
            for (int d = 0; d < dimension; d++) {
                newSolution.position[d] = explorers[i].position[d] + 
                    f * (explorers[r1].position[d] - explorers[r2].position[d]);
            }
        }
        
        boundCheck(newSolution.position);
        newSolution.fitness = evaluate(newSolution.position);
        
        // Step 10-16: 判斷是否成功並更新
        // Step 11: if f(s_i') < f(s_i^t) then
        if (newSolution.fitness < explorers[i].fitness) {
            double improvement = explorers[i].fitness - newSolution.fitness;
            
            // Step 12: f_i → S^f, c_i → S^c, s_i^t → A
            Sf_table.push_back(f);
            Sc_table.push_back(c);
            archive.push_back(ArchiveItem(explorers[i], currentIteration));
            
            // 記錄最近的改進值（保留最近4次）
            region.recentImprovements.insert(region.recentImprovements.begin(), improvement);
            if (region.recentImprovements.size() > 4) {
                region.recentImprovements.pop_back();
            }
            
            region.successfulF.push_back(f);
            region.successfulC.push_back(c);
            
            // Step 13: s_i^{t+1} = s_i'
            explorers[i] = newSolution;
            updateBest(newSolution);
            
            // Step 20-21: 觸發空間網調整
            NetAdjustment(newSolution);
        } else {
            // Step 14-15: else s_i^{t+1} = s_i^t
            // 保持不變
        }
        
        // 更新區域最佳值
        if (newSolution.fitness < region.bestFitness) {
            region.bestFitness = newSolution.fitness;
        }
    }
    
    // Step 18: F = F + N(s) - 已在 evaluate 中計算
    
    // Step 19: Update the best solution x* - 已在 updateBest 中完成
    
    // Step 23-28: 更新每個區域的參數
    for (int i = 0; i < nRegion; i++) {
        Region& region = regions[i];
        
        // Step 24: 更新 v_b (unvisitedCount)
        // 如果本輪沒被訪問，v_b 加 1（隱含：被訪問過的已在上面重置為 1）
        if (!region.visitedThisRound) {
            region.unvisitedCount++;
        }
        
        // Step 25-27: 若 v_b > 1，則重置 v_a = 1（防止強者恆強機制）
        if (region.unvisitedCount > 1) {
            region.visitCount = 1;
        }
        
        // Step 28: Update the parameter m_i^c and m_i^f by using Eq. (3.9)
        if (region.successfulF.size() > 0) {
            // 使用公式 (3.9) 和 (3.10) 計算加權平均
            // m_i = Σ(w_{i,j} · (S_{i,j})^2) / Σ(w_{i,j} · S_{i,j})
            // w_{i,j} = (f(s^t_{Si,k}) - f(s^{t+1}_{Si,k})) / Σ(f(s^t_{Si,k}) - f(s^{t+1}_{Si,k}))
            
            int N = region.successfulF.size();
            vector<double> weights_param(N, 1.0 / N);  // 簡化：使用均等權重
            
            // 如果有改進值資訊，使用加權平均
            if (region.recentImprovements.size() > 0 && region.recentImprovements.size() == region.successfulF.size()) {
                double sumImprovement = 0.0;
                for (double imp : region.recentImprovements) {
                    sumImprovement += imp;
                }
                if (sumImprovement > 0) {
                    for (size_t j = 0; j < region.recentImprovements.size(); j++) {
                        weights_param[j] = region.recentImprovements[j] / sumImprovement;
                    }
                }
            }
            
            // 計算 m_i^f
            double numerator_f = 0.0, denominator_f = 0.0;
            for (int j = 0; j < N; j++) {
                numerator_f += weights_param[j] * region.successfulF[j] * region.successfulF[j];
                denominator_f += weights_param[j] * region.successfulF[j];
            }
            if (denominator_f > 1e-10) {
                region.meanF = numerator_f / denominator_f;
            }
            
            // 計算 m_i^c
            double numerator_c = 0.0, denominator_c = 0.0;
            for (int j = 0; j < N; j++) {
                numerator_c += weights_param[j] * region.successfulC[j] * region.successfulC[j];
                denominator_c += weights_param[j] * region.successfulC[j];
            }
            if (denominator_c > 1e-10) {
                region.meanC = numerator_c / denominator_c;
            }
            
            // 限制範圍
            region.meanF = max(0.1, min(1.0, region.meanF));
            region.meanC = max(0.1, min(1.0, region.meanC));
            
            // 【關鍵修正】：清空成功參數記錄，以便下一代重新收集
            // 這確保參數自適應是基於當前世代的成功經驗，而非全局累積
            region.successfulF.clear();
            region.successfulC.clear();
        }
    }
}

// ==================== Step 4: 潛力點搜尋 (開發者) ====================
// 根據 Algorithm 17: Space Net Optimization: Point Search
void SNO::PointSearch() {
    
    // Step 1: for i = 1 to N(x) do
    for (size_t i = 0; i < miners.size(); i++) {
        // 檢查評估次數限制
        if (currentEvaluations >= maxEvaluations) break;
        
        // Step 2: Rank all net points according to objective value
        vector<int> sortedIndices(nNet);
        for (int j = 0; j < nNet; j++) sortedIndices[j] = j;
        
        sort(sortedIndices.begin(), sortedIndices.end(), 
             [this](int a, int b) { return spaceNet[a].fitness < spaceNet[b].fitness; });
        
        // Step 3: Select N_s from the top δ_{ρ_max}^{0,1} · N(N) of net points
        // 動態計算搜尋範圍比例：從 rhoMax 線性遞減到 0.05（最少保留5%最優網格點）
        // 論文：「被搜尋的網格點比例會隨著已耗用評估次數增加而逐漸遞減」
        double progress = static_cast<double>(currentEvaluations) / maxEvaluations;
        double currentRho = rhoMax * (1.0 - 0.95 * progress);  // rhoMax → 0.05*rhoMax
        int nTopPoints = max(1, static_cast<int>(nNet * currentRho));
        
        // 從前 currentRho 比例的網格點中選擇一個
        int selectedIdx = sortedIndices[randInt(0, nTopPoints - 1)];
        NetPoint& selectedNet = spaceNet[selectedIdx];
        
        // Step 4: Randomly select a region and get the parameter m^f, m^c
        int randomRegion = randInt(0, nRegion - 1);
        Region& region = regions[randomRegion];
        
        // Step 5: Generate the parameter f_i and c_i by using Eq. (3.5) and Eq. (3.6)
        double f = cauchyRandom(region.meanF, 0.1);
        double c = normalRandom(region.meanC, 0.1);
        
        // 確保參數在合理範圍內
        // 論文第6頁：「若 f_i 大於 1 時，將直接設為 1」
        f = max(0.0, min(1.0, f));
        c = max(0.0, min(1.0, c));
        
        // Step 6: Generate the new solution x'_θ by using Eq. (3.11)
        Solution newSolution(dimension);
        double F_r = progress;  // 進度比例（使用上面已計算的 progress）
        
        // 嚴格保證 r1 != r2 != i
        int r1, r2;
        do { r1 = randInt(0, miners.size() - 1); } while (r1 == static_cast<int>(i) && miners.size() > 1);
        do { r2 = randInt(0, miners.size() - 1); } while ((r2 == static_cast<int>(i) || r2 == r1) && miners.size() > 2);
        
        // Eq. (3.11):
        // x'_{θ,j} = { N_{s,j} + f_i · (x^t_{r1,j} - x^t_{r2,j}), if rand(0,1,u) < F_r^β
        //            { x^t_{θ,j} + f_i · (x^t_{r1,j} - x^t_{r2,j}), otherwise
        double threshold = pow(F_r, beta);
        
        for (int d = 0; d < dimension; d++) {
            if (randDouble(0, 1) < threshold) {
                // 前期策略：以網格點為中心
                newSolution.position[d] = selectedNet.position[d] + 
                    f * (miners[r1].position[d] - miners[r2].position[d]);
            } else {
                // 後期策略：以當前開發者為中心
                newSolution.position[d] = miners[i].position[d] + 
                    f * (miners[r1].position[d] - miners[r2].position[d]);
            }
        }
        
        boundCheck(newSolution.position);
        newSolution.fitness = evaluate(newSolution.position);
        
        // Step 7-11: 判斷是否接受新解
        // Step 7: if f(x'_θ) < f(x^t_θ) then
        if (newSolution.fitness < miners[i].fitness) {
            // Step 8: x^{t+1}_θ = x'_θ
            miners[i] = newSolution;
            updateBest(newSolution);
        } else {
            // Step 9-10: else x^{t+1}_θ = x^t_θ
            // 保持不變
        }
        // Step 12-13: F = F + 1, N = NetAdjustment(...)
        NetAdjustment(newSolution);
    }
    
    // Step 14: end for
    
    // Step 15: Update the best solution x*
    // 已在 updateBest 中完成
}

// ==================== Step 5: 空間網調整 ====================
// 根據 Algorithm 18: Space Net Optimization: Net Adjustment
void SNO::NetAdjustment(const Solution& newSolution) {
    
    // Input: The history solution z' (newSolution)
    const Solution& z_prime = newSolution;
    
    // Step 1: Rank all net points according to the distance between z' and each net point
    vector<pair<double, int>> distances;  // <距離, 索引>
    
    for (int i = 0; i < nNet; i++) {
        double dist = 0.0;
        for (int d = 0; d < dimension; d++) {
            double diff = spaceNet[i].position[d] - z_prime.position[d];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        distances.push_back({dist, i});
    }
    
    // 排序：距離由近到遠
    sort(distances.begin(), distances.end());
    
    // Step 2: Select the net points which need to be adjusted N_a
    // 動態調整數量：從 1 (前期) 逐漸增加至 nNA (後期)
    // 論文：「搜尋前期...網格點被吸引的數量較少...搜尋後期逐漸增加網格點被吸引的數量」
    double progress = static_cast<double>(currentEvaluations) / maxEvaluations;
    int dynamic_nNA = 1 + static_cast<int>((nNA - 1) * progress);
    int nAdjust = min(dynamic_nNA, nNet);
    
    // Step 3: for i = 1 to N(N_a) do
    for (int i = 0; i < nAdjust; i++) {
        // 檢查評估次數限制
        if (currentEvaluations >= maxEvaluations) break;
        
        int netIdx = distances[i].second;
        NetPoint& currentNet = spaceNet[netIdx];
        
        // Step 4: Randomly select a region and get the parameter m^f, m^c
        int randomRegion = randInt(0, nRegion - 1);
        Region& region = regions[randomRegion];
        
        // Step 5: Generate the parameter f_i and c_i by using Eq. (3.5) and Eq. (3.6)
        double f = cauchyRandom(region.meanF, 0.1);
        double c = normalRandom(region.meanC, 0.1);
        
        // 確保參數在合理範圍內
        // 論文第6頁：「若 f_i 大於 1 時，將直接設為 1」
        f = max(0.0, min(1.0, f));
        c = max(0.0, min(1.0, c));
        
        // 嚴格保證 r1 != r2
        int r1 = -1, r2 = -1;
        if (explorers.size() > 0) {
            r1 = randInt(0, explorers.size() - 1);
            if (explorers.size() > 1) {
                do { r2 = randInt(0, explorers.size() - 1); } while (r2 == r1);
            } else {
                r2 = r1; // 只有一個探索者時無法避免
            }
        }
        
        // Step 6: Generate the temporary net point N^1_{a,i} by using Eq. (3.12)
        // N^1_{a,i,j} = z'_j + f_i · (z^t_{r1,j} - z^t_{r2,j})
        NetPoint temp1(dimension);
        if (r1 >= 0 && r2 >= 0) {
            for (int d = 0; d < dimension; d++) {
                temp1.position[d] = z_prime.position[d] + 
                    f * (explorers[r1].position[d] - explorers[r2].position[d]);
            }
        } else {
            // 如果沒有探索者，使用當前網點
            temp1.position = currentNet.position;
        }
        boundCheck(temp1.position);
        temp1.fitness = evaluate(temp1.position);
        
        // Step 7: Generate the temporary net point N^2_{a,i} by using Eq. (3.13)
        // N^2_{a,i,j} = N_{a,i,j} + f_i · (z'_j - N_{a,i,j}) + f_i · (z^t_{r1,j} - z^t_{r2,j})
        NetPoint temp2(dimension);
        if (r1 >= 0 && r2 >= 0) {
            for (int d = 0; d < dimension; d++) {
                temp2.position[d] = currentNet.position[d] + 
                    f * (z_prime.position[d] - currentNet.position[d]) + 
                    f * (explorers[r1].position[d] - explorers[r2].position[d]);
            }
        } else {
            // 如果沒有探索者，簡化為向 z' 移動
            for (int d = 0; d < dimension; d++) {
                temp2.position[d] = currentNet.position[d] + 
                    f * (z_prime.position[d] - currentNet.position[d]);
            }
        }
        boundCheck(temp2.position);
        temp2.fitness = evaluate(temp2.position);
        
        // Step 8: Select N'_{a,i} by using Eq. (3.14)
        // D 函數：從 temp1 和 temp2 中選擇距離參考點最近的
        // 【關鍵修正】：絕對不能將參考點本身納入候選，否則距離永遠為0
        NetPoint selectedNet(dimension);
        double progress = static_cast<double>(currentEvaluations) / maxEvaluations;
        double F_r = progress;
        
        // 計算歐幾里得距離的輔助函數
        auto calcDistance = [this](const vector<double>& a, const vector<double>& b) -> double {
            double dist = 0.0;
            for (int d = 0; d < dimension; d++) {
                double diff = a[d] - b[d];
                dist += diff * diff;
            }
            return sqrt(dist);
        };
        
        if (i == 0) {
            // 只有排第一的點 (i=0) 直接變成 z'
            selectedNet.position = z_prime.position;
            selectedNet.fitness = z_prime.fitness;
        } else if (randDouble(0, 1) < F_r) {
            // 後期策略：選擇 temp1 或 temp2 中，距離 z' 較近的那一個
            // D(N^1_{a,i}, N^2_{a,i}, z') = 從 temp1, temp2 選擇距離 z' 最近的
            double dist1 = calcDistance(temp1.position, z_prime.position);
            double dist2 = calcDistance(temp2.position, z_prime.position);
            
            if (dist1 < dist2) {
                selectedNet = temp1;
            } else {
                selectedNet = temp2;
            }
        } else {
            // 前期策略：選擇 temp1 或 temp2 中，距離原網點 (currentNet) 較近的那一個
            // D(N^1_{a,i}, N^2_{a,i}, N_{a,i}) = 從 temp1, temp2 選擇距離 currentNet 最近的
            double dist1 = calcDistance(temp1.position, currentNet.position);
            double dist2 = calcDistance(temp2.position, currentNet.position);
            
            if (dist1 < dist2) {
                selectedNet = temp1;
            } else {
                selectedNet = temp2;
            }
        }
        
        // Step 9-13: 判斷是否接受新網點
        // Step 9: if f(N'_{a,i}) < f(N_{a,i}) then
        if (selectedNet.fitness < currentNet.fitness) {
            // Step 10: N_{a,i} = N'_{a,i}
            spaceNet[netIdx] = selectedNet;
        } else {
            // Step 11-12: else terminate the following adjustment of net point
            // 如果沒有改善，提前終止後續調整
            break;
        }
    }
    
    // Step 14: F = F + 1 - 已在 evaluate 中計算
    
    // Step 15: end for
    
    // Step 16: Update the best solution x* - 已在 updateBest 中完成
}

// ==================== Step 6: 群體數量調整 ====================
// 根據 Algorithm 19: Space Net Optimization: Population Adjustment
void SNO::PopulationAdjustment() {
    
    // Input: The explorer s, the miner x, the space net N, the external archive A, the ratio of better net points ρ_max
    
    // Step 1: Calculate the new size of explorer N(s) by using Eq. (3.15)
    // N(z) = (N(z)^final - N(z)^init) · √F_r^{(1-√F_r)} + N(z)^init,  z ∈ {s, x}
    double F_r = static_cast<double>(currentEvaluations) / maxEvaluations;
    double exponent = 1.0 - sqrt(F_r);
    double adjust_factor = sqrt(pow(F_r, exponent));
    
    // 計算探索者目標數量：N(s) = (0 - N(s)^init) · √F_r^{(1-√F_r)} + N(s)^init
    int nExplorersTarget = static_cast<int>((0 - nSInit) * adjust_factor + nSInit);
    nExplorersTarget = max(0, nExplorersTarget);  // 確保非負
    
    // Step 2: Delete the worse solutions from explorer and resize the external archive A
    if (explorers.size() > static_cast<size_t>(nExplorersTarget)) {
        // 排序：適應值好的在前面
        sort(explorers.begin(), explorers.end(), 
             [](const Solution& a, const Solution& b) { return a.fitness < b.fitness; });
        
        // 將被移除的探索者加入外部檔案 A
        for (size_t i = nExplorersTarget; i < explorers.size(); i++) {
            archive.push_back(ArchiveItem(explorers[i], currentIteration));
        }
        
        explorers.resize(nExplorersTarget);
    }
    
    // Step 3: Calculate the new size of miner N(x) by using Eq. (3.15)
    // N(x) = (N(x)^final - N(x)^init) · √F_r^{(1-√F_r)} + N(x)^init
    int nXInit = static_cast<int>(nSInit * nXRatio);
    int nXFinal = static_cast<int>(nSInit * nXFinalRatio);
    int nMinersTarget = static_cast<int>((nXFinal - nXInit) * adjust_factor + nXInit);
    
    // Step 4-9: for i = 1 to (N(N) - |x|) do
    // 如果需要增加開發者數量
    while (miners.size() < static_cast<size_t>(nMinersTarget) && currentEvaluations < maxEvaluations) {
        
        // Step 5: Randomly select N_s from the top ρ_max^{0,1} · N(N) of net points
        // 先對網點按適應值排序
        vector<int> sortedIndices(nNet);
        for (int j = 0; j < nNet; j++) sortedIndices[j] = j;
        
        sort(sortedIndices.begin(), sortedIndices.end(), 
             [this](int a, int b) { return spaceNet[a].fitness < spaceNet[b].fitness; });
        
        // 從前 ρ_max 比例的網格點中選擇一個
        int nTopPoints = max(1, static_cast<int>(nNet * rhoMax));
        int selectedIdx = sortedIndices[randInt(0, nTopPoints - 1)];
        NetPoint& selectedNet = spaceNet[selectedIdx];
        
        // Step 6: Create a new solution x' by using Eq. (3.16) and evaluate the objective value of x'
        Solution newMiner(dimension);
        
        // Eq. (3.16):
        // x'_j = { F_r^2 · N_{s,j} + (1 - F_r^2) · rand(L_j, U_j, u),  if rand(0,1,u) < 0.5
        //        { N_{s,j},                                             otherwise
        
        for (int d = 0; d < dimension; d++) {
            if (randDouble(0, 1) < 0.5) {
                // 根據進度在網點和隨機值之間插值
                double F_r_squared = F_r * F_r;
                newMiner.position[d] = F_r_squared * selectedNet.position[d] + 
                    (1.0 - F_r_squared) * randDouble(lowerBound[d], upperBound[d]);
            } else {
                // 直接使用網點位置
                newMiner.position[d] = selectedNet.position[d];
            }
        }
        
        boundCheck(newMiner.position);
        newMiner.fitness = evaluate(newMiner.position);
        
        // Step 7: Put the new solution x' into the miner
        miners.push_back(newMiner);
        updateBest(newMiner);
        
        // Step 8: F = F + 1 - 已在 evaluate 中計算
    }
    
    // Step 9: end for
    
    // Step 10: Update the best solution x*
    // 已在 updateBest 中完成
}

// ==================== 輔助函數 ====================

double SNO::evaluate(const vector<double>& position) {
    currentEvaluations++;
    
    // 每100次評估輸出一次進度
    if (currentEvaluations % 100 == 0) {
        cout << "\r  [" << currentEvaluations << "/" << maxEvaluations 
             << "] Best: " << scientific << setprecision(6) << bestSolution.fitness 
             << flush;
    }
    
    // 呼叫 CEC21 測試函數，傳遞變形選項
    return TestFunction::evaluate(functionName, position, dimension, useBias, useShift, useRotation);
}

void SNO::updateBest(const Solution& sol) {
    if (sol.fitness < bestSolution.fitness) {
        bestSolution = sol;
    }
}

void SNO::boundCheck(vector<double>& position) {
    for (int d = 0; d < dimension; d++) {
        if (position[d] < lowerBound[d]) {
            // 反射法：超出下界則反彈
            position[d] = lowerBound[d] + (lowerBound[d] - position[d]);
            // 如果反彈後還是超出上界（極端情況），則強制設為邊界
            if (position[d] > upperBound[d]) position[d] = upperBound[d];
        } else if (position[d] > upperBound[d]) {
            // 反射法：超出上界則反彈
            position[d] = upperBound[d] - (position[d] - upperBound[d]);
            if (position[d] < lowerBound[d]) position[d] = lowerBound[d];
        }
    }
}

double SNO::randDouble(double min, double max) {
    uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

int SNO::randInt(int min, int max) {
    uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

double SNO::cauchyRandom(double location, double scale) {
    cauchy_distribution<double> dist(location, scale);
    return dist(rng);
}

double SNO::normalRandom(double mean, double stddev) {
    normal_distribution<double> dist(mean, stddev);
    return dist(rng);
}

int SNO::rouletteWheelSelection(const vector<double>& weights) {
    double sum = 0.0;
    for (double w : weights) sum += w;
    
    if (sum <= 0) return randInt(0, weights.size() - 1);
    
    double r = randDouble(0, sum);
    double cumulative = 0.0;
    
    for (size_t i = 0; i < weights.size(); i++) {
        cumulative += weights[i];
        if (r <= cumulative) return i;
    }
    
    return weights.size() - 1;
}

