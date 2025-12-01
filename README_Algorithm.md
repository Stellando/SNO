# SNO 演算法架構說明文件

## 目錄
1. [演算法概述](#演算法概述)
2. [資料結構](#資料結構)
3. [SNO 類別成員變數](#sno-類別成員變數)
4. [函式功能說明](#函式功能說明)
5. [執行流程](#執行流程)
6. [演算法核心機制](#演算法核心機制)

---

## 演算法概述

**SNO (Space Net Optimization)** 是一種用於解決連續型最佳化問題的超啟發式演算法。該演算法結合了：
- **探索者 (Explorers)**：負責全域搜尋，探索潛力區域
- **開發者 (Miners)**：負責局部搜尋，開發潛力點
- **空間網 (Space Net)**：劃分搜尋空間，引導搜尋方向
- **區域機制 (Regions)**：評估不同區域的搜尋期望值，實現自適應參數調整

---

## 資料結構

### 1. `Solution` - 解的結構
```cpp
struct Solution {
    vector<double> position;  // 位置向量（決策變數）
    double fitness;           // 適應值（目標函數值）
}
```
**用途**：儲存單一解的資訊
- `position`: 在 D 維空間中的座標
- `fitness`: 該解的目標函數值（越小越好）

---

### 2. `NetPoint` - 空間網點結構
```cpp
struct NetPoint {
    vector<double> position;  // 網格點位置
    double fitness;           // 目標函數值
    int regionID;             // 所屬區域ID
}
```
**用途**：空間網的組成元素，用於劃分和引導搜尋空間
- 將搜尋空間劃分為多個區域
- 作為探索者和開發者的參考點
- 隨著搜尋過程動態調整位置

---

### 3. `Region` - 區域資訊結構
```cpp
struct Region {
    int id;                                    // 區域ID
    vector<int> netPointIndices;               // 所屬的網格點索引列表
    
    // === 搜尋期望值相關 ===
    double expectedValue;                      // 搜尋期望值 e_i (Eq. 3.3)
    int visitCount;                            // v_a: 區域被造訪累計次數
    int unvisitedCount;                        // v_b: 區域未被造訪計數器
    bool visitedThisRound;                     // 本輪是否被訪問（輔助標記）
    vector<double> recentImprovements;         // 最近4次改進值
    double bestFitness;                        // 區域最佳值 f(R_{i,p}^t)
    
    // === 正規化前的原始值 ===
    double rawVisitRatio;                      // 原始訪問比例 v_b/v_a
    double rawImprovementSum;                  // 原始改進總和
    double rawBestValue;                       // 原始最佳值項
    
    // === 參數自適應機制 ===
    vector<double> successfulF;                // 成功的突變因子 f 歷史記錄
    vector<double> successfulC;                // 成功的交配率 c 歷史記錄
    double meanF;                              // 參數均值 m_i^f (Eq. 3.9)
    double meanC;                              // 參數均值 m_i^c (Eq. 3.9)
}
```
**用途**：管理搜尋空間的子區域
- **期望值計算**：根據訪問頻率、改進程度、最佳值計算區域價值
- **參數自適應**：記錄成功參數，動態調整 meanF 和 meanC
- **防止冷落機制**：透過 v_a 和 v_b 確保所有區域都被探索

---

### 4. `ArchiveItem` - 外部檔案結構
```cpp
struct ArchiveItem {
    vector<double> position;  // 被拋棄的解的位置
    double fitness;           // 適應值
    int iteration;            // 被拋棄時的迭代次數
}
```
**用途**：儲存被淘汰的解，避免資訊遺失
- 當探索者被更好的解取代時，舊解存入檔案
- 當探索者數量縮減時，被移除的解存入檔案

---

## SNO 類別成員變數

### 測試函數相關
```cpp
string functionName;        // 測試函數名稱（如 "BentCigar"）
int dimension;              // 問題維度 D
int maxEvaluations;         // 最大評估次數
int currentEvaluations;     // 當前評估次數
int currentIteration;       // 當前迭代次數
```

### SNO 參數
```cpp
int nSInit;                 // N(s)^init: 探索者初始數量（預設 200）
double nXRatio;             // N(x)^init 比例（預設 0.1）
double nXFinalRatio;        // N(x)^final 比例（預設 0.2）
int nNet;                   // N(N): 空間網大小（預設 169）
int nRegion;                // N(R): 區域數量（預設 169）
double mF;                  // m^f: 突變參數初始值（預設 0.3）
double mC;                  // m^c: 交配參數初始值（預設 0.5）
double rhoMax;              // ρ_max: 最大比例（預設 0.5）
int nNA;                    // N(N_a): 最大調整次數（預設 10）
double alpha;               // α 參數（預設 1.0）
double beta;                // β 參數（預設 3.0）
```

### CEC21 變形選項
```cpp
bool useBias;               // 是否使用 Bias 變形
bool useShift;              // 是否使用 Shift 變形
bool useRotation;           // 是否使用 Rotation 變形
```

### 群體
```cpp
vector<Solution> explorers; // 探索者群體 s（全域搜尋）
vector<Solution> miners;    // 開發者群體 x（局部搜尋）
vector<NetPoint> spaceNet;  // 空間網 N（引導搜尋）
vector<Region> regions;     // 區域集合 R（區域管理）
vector<ArchiveItem> archive;// 外部檔案 A（淘汰解儲存）
```

### 其他
```cpp
Solution bestSolution;      // 全域最佳解 x*
vector<double> lowerBound;  // 搜尋範圍下界（預設 -100）
vector<double> upperBound;  // 搜尋範圍上界（預設 +100）
mt19937 rng;                // 隨機數生成器
```

---

## 函式功能說明

### 主要演算法函式

#### 1. `SNO::run()`
**功能**：演算法主循環
```cpp
void SNO::run()
```
**執行步驟**：
1. 呼叫 `Initialization()` 初始化
2. 迭代循環直到達到最大評估次數：
   - `ExpectedValue()` - 計算區域期望值
   - `RegionSearch()` - 探索者搜尋
   - `PointSearch()` - 開發者搜尋
   - `PopulationAdjustment()` - 群體數量調整

---

#### 2. `SNO::Initialization()`
**功能**：初始化所有群體和資料結構（Algorithm 14）
```cpp
void SNO::Initialization()
```
**執行步驟**：
1. 初始化探索者群體 (N(s)^init = 200)
2. 初始化開發者群體 (N(x)^init = 20)
3. 初始化空間網 (N(N) = 169)
4. 初始化區域資訊 (N(R) = 169)
   - 設定 v_a = 1, v_b = 1
   - 設定 meanF = 0.3, meanC = 0.5
5. 將空間網點分配到各區域
6. 建立空檔案 A
7. 評估所有個體的適應值

---

#### 3. `SNO::ExpectedValue()`
**功能**：計算每個區域的搜尋期望值（Algorithm 15）
```cpp
void SNO::ExpectedValue()
```
**公式 (3.3)**：
$$e_i = \tilde{c}(v_{b,i}/v_{a,i}) + \tilde{c}(\Sigma \text{improvements}) + \alpha \cdot (1.0 - \tilde{c}(f(R_{i,p}^t)))$$

**執行步驟**：
1. 計算每個區域的原始值：
   - `rawVisitRatio` = v_b / v_a（冷門程度）
   - `rawImprovementSum` = 最近4次改進值總和
   - `rawBestValue` = 區域最佳適應值
2. 正規化各指標（Eq. 3.4）：$\tilde{c}(x) = (x - \min) / (\max - \min)$
3. 計算期望值 e_i
4. 期望值越高 → 輪盤法選中機率越高

**意義**：
- 高 v_b/v_a → 很久沒被訪問 → 提高期望值
- 高改進總和 → 該區域有潛力 → 提高期望值
- 低最佳值 → 該區域品質好 → 提高期望值

---

#### 4. `SNO::RegionSearch()`
**功能**：探索者進行區域搜尋（Algorithm 16）
```cpp
void SNO::RegionSearch()
```
**執行步驟**：
1. **重置訪問標記**：所有區域的 `visitedThisRound = false`
2. **輪盤法選擇區域**：根據期望值選擇區域 R_s
3. **更新訪問狀態**：
   - 首次訪問：`visitCount++`, `unvisitedCount = 1`
4. **生成參數**（Eq. 3.5-3.6）：
   - f ~ Cauchy(meanF, 0.1)
   - c ~ Normal(meanC, 0.1)
5. **選擇網點**（Eq. 3.7）：
   - 10% 機率：Tournament 選擇
   - 90% 機率：選擇區域最佳網點
6. **生成新解**（Eq. 3.8）：
   ```
   前期 (Fr^α 小): s'_j = N_s + f·(s_r1 - s_r2)  [以網點為中心]
   後期 (Fr^α 大): s'_j = s_i + f·(N_s - s_r2)  [以當前解為中心]
   ```
7. **貪婪選擇**：若 f(s') < f(s_i)
   - 更新探索者
   - 記錄成功參數 → successfulF, successfulC
   - 舊解存入檔案 A
   - 觸發 `NetAdjustment()`
8. **更新區域參數**（Step 23-28）：
   - 未被訪問的區域：`unvisitedCount++`
   - 若 unvisitedCount > 1：`visitCount = 1`（防止強者恆強）
   - 更新 meanF 和 meanC（Eq. 3.9-3.10）

---

#### 5. `SNO::PointSearch()`
**功能**：開發者進行點搜尋（Algorithm 17）
```cpp
void SNO::PointSearch()
```
**執行步驟**：
1. **排序網點**：依適應值由好到壞排序
2. **選擇精英網點**：從前 ρ_max·N(N) 的網點中選一個
3. **隨機選擇區域**：獲取該區域的 meanF 和 meanC
4. **生成參數**：f ~ Cauchy, c ~ Normal
5. **生成新解**（Eq. 3.11）：
   ```
   前期 (Fr^β 小): x'_j = N_s + f·(x_r1 - x_r2)  [以網點為中心]
   後期 (Fr^β 大): x'_j = x_θ + f·(x_r1 - x_r2)  [以當前解為中心]
   ```
6. **貪婪選擇**：若 f(x') < f(x_θ)
   - 更新開發者
   - 觸發 `NetAdjustment()`

**與 RegionSearch 的差異**：
- RegionSearch：根據期望值選擇區域（自適應）
- PointSearch：選擇最佳網點（精英導向）
- β > α：開發者更快轉向局部搜尋

---

#### 6. `SNO::NetAdjustment()`
**功能**：調整空間網點位置（Algorithm 18）
```cpp
void SNO::NetAdjustment(const Solution& newSolution)
```
**執行步驟**：
1. **排序網點**：依與新解 z' 的距離排序
2. **選擇調整對象**：選最近的 N(N_a) 個網點（預設 10）
3. **對每個網點 i**：
   - 隨機選擇區域參數 meanF, meanC
   - 生成臨時網點 N¹（Eq. 3.12）：
     ```
     N¹_j = z'_j + f·(z_r1 - z_r2)
     ```
   - 生成臨時網點 N²（Eq. 3.13）：
     ```
     N²_j = N_a,j + f·(z'_j - N_a,j) + f·(z_r1 - z_r2)
     ```
   - **D 函數選擇**（Eq. 3.14，基於距離）：
     ```
     i=0:           選擇 z'
     Fr 大(後期):   從 {N¹, N², z'} 中選距離 z' 最近的
     Fr 小(前期):   從 {N¹, N², N_a} 中選距離 N_a 最近的
     ```
4. **貪婪更新**：若 f(N') < f(N_a) → 更新網點
5. **提早終止**：若無改善則 break

**意義**：
- 向有希望的解方向拉動網點
- 前期保持多樣性，後期向好解收斂
- 提早終止避免浪費評估次數

---

#### 7. `SNO::PopulationAdjustment()`
**功能**：動態調整群體大小（Algorithm 19）
```cpp
void SNO::PopulationAdjustment()
```
**公式 (3.15)**：
$$N(z) = (N(z)^{\text{final}} - N(z)^{\text{init}}) \cdot \sqrt{F_r^{(1-\sqrt{F_r})}} + N(z)^{\text{init}}$$

**執行步驟**：
1. **調整探索者數量**：
   - 計算目標數量 N(s)（從 200 → 0）
   - 排序探索者，保留前 N(s) 個最佳
   - 被移除的存入檔案 A
2. **調整開發者數量**：
   - 計算目標數量 N(x)（從 20 → 40）
   - 若不足，從精英網點生成新開發者（Eq. 3.16）：
     ```
     50%: x'_j = Fr²·N_s + (1-Fr²)·rand(L,U)
     50%: x'_j = N_s
     ```

**演化策略**：
- **前期**：多探索者 + 少開發者 → 全域搜尋
- **後期**：少探索者 + 多開發者 → 局部搜尋
- **非線性調整**：$\sqrt{F_r^{(1-\sqrt{F_r})}}$ 確保平滑過渡

---

### 輔助函式

#### 8. `SNO::evaluate()`
**功能**：評估解的適應值
```cpp
double SNO::evaluate(const vector<double>& position)
```
- 累加評估次數 `currentEvaluations++`
- 呼叫 CEC21 測試函數
- 根據 useBias, useShift, useRotation 選擇對應變形

#### 9. `SNO::updateBest()`
**功能**：更新全域最佳解
```cpp
void SNO::updateBest(const Solution& sol)
```

#### 10. `SNO::boundCheck()`
**功能**：邊界檢查，確保解在 [L, U] 範圍內
```cpp
void SNO::boundCheck(vector<double>& position)
```

#### 11. 隨機數生成函式
```cpp
double randDouble(double min, double max)        // 均勻分布
int randInt(int min, int max)                    // 整數均勻分布
double cauchyRandom(double location, double scale) // 柯西分布
double normalRandom(double mean, double stddev)  // 常態分布
```

#### 12. `SNO::rouletteWheelSelection()`
**功能**：輪盤法選擇
```cpp
int rouletteWheelSelection(const vector<double>& weights)
```
- 根據權重（期望值）選擇區域
- 權重越高，被選中機率越大

---

## 執行流程

### 整體流程圖
```
┌─────────────────────────────────────┐
│  1. Initialization                  │
│  - 初始化探索者、開發者、空間網      │
│  - 初始化區域資訊 (v_a=1, v_b=1)    │
│  - 評估所有個體                      │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  主迭代循環 (while F < MaxEval)     │
│                                     │
│  2. ExpectedValue                   │
│  - 計算每個區域的期望值              │
│  - 依據 v_b/v_a, 改進值, 最佳值     │
│                                     │
│  3. RegionSearch (探索者)           │
│  - 輪盤法選擇區域                    │
│  - 生成新解 (Eq. 3.8)               │
│  - 若改善 → NetAdjustment           │
│  - 更新區域參數 (meanF, meanC)      │
│                                     │
│  4. PointSearch (開發者)            │
│  - 選擇精英網點                      │
│  - 生成新解 (Eq. 3.11)              │
│  - 若改善 → NetAdjustment           │
│                                     │
│  5. PopulationAdjustment            │
│  - 調整探索者數量 (200→0)           │
│  - 調整開發者數量 (20→40)           │
│                                     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  輸出最佳解 x*                       │
└─────────────────────────────────────┘
```

### 單次迭代詳細流程

```
Iteration t:
│
├─ ExpectedValue()
│  ├─ 計算 rawVisitRatio = v_b / v_a
│  ├─ 計算 rawImprovementSum
│  ├─ 正規化所有指標
│  └─ 計算期望值 e_i (Eq. 3.3)
│
├─ RegionSearch() [對每個探索者]
│  ├─ 輪盤法選擇區域 R_s (依 e_i)
│  ├─ 標記區域被訪問 (visitCount++, unvisitedCount=1)
│  ├─ 生成參數 f~Cauchy, c~Normal
│  ├─ 選擇網點 N_s (Tournament 或 最佳)
│  ├─ 生成新解 s' (Eq. 3.8)
│  ├─ if f(s') < f(s_i):
│  │  ├─ s_i ← s'
│  │  ├─ 記錄成功參數
│  │  └─ NetAdjustment(s')
│  │     ├─ 選擇最近的 10 個網點
│  │     ├─ 生成 N¹ (Eq. 3.12), N² (Eq. 3.13)
│  │     ├─ D 函數選擇 N' (基於距離)
│  │     └─ 若 f(N') < f(N_a) → 更新網點
│  └─ 更新所有區域狀態
│     ├─ 未訪問區域 unvisitedCount++
│     ├─ 若 unvisitedCount>1 → visitCount=1
│     └─ 更新 meanF, meanC (Eq. 3.9-3.10)
│
├─ PointSearch() [對每個開發者]
│  ├─ 排序網點 (依適應值)
│  ├─ 選擇前 ρ_max·N(N) 的網點
│  ├─ 隨機選區域 → 獲取 meanF, meanC
│  ├─ 生成新解 x' (Eq. 3.11)
│  ├─ if f(x') < f(x_θ):
│  │  ├─ x_θ ← x'
│  │  └─ NetAdjustment(x')
│  └─ 更新最佳解
│
└─ PopulationAdjustment()
   ├─ 計算 Fr = currentEval / maxEval
   ├─ 計算探索者目標數 (Eq. 3.15)
   ├─ 淘汰較差探索者 → 存入檔案 A
   ├─ 計算開發者目標數 (Eq. 3.15)
   └─ 若不足，從精英網點生成新開發者 (Eq. 3.16)
```

---

## 演算法核心機制

### 1. 區域期望值機制
**目的**：智慧分配搜尋資源

**公式**：
$$e_i = \tilde{c}\left(\frac{v_{b,i}}{v_{a,i}}\right) + \tilde{c}(\Sigma \text{improvements}) + \alpha \cdot \left(1.0 - \tilde{c}(f(R_{i,p}^t))\right)$$

**三個因素**：
1. **v_b / v_a**：冷門程度（很久沒去的區域分數高）
2. **Σ improvements**：改進潛力（最近有改進的區域分數高）
3. **1.0 - c̃(f)**：區域品質（適應值好的區域分數高）

**防止強者恆強機制**：
- 若區域連續 2 次沒被選中（v_b > 1）
- 立即重置 v_a = 1
- 下次 e_i = v_b/1 會暴增 → 強迫被選中

---

### 2. 參數自適應機制
**目的**：每個區域維護自己的最佳參數

**更新公式 (Eq. 3.9-3.10)**：
$$m_i^f = \frac{\sum w_{i,j} \cdot (S_{i,j}^f)^2}{\sum w_{i,j} \cdot S_{i,j}^f}$$

**權重**：
$$w_{i,j} = \frac{f(s_i^t) - f(s_i^{t+1})}{\sum (f(s_i^t) - f(s_i^{t+1}))}$$

**意義**：
- 改進越大的參數，權重越高
- 每個區域學習自己的最佳 f 和 c
- 不同區域有不同搜尋策略

---

### 3. 動態搜尋策略
**前期（Fr 小）**：
- 探索者多（200）、開發者少（20）
- 以網點為中心生成新解
- D 函數選擇距離原網點近的
- **目標**：全域搜尋，保持多樣性

**後期（Fr 大）**：
- 探索者少（0）、開發者多（40）
- 以當前解為中心生成新解
- D 函數選擇距離新解近的
- **目標**：局部搜尋，精細開發

**轉換機制**：
- RegionSearch: 閾值 = Fr^α（α=1.0，線性轉換）
- PointSearch: 閾值 = Fr^β（β=3.0，快速轉換）

---

### 4. 空間網調整機制
**目的**：將網點拉向有希望的區域

**調整策略**：
1. 選擇最近的 10 個網點
2. 生成兩個候選點 N¹, N²
3. D 函數基於**距離**選擇（而非適應值）
4. 僅當有改善才更新，否則提早終止

**D 函數邏輯**：
```
if i == 0:
    選擇 z'（強制向新解收斂）
elif rand < Fr:
    從 {N¹, N², z'} 選距離 z' 最近的（後期：向好解收斂）
else:
    從 {N¹, N², N_a} 選距離 N_a 最近的（前期：保持多樣性）
```

---

### 5. CEC21 變形控制
**三位元控制**：
- **Bias**: 偏移測試函數的最佳解位置
- **Shift**: 平移搜尋空間
- **Rotation**: 旋轉座標系

**8 種組合**：
```
000: Basic           (最簡單)
001: Rotation
010: Shift
011: Shift+Rotation
100: Bias
101: Bias+Rotation
110: Bias+Shift
111: All             (最困難)
```

---

## 總結

SNO 演算法的核心優勢：
1. ✅ **自適應區域選擇**：根據期望值動態分配搜尋資源
2. ✅ **參數自學習**：每個區域維護最佳參數
3. ✅ **防止冷落機制**：確保所有區域都被探索
4. ✅ **動態策略轉換**：前期探索 → 後期開發
5. ✅ **空間網引導**：動態調整搜尋方向
6. ✅ **群體大小調整**：根據進度自動調整探索/開發比例

這些機制共同作用，使 SNO 能夠在全域搜尋和局部搜尋之間取得良好平衡，有效解決各種連續型最佳化問題。
