# 糖尿病資料集 - 資料探索分析（中文變數版）

以下是使用中文變數名稱的資料探索程式碼，讓不熟悉英文的同學更容易理解和練習。

## **步驟 1：環境設定與載入資料**

```python
# 引入必要的函式庫
import pandas as pd          # 資料處理函式庫
import matplotlib.pyplot as plt  # 繪圖函式庫
import seaborn as sns        # 統計圖表函式庫
import numpy as np           # 數值計算函式庫
import warnings              # 警告控制函式庫

# 設定忽略警告訊息
warnings.simplefilter(action='ignore', category=FutureWarning)

# 設定中文字體（避免圖表中文顯示問題）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 設定圖表品質
plt.rcParams['figure.dpi'] = 100      # 螢幕顯示解析度
plt.rcParams['savefig.dpi'] = 300     # 儲存圖片解析度

# --- 載入資料 ---
try:
    # 資料網址
    資料網址 = "https://github.com/acechan0626/pythondata/raw/main/pima-indians-diabetes.csv"
    
    # 讀取資料
    原始資料 = pd.read_csv(資料網址)
    
    print("✅ 資料載入成功！")
    print(f"資料大小：{原始資料.shape[0]} 筆資料，{原始資料.shape[1]} 個欄位")
    
    # 顯示資料前5筆
    print("\n前5筆資料：")
    print(原始資料.head())
    
    # 顯示資料基本資訊
    print("\n資料基本資訊：")
    print(原始資料.info())
    
except Exception as 錯誤訊息:
    print(f"❌ 載入失敗：{錯誤訊息}")
    print("請檢查網路連線或手動上傳檔案")
```

## **步驟 2：資料欄位重新命名（中文化）**

```python
# 將英文欄位名稱改為中文，方便理解
if '原始資料' in locals():
    # 建立欄位對照表
    欄位對照表 = {
        'Number_pregnant': '懷孕次數',
        'Glucose_concentration': '血糖濃度', 
        'Blood_pressure': '血壓',
        'Triceps': '三頭肌厚度',
        'Insulin': '胰島素',
        'BMI': '身體質量指數',
        'Pedigree': '糖尿病家族史',
        'Age': '年齡',
        'Class': '糖尿病診斷',
        'Group': '分組'
    }
    
    # 重新命名欄位
    糖尿病資料 = 原始資料.rename(columns=欄位對照表)
    
    print("=== 欄位重新命名完成 ===")
    print("新的欄位名稱：")
    for 舊名稱, 新名稱 in 欄位對照表.items():
        if 舊名稱 in 原始資料.columns:
            print(f"{舊名稱} → {新名稱}")
    
    print(f"\n重新命名後的資料：")
    print(糖尿病資料.head())
```

## **步驟 3：資料基本統計分析**

```python
# 基本統計分析
if '糖尿病資料' in locals():
    print("\n=== 資料基本統計 ===")
    
    # 顯示描述性統計
    統計摘要 = 糖尿病資料.describe()
    print("數值欄位統計摘要：")
    print(統計摘要.round(2))  # 四捨五入到小數點後2位
    
    # 分析目標變數
    if '糖尿病診斷' in 糖尿病資料.columns:
        print(f"\n=== 糖尿病診斷分布 ===")
        
        # 計算各類別數量
        診斷分布 = 糖尿病資料['糖尿病診斷'].value_counts()
        print("診斷結果統計：")
        print(f"健康（0）：{診斷分布[0]} 人")
        print(f"糖尿病（1）：{診斷分布[1]} 人")
        
        # 計算糖尿病比例
        糖尿病比例 = 糖尿病資料['糖尿病診斷'].mean()
        print(f"糖尿病患者比例：{糖尿病比例:.1%}")
```

## **步驟 4：繪製直方圖**

```python
# === 1. 直方圖分析 ===
if '糖尿病資料' in locals():
    print("\n--- 開始繪製直方圖 ---")
    
    # 選擇數值型欄位
    數值欄位 = 糖尿病資料.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # 排除目標變數，只分析特徵
    特徵欄位 = [欄位 for 欄位 in 數值欄位 if 欄位 != '糖尿病診斷']
    
    print(f"要分析的特徵：{特徵欄位}")
    
    # 計算子圖佈局
    每行圖數 = 3  # 每行放3個圖
    總行數 = (len(特徵欄位) + 每行圖數 - 1) // 每行圖數  # 計算需要幾行
    
    # 建立子圖
    圖片, 子圖陣列 = plt.subplots(總行數, 每行圖數, figsize=(15, 總行數 * 4))
    
    # 處理子圖陣列格式
    if 總行數 == 1:
        子圖陣列 = [子圖陣列] if 每行圖數 == 1 else 子圖陣列
    else:
        子圖陣列 = 子圖陣列.flatten()  # 攤平成一維陣列
    
    # 為每個特徵繪製直方圖
    for 索引, 欄位名稱 in enumerate(特徵欄位):
        if 索引 < len(子圖陣列):
            # 繪製直方圖 + 密度曲線
            sns.histplot(糖尿病資料[欄位名稱], kde=True, ax=子圖陣列[索引], alpha=0.7)
            
            # 設定圖表標題和標籤
            子圖陣列[索引].set_title(f'{欄位名稱} 分布圖', fontsize=12, fontweight='bold')
            子圖陣列[索引].set_xlabel(欄位名稱)
            子圖陣列[索引].set_ylabel('頻率')
            子圖陣列[索引].grid(True, alpha=0.3)  # 添加網格
    
    # 移除多餘的子圖
    for 索引 in range(len(特徵欄位), len(子圖陣列)):
        圖片.delaxes(子圖陣列[索引])
    
    # 調整佈局並儲存
    plt.tight_layout()
    plt.savefig('直方圖.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    print("✅ 直方圖繪製完成，已儲存為 '直方圖.png'")
```

## **步驟 5：繪製箱型圖**

```python
# === 2. 箱型圖分析 ===
if '糖尿病資料' in locals():
    print("\n--- 開始繪製箱型圖 ---")
    
    # 建立子圖（使用相同佈局）
    圖片, 子圖陣列 = plt.subplots(總行數, 每行圖數, figsize=(15, 總行數 * 4))
    
    # 處理子圖陣列格式
    if 總行數 == 1:
        子圖陣列 = [子圖陣列] if 每行圖數 == 1 else 子圖陣列
    else:
        子圖陣列 = 子圖陣列.flatten()
    
    # 為每個特徵繪製箱型圖
    for 索引, 欄位名稱 in enumerate(特徵欄位):
        if 索引 < len(子圖陣列):
            # 繪製箱型圖
            sns.boxplot(x=糖尿病資料[欄位名稱], ax=子圖陣列[索引])
            
            # 設定圖表標題
            子圖陣列[索引].set_title(f'{欄位名稱} 箱型圖', fontsize=12, fontweight='bold')
            子圖陣列[索引].set_xlabel(欄位名稱)
            子圖陣列[索引].grid(True, alpha=0.3)
    
    # 移除多餘的子圖
    for 索引 in range(len(特徵欄位), len(子圖陣列)):
        圖片.delaxes(子圖陣列[索引])
    
    # 調整佈局並儲存
    plt.tight_layout()
    plt.savefig('箱型圖.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    print("✅ 箱型圖繪製完成，已儲存為 '箱型圖.png'")
```

## **步驟 6：繪製相關矩陣熱力圖**

```python
# === 3. 相關性分析 ===
if '糖尿病資料' in locals():
    print("\n--- 開始分析變數相關性 ---")
    
    # 計算相關矩陣
    相關矩陣 = 糖尿病資料[數值欄位].corr()
    
    # 建立圖片
    plt.figure(figsize=(12, 10))
    
    # 建立上三角遮罩（避免重複顯示）
    上三角遮罩 = np.triu(np.ones_like(相關矩陣, dtype=bool))
    
    # 繪製熱力圖
    sns.heatmap(相關矩陣,
                mask=上三角遮罩,        # 使用遮罩
                annot=True,             # 顯示數值
                cmap='coolwarm',        # 顏色主題
                fmt='.2f',              # 數值格式
                center=0,               # 顏色中心點
                square=True,            # 正方形格子
                cbar_kws={"shrink": .8}) # 顏色條大小
    
    # 設定標題
    plt.title('變數相關性熱力圖', fontsize=16, fontweight='bold', pad=20)
    
    # 調整佈局並儲存
    plt.tight_layout()
    plt.savefig('相關矩陣.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    print("✅ 相關矩陣繪製完成，已儲存為 '相關矩陣.png'")
    
    # 找出高相關性的變數對
    print("\n=== 高相關性變數分析（相關係數 > 0.5）===")
    高相關變數對 = []
    
    # 檢查所有變數對的相關性
    for i in range(len(相關矩陣.columns)):
        for j in range(i+1, len(相關矩陣.columns)):
            相關係數 = 相關矩陣.iloc[i, j]
            if abs(相關係數) > 0.5:
                變數1 = 相關矩陣.columns[i]
                變數2 = 相關矩陣.columns[j]
                高相關變數對.append((變數1, 變數2, 相關係數))
    
    # 顯示結果
    if 高相關變數對:
        for 變數1, 變數2, 係數 in 高相關變數對:
            相關程度 = "強正相關" if 係數 > 0.7 else "中等正相關" if 係數 > 0.5 else "強負相關" if 係數 < -0.7 else "中等負相關"
            print(f"{變數1} ↔ {變數2}: {係數:.3f} ({相關程度})")
    else:
        print("沒有發現高相關性的變數對")
```

## **步驟 7：繪製散佈圖矩陣 (Pairplot)**

```python
# === 4. 散佈圖矩陣分析 ===
if '糖尿病資料' in locals():
    print("\n--- 開始繪製散佈圖矩陣 ---")
    
    # 選擇重要變數進行分析（避免圖表過於複雜）
    重要變數 = ['血糖濃度', '血壓', '身體質量指數', '年齡', '糖尿病診斷']
    
    # 確認選擇的變數都存在
    存在的變數 = [變數 for 變數 in 重要變數 if 變數 in 糖尿病資料.columns]
    
    print(f"選擇分析的變數：{存在的變數}")
    
    if len(存在的變數) > 1:
        # 準備繪圖資料
        繪圖資料 = 糖尿病資料[存在的變數].copy()
        
        # 如果有診斷欄位，轉換為類別型態
        if '糖尿病診斷' in 繪圖資料.columns:
            繪圖資料['糖尿病診斷'] = 繪圖資料['糖尿病診斷'].astype('category')
            
            # 繪製散佈圖矩陣，按診斷結果分色
            散佈圖 = sns.pairplot(繪圖資料,
                                hue='糖尿病診斷',      # 按診斷結果分色
                                diag_kind='kde',       # 對角線顯示密度圖
                                plot_kws={'alpha': 0.6}, # 點的透明度
                                diag_kws={'alpha': 0.7}) # 密度圖透明度
            
            # 設定圖例標籤
            散佈圖._legend.set_title('診斷結果')
            新標籤 = ['健康', '糖尿病']
            for 文字, 標籤 in zip(散佈圖._legend.texts, 新標籤):
                文字.set_text(標籤)
        else:
            # 沒有診斷欄位時的散佈圖
            散佈圖 = sns.pairplot(繪圖資料, diag_kind='kde')
        
        # 設定主標題
        散佈圖.fig.suptitle('重要變數散佈圖矩陣', y=1.02, fontsize=16, fontweight='bold')
        
        # 儲存圖片
        plt.savefig('散佈圖矩陣.png', bbox_inches='tight', dpi=300)
        plt.show()
        
        print("✅ 散佈圖矩陣繪製完成，已儲存為 '散佈圖矩陣.png'")
    else:
        print("❌ 變數數量不足，無法繪製散佈圖矩陣")

print("\n🎉 所有圖表分析完成！")
```

## **步驟 8：分組統計分析**

```python
# === 5. 按診斷結果分組分析 ===
if '糖尿病資料' in locals() and '糖尿病診斷' in 糖尿病資料.columns:
    print("\n--- 按診斷結果分組統計分析 ---")
    
    # 計算分組統計
    分組統計 = 糖尿病資料.groupby('糖尿病診斷')[特徵欄位].agg(['mean', 'std', 'median'])
    
    print("=== 各組平均值比較 ===")
    平均值比較 = 分組統計.xs('mean', level=1, axis=1).round(2)
    平均值比較.index = ['健康組', '糖尿病組']
    print(平均值比較)
    
    print("\n=== 各組標準差比較 ===")
    標準差比較 = 分組統計.xs('std', level=1, axis=1).round(2)
    標準差比較.index = ['健康組', '糖尿病組']
    print(標準差比較)
    
    # 進行統計檢驗
    from scipy import stats
    
    print("\n=== 組間差異顯著性檢驗 ===")
    print("檢驗兩組間是否有顯著差異（t檢驗）：")
    
    for 欄位名稱 in 特徵欄位:
        # 分別取得兩組資料
        健康組資料 = 糖尿病資料[糖尿病資料['糖尿病診斷'] == 0][欄位名稱]
        糖尿病組資料 = 糖尿病資料[糖尿病資料['糖尿病診斷'] == 1][欄位名稱]
        
        # 進行t檢驗
        t統計量, p值 = stats.ttest_ind(健康組資料, 糖尿病組資料)
        
        # 判斷顯著性
        if p值 < 0.001:
            顯著性 = "極顯著 (***)"
        elif p值 < 0.01:
            顯著性 = "很顯著 (**)"
        elif p值 < 0.05:
            顯著性 = "顯著 (*)"
        else:
            顯著性 = "不顯著"
        
        print(f"{欄位名稱}: t={t統計量:.3f}, p={p值:.4f} - {顯著性}")
    
    print("\n說明：")
    print("* p<0.05 (顯著差異)")
    print("** p<0.01 (很顯著差異)")  
    print("*** p<0.001 (極顯著差異)")
```

## **完整簡化版本（適合課堂快速演示）**

```python
# === 完整簡化版本 ===

# 1. 引入函式庫
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.simplefilter('ignore')

# 2. 載入資料
資料網址 = "https://github.com/acechan0626/pythondata/raw/main/pima-indians-diabetes.csv"
原始資料 = pd.read_csv(資料網址)

# 3. 中文化欄位名稱
欄位對照 = {
    'Number_pregnant': '懷孕次數',
    'Glucose_concentration': '血糖濃度', 
    'Blood_pressure': '血壓',
    'Triceps': '三頭肌厚度',
    'Insulin': '胰島素',
    'BMI': '身體質量指數',
    'Pedigree': '糖尿病家族史',
    'Age': '年齡',
    'Class': '糖尿病診斷'
}
資料 = 原始資料.rename(columns=欄位對照)
print("資料載入完成！")

# 4. 選擇分析欄位
特徵 = ['懷孕次數', '血糖濃度', '血壓', '三頭肌厚度', '胰島素', '身體質量指數', '糖尿病家族史', '年齡']

# 5. 直方圖
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, 欄位 in enumerate(特徵):
    sns.histplot(資料[欄位], kde=True, ax=axes[i])
    axes[i].set_title(f'{欄位}分布')
fig.delaxes(axes[8])  # 刪除多餘子圖
plt.tight_layout()
plt.savefig('直方圖.png')
plt.show()

# 6. 箱型圖
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, 欄位 in enumerate(特徵):
    sns.boxplot(x=資料[欄位], ax=axes[i])
    axes[i].set_title(f'{欄位}箱型圖')
fig.delaxes(axes[8])
plt.tight_layout()
plt.savefig('箱型圖.png')
plt.show()

# 7. 相關矩陣
plt.figure(figsize=(10, 8))
相關矩陣 = 資料[特徵 + ['糖尿病診斷']].corr()
sns.heatmap(相關矩陣, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('相關性分析')
plt.savefig('相關矩陣.png')
plt.show()

# 8. 散佈圖矩陣
重要欄位 = ['血糖濃度', '血壓', '身體質量指數', '年齡', '糖尿病診斷']
sns.pairplot(資料[重要欄位], hue='糖尿病診斷', diag_kind='kde')
plt.savefig('散佈圖矩陣.png')
plt.show()

print("所有分析完成！")
```

## **使用說明**

### **優點**
1. **中文變數名**：更容易理解變數含義
2. **中文註解**：每個步驟都有清楚說明
3. **教學友善**：適合中文環境的教學

### **注意事項**
1. **字體設定**：程式碼包含中文字體設定
2. **檔案命名**：圖片以中文命名，方便識別
3. **變數命名**：使用有意義的中文變數名

這個版本特別適合中文教學環境，讓學生更容易理解程式邏輯和資料分析概念！
