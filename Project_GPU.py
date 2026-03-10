import sys
import os
import logging
import argparse
import warnings
import time

# === 1. 导入库 ===
try:
    # 尝试导入 GPU 库 (只用于核心训练)
    import cuml
    from cuml.ensemble import RandomForestClassifier as GPURF
    import cudf       
    import cupy as cp 
    print("✅ RAPIDS (cuML/cuDF) 导入成功，准备使用 GPU 训练！")
except ImportError as e:
    print(f"❌ 无法导入 RAPIDS 库，请检查环境。\n详细信息: {e}")
    sys.exit(1)

# 导入 CPU 库 (用于稳定的数据清洗)
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder as CPU_LE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.unicode_minus'] = False

def shorten(name, max_len=20):
    name = str(name)
    return name if len(name) <= max_len else name[:10] + "..." + name[-5:]

def cpu_random_oversampling(X, y):
    """CPU 版过采样，稳定可靠，防止 GPU 在 concat 时崩溃"""
    df = X.copy()
    df['label_temp_col'] = y
    max_size = df['label_temp_col'].value_counts().max()
    lst = [df]
    for class_index, group in df.groupby('label_temp_col'):
        if len(group) < max_size:
            needed = max_size - len(group)
            resampled = group.sample(n=needed, replace=True, random_state=42)
            lst.append(resampled)
    df_resampled = pd.concat(lst)
    y_resampled = df_resampled['label_temp_col']
    X_resampled = df_resampled.drop(columns=['label_temp_col'])
    return X_resampled, y_resampled

def cosmic_body_classification(csv_path):
    # --- 0. 检查 GPU ---
    try:
        dev_info = cp.cuda.runtime.getDeviceProperties(0)
        logging.info(f"🚀 [硬件检查] 正在使用 GPU: {dev_info['name'].decode()}")
    except:
        logging.error("❌ 未检测到 GPU！")
        return

    if not os.path.exists(csv_path):
        logging.error(f"❌ 文件未找到: {csv_path}")
        return

    # --- 1. 数据清洗 (CPU 模式 - 安全区) ---
    logging.info("📖 [CPU] 读取数据与预处理 (混合架构以保证稳定)...")
    df = pd.read_csv(csv_path)

    label_col = 'gz2_class'
    if label_col not in df.columns:
        logging.error(f"❌ 未找到标签列 '{label_col}'")
        return

    # 基础过滤
    class_counts = df[label_col].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    df = df[df[label_col].isin(valid_classes)]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 填充与标准化
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])
    
    # 异常值剔除
    Q1 = np.percentile(X_scaled, 25, axis=0)
    Q3 = np.percentile(X_scaled, 75, axis=0)
    IQR = Q3 - Q1
    outlier_mask = ~((X_scaled < (Q1 - 1.5 * IQR)) | (X_scaled > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df.iloc[outlier_mask].copy()
    
    # 二次过滤 (防止样本不足导致分割报错)
    counts_post = df[label_col].value_counts()
    safe_classes = counts_post[counts_post >= 2].index 
    df = df[df[label_col].isin(safe_classes)].copy()
    logging.info(f"✅ [CPU] 数据清洗完毕，有效样本数：{len(df)}")

    # 准备数据
    X = df[numeric_cols]
    y = df[label_col]
    le = CPU_LE()
    y_encoded = le.fit_transform(y)

    # --- 2. 特征筛选 (CPU 版 RF) ---
    # 这里用 sklearn 的 RF 快速筛一遍，避免 GPU 反复显存分配
    from sklearn.ensemble import RandomForestClassifier as CPURF
    logging.info("🔍 [CPU] 初步特征筛选...")
    rf_selector = CPURF(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_selector.fit(X, y_encoded)
    
    importances = rf_selector.feature_importances_
    threshold = np.percentile(importances, 50) 
    selected_mask = importances >= threshold
    selected_features = [numeric_cols[i] for i, s in enumerate(selected_mask) if s]
    X_selected = X[selected_features]
    
    # 记录 Top 15 用于画图
    top_indices = np.argsort(importances)[-15:][::-1]
    top_15_names = [numeric_cols[i] for i in top_indices]
    logging.info(f"📌 筛选保留 {len(selected_features)} 个特征")

    # --- 3. 分割与过采样 (CPU) ---
    logging.info("✂️ [CPU] 数据分割与过采样...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    X_train_res, y_train_res = cpu_random_oversampling(X_train, y_train)

    # --- 4. [关键] 数据上云 (CPU -> GPU) ---
    logging.info("🚀 [GPU] 数据转换并上传显存 (Data Upload)...")
    
    # ⚠️ 强制类型转换：cuML 极度依赖 float32 和 int32
    X_train_np = X_train_res.to_numpy().astype('float32')
    y_train_np = y_train_res.to_numpy().astype('int32')
    X_test_np = X_test.to_numpy().astype('float32')
    
    # 创建 GPU DataFrame
    X_train_gpu = cudf.DataFrame(X_train_np)
    y_train_gpu = cudf.Series(y_train_np)
    X_test_gpu = cudf.DataFrame(X_test_np)

    # --- 5. 模型训练 (GPU 核心加速) ---
    logging.info("⚙️ [GPU] 启动 cuML 随机森林训练...")
    start_time = time.time()
    
    # GPU 模型参数
    model = GPURF(n_estimators=200, max_depth=15, n_bins=128, n_streams=1, random_state=42)
    model.fit(X_train_gpu, y_train_gpu)
    
    end_time = time.time()
    logging.info(f"⚡ [GPU] 训练完成！耗时: {end_time - start_time:.4f} 秒")

    # --- 6. 评估 ---
    logging.info("📊 正在评估...")
    y_pred_gpu = model.predict(X_test_gpu)
    
    # 转回 CPU 进行指标计算
    y_pred = y_pred_gpu.to_numpy().astype('int32')
    
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"\n🎯 测试集准确率：{acc:.4f}")

    print("\n" + "="*40)
    print("📊 分类报告")
    print("="*40)
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # ============= 图表绘制 (独立 5 张) =============
    logging.info("🎨 正在生成 5 张独立图表...")
    
    # 获取特征重要性 (从 GPU 模型取回)
    try:
        final_imps = model.feature_importances_.to_numpy()
    except:
        final_imps = model.feature_importances_ # 兼容性写法
        
    sorted_idx = np.argsort(final_imps)[-15:] # Top 15
    
    # 1. 特征重要性
    plt.figure(figsize=(10, 8))
    bars = plt.barh([shorten(selected_features[i]) for i in sorted_idx], final_imps[sorted_idx], color='#4c72b0')
    plt.title(f"Top 15 Feature Importances (GPU Model)", fontsize=16)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show() 

    # 2. 相关性热力图
    plt.figure(figsize=(12, 10))
    top_features_df = df[top_15_names] 
    sns.heatmap(top_features_df.corr(), cmap="coolwarm", annot=True, fmt=".2f", 
                annot_kws={"size": 8}, linewidths=.5)
    plt.title("Correlation Heatmap (Top 15 Features)", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show() 

    # 3. 混淆矩阵 (已修复格式问题)
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(np.log1p(cm), cmap="YlGnBu", 
                annot=cm,  # 显示整数
                fmt='d',   # 整数格式
                xticklabels=[shorten(c) for c in le.classes_],
                yticklabels=[shorten(c) for c in le.classes_])
    plt.title("Confusion Matrix (GPU Prediction)", fontsize=16)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show() 

    # 4. 预测置信度
    plt.figure(figsize=(10, 6))
    # predict_proba 返回 GPU 矩阵，需转回 CPU
    probas = model.predict_proba(X_test_gpu)
    max_probas = cp.asnumpy(cp.max(probas, axis=1)) # 显式转 Numpy
    sns.histplot(max_probas, bins=30, kde=True, color='purple')
    plt.title("Prediction Confidence Distribution", fontsize=16)
    plt.xlabel("Max Probability")
    plt.tight_layout()
    plt.show() 

    # 5. 类别分布
    plt.figure(figsize=(10, 10))
    class_dist = df[label_col].value_counts()
    colors = sns.color_palette('pastel')[0:len(class_dist)]
    plt.pie(class_dist.values, labels=[shorten(c) for c in class_dist.index], 
            autopct='%1.1f%%', startangle=140, colors=colors, 
            pctdistance=0.85, explode=[0.02]*len(class_dist))
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title("Class Distribution", fontsize=16)
    plt.tight_layout()
    plt.show() 

    logging.info("✅ 任务全部完成！GPU 表现完美。")

def main():
    path = "/home/umdrasyl/下载/gz2_hart16.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default=path)
    args = parser.parse_args()
    cosmic_body_classification(args.csv_path)

if __name__ == "__main__":
    main()