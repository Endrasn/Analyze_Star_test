import sys
import os
import logging
import argparse
import warnings

# === 1. 导入标准 CPU 库 ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")
# 设置绘图风格
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.unicode_minus'] = False


def shorten(name, max_len=20):
    """缩短过长的名字"""
    name = str(name)
    return name if len(name) <= max_len else name[:10] + "..." + name[-5:]


def cpu_random_oversampling(X, y):
    """CPU 版过采样，稳定可靠"""
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
    if not os.path.exists(csv_path):
        logging.error(f"❌ 文件未找到: {csv_path}")
        return

    # --- 1. 读取数据 ---
    logging.info("📖 [CPU] 正在读取数据...")
    df = pd.read_csv(csv_path)

    label_col = 'gz2_class'
    if label_col not in df.columns:
        logging.error(f"❌ 未找到标签列 '{label_col}'")
        return

    # 过滤稀有类别
    class_counts = df[label_col].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    df = df[df[label_col].isin(valid_classes)]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # --- 2. 预处理 ---
    logging.info("🔍 [CPU] 清洗数据...")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])

    # 异常值剔除
    Q1 = np.percentile(X_scaled, 25, axis=0)
    Q3 = np.percentile(X_scaled, 75, axis=0)
    IQR = Q3 - Q1
    outlier_mask = ~((X_scaled < (Q1 - 1.5 * IQR)) | (X_scaled > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df.iloc[outlier_mask].copy()

    # 二次过滤
    counts_post = df[label_col].value_counts()
    safe_classes = counts_post[counts_post >= 2].index
    df = df[df[label_col].isin(safe_classes)].copy()
    logging.info(f"✅ 有效样本数：{len(df)}")

    X = df[numeric_cols]
    y = df[label_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # --- 3. 特征筛选 ---
    logging.info("🔍 [CPU] 特征筛选...")
    rf_selector = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_selector.fit(X, y_encoded)

    importances = rf_selector.feature_importances_
    threshold = np.percentile(importances, 50)
    selected_mask = importances >= threshold
    selected_features = [numeric_cols[i] for i, s in enumerate(selected_mask) if s]
    X_selected = X[selected_features]

    # 获取 Top 15 特征的索引
    top_indices = np.argsort(importances)[-15:][::-1]
    top_15_names = [numeric_cols[i] for i in top_indices]
    logging.info(f"📌 筛选出 {len(selected_features)} 个特征，将只展示 Top 15")

    # --- 4. 分割与训练 ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    logging.info("⚖️ [CPU] 样本平衡处理...")
    X_train_res, y_train_res = cpu_random_oversampling(X_train, y_train)

    logging.info("⚙️ [CPU] 训练模型...")
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)

    logging.info("📊 评估结果...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"\n🎯 测试集准确率：{acc:.4f}")

    print("\n" + "=" * 40)
    print("📊 分类报告")
    print("=" * 40)
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # ============= 图表绘制 (独立绘制 5 张图) =============
    logging.info("🎨 正在生成 5 张独立图表...")

    final_imps = model.feature_importances_
    sorted_idx = np.argsort(final_imps)[-15:]

    # 1. 特征重要性
    plt.figure(figsize=(10, 8))
    bars = plt.barh([shorten(selected_features[i]) for i in sorted_idx], final_imps[sorted_idx], color='#4c72b0')
    plt.title("Top 15 Feature Importances", fontsize=16)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

    # 2. 相关性热力图
    plt.figure(figsize=(12, 10))
    top_features_df = df[top_15_names]
    sns.heatmap(top_features_df.corr(), cmap="coolwarm", annot=True, fmt=".2f",
                annot_kws={"size": 8}, linewidths=.5)
    plt.title("Correlation Heatmap (Top 15 Features Only)", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 3. 混淆矩阵 (关键修复点)
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    # 💡 修复：annot=cm 传入整数矩阵，fmt='d' 就能正常工作了
    sns.heatmap(np.log1p(cm), cmap="YlGnBu",
                annot=cm,  # <--- 这里改成了 cm (整数)，而不是 True (默认用log后的数据)
                fmt='d',  # <--- 整数格式
                xticklabels=[shorten(c) for c in le.classes_],
                yticklabels=[shorten(c) for c in le.classes_])
    plt.title("Confusion Matrix (Color: Log Scale, Text: Count)", fontsize=16)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 4. 预测置信度
    plt.figure(figsize=(10, 6))
    probas = model.predict_proba(X_test)
    max_probas = np.max(probas, axis=1)
    sns.histplot(max_probas, bins=30, kde=True, color='purple')
    plt.title("Prediction Confidence Distribution", fontsize=16)
    plt.xlabel("Max Probability (Confidence)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # 5. 类别分布
    plt.figure(figsize=(10, 10))
    class_dist = df[label_col].value_counts()
    colors = sns.color_palette('pastel')[0:len(class_dist)]
    plt.pie(class_dist.values, labels=[shorten(c) for c in class_dist.index],
            autopct='%1.1f%%', startangle=140, colors=colors,
            pctdistance=0.85, explode=[0.02] * len(class_dist))
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title("Class Distribution", fontsize=16)
    plt.tight_layout()
    plt.show()

    logging.info("✅ 所有图表渲染完毕！")


def main():
    path = ""
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default=path)
    args = parser.parse_args()
    cosmic_body_classification(args.csv_path)


if __name__ == "__main__":
    main()