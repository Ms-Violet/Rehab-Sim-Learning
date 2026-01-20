import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ================= 配置参数 =================
DATA_DIR = './data1'    
RANDOM_SEED = 42

# XGBoost 参数 (基准配置)
XGB_PARAMS = {
    'n_estimators': 1000,    # 树的数量
    'learning_rate': 0.05,   # 学习率
    'max_depth': 6,          # 树深
    'subsample': 0.8,        # 样本采样率
    'colsample_bytree': 0.8, # 特征采样率
    'random_state': RANDOM_SEED
}

# ================= 1. 特征提取工具 =================
def extract_features(time_series_data):
    """
    将 (Length, Features) 的时间序列转换为一维统计特征向量
    """
    # 1. 均值
    mean_val = np.mean(time_series_data, axis=0)
    # 2. 标准差
    std_val = np.std(time_series_data, axis=0)
    # 3. 最大值
    max_val = np.max(time_series_data, axis=0)
    # 4. 均方根 (RMS)
    rms_val = np.sqrt(np.mean(time_series_data**2, axis=0))
    
    # 拼接: 维度 = 肌肉通道数 * 4
    return np.concatenate([mean_val, std_val, max_val, rms_val])

# ================= 2. 数据加载函数 (复刻逻辑) =================
def load_data_simple():
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files:
        print(f"错误: 文件夹 {DATA_DIR} 中没有找到 CSV 文件")
        return None, None, None, None

    # 按文件拆分
    train_files, test_files = train_test_split(all_files, test_size=0.1, random_state=RANDOM_SEED)
    
    print(f"训练集文件数: {len(train_files)}")
    print(f"测试集文件数: {len(test_files)}")

    def process_files(file_list):
        data_features = []
        data_labels = []
        valid_files = [] 

        for fp in file_list:
            # --- 文件名解析 ---
            basename = os.path.basename(fp)
            parts = basename.split('_')
            try:
                k_val = float(parts[0]) / 1000.0
                c_val = float(parts[1]) / 1000.0
                label = np.array([k_val, c_val], dtype=np.float32)
            except:
                continue

            # --- CSV 读取 ---
            df = pd.read_csv(fp)
            # 跳过第1列(Time)，取后续肌肉数据
            features_raw = df.iloc[:, 1:].values.astype(np.float32) 
            
            # --- 特征提取 (全分辨率) ---
            # 直接使用全量数据计算统计量，不进行下采样
            feat_vector = extract_features(features_raw)
            
            data_features.append(feat_vector)
            data_labels.append(label)
            valid_files.append(basename)
            
        return np.array(data_features), np.array(data_labels), valid_files

    X_train, y_train, _ = process_files(train_files)
    X_test, y_test, test_filenames = process_files(test_files)
    
    return X_train, X_test, y_train, y_test, test_filenames

# ================= 3. 主程序 =================
def main():
    print("正在加载数据 (XGBoost Mode)...")
    X_train, X_test, y_train, y_test, test_filenames = load_data_simple()
    
    if X_train is None or len(X_train) == 0:
        print("数据加载失败。")
        return

    print(f"\n特征维度: {X_train.shape[1]}")

    # --- 归一化 ---
    # 虽然 XGBoost 对尺度不敏感，但为了训练稳定性，建议做标准化
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    # --- 模型定义与训练 ---
    print("\n开始训练 XGBoost...")
    
    # 初始化单模型
    xgb = XGBRegressor(**XGB_PARAMS)
    
    # 使用 MultiOutputRegressor 包装，使其支持多目标 (K, C)
    # 这本质上是训练了两个独立的 XGBoost 模型
    model = MultiOutputRegressor(xgb)
    
    model.fit(X_train_scaled, y_train)
    print(f"训练完成。")

    # --- 预测 ---
    print("\n正在预测...")
    preds = model.predict(X_test_scaled)
    
    # 物理修正: K, C >= 0
    preds = np.maximum(0, preds)

    # --- 结果打印 ---
    print("\n" + "="*70)
    print(f"{'Filename':<20} | {'True K':<8} {'Pred K':<8} | {'True C':<8} {'Pred C':<8}")
    print("-" * 70)
    
    total_ae_k = 0
    total_ae_c = 0
    
    for i in range(len(preds)):
        fname = test_filenames[i]
        t_k, t_c = y_test[i]
        p_k, p_c = preds[i]
        
        print(f"{fname:<20} | {t_k:<8.4f} {p_k:<8.4f} | {t_c:<8.4f} {p_c:<8.4f}")
        total_ae_k += abs(t_k - p_k)
        total_ae_c += abs(t_c - p_c)

    mae_k = total_ae_k / len(preds)
    mae_c = total_ae_c / len(preds)

    # --- 评估指标 ---
    def get_metrics(true, pred):
        return np.sqrt(mean_squared_error(true, pred)), mean_absolute_error(true, pred), r2_score(true, pred)

    rmse_k, _, r2_k = get_metrics(y_test[:, 0], preds[:, 0])
    rmse_c, _, r2_c = get_metrics(y_test[:, 1], preds[:, 1])

    print("\n" + "="*60)
    print("           XGBoost 综合性能评估报告")
    print("="*60)
    print(f"{'Metric':<10} | {'Parameter K':<15} | {'Parameter C':<15}")
    print("-" * 60)
    print(f"{'RMSE':<10} | {rmse_k:<15.6f} | {rmse_c:<15.6f}")
    print(f"{'MAE':<10} | {mae_k:<15.6f} | {mae_c:<15.6f}")
    print(f"{'R^2':<10} | {r2_k:<15.4f} | {r2_c:<15.4f}")
    print("="*60 + "\n")

    # --- 可视化 ---
    plt.figure(figsize=(14, 5))
    
    # K Value
    plt.subplot(1, 2, 1)
    plt.scatter(y_test[:, 0], preds[:, 0], c='b', alpha=0.6, label='Data')
    min_k, max_k = min(y_test[:, 0].min(), preds[:, 0].min()), max(y_test[:, 0].max(), preds[:, 0].max())
    plt.plot([min_k, max_k], [min_k, max_k], 'r--', label='Ideal')
    plt.title(f"K Value (XGBoost)\nMAE: {mae_k:.4f}")
    plt.xlabel("True K"); plt.ylabel("Predicted K"); plt.legend(); plt.grid(True, alpha=0.3)
    
    # C Value
    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:, 1], preds[:, 1], c='g', alpha=0.6, label='Data')
    min_c, max_c = min(y_test[:, 1].min(), preds[:, 1].min()), max(y_test[:, 1].max(), preds[:, 1].max())
    plt.plot([min_c, max_c], [min_c, max_c], 'r--', label='Ideal')
    plt.title(f"C Value (XGBoost)\nMAE: {mae_c:.4f}")
    plt.xlabel("True C"); plt.ylabel("Predicted C"); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # 保存
    joblib.dump(model, 'model/xgboost_simple_model.pkl')
    joblib.dump(scaler_x, 'model/xgboost_simple_scaler.pkl')
    print("Model saved.")

if __name__ == '__main__':
    main()