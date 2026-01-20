import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import time

# ================= 配置参数 =================
DATA_ROOT = './data2'
SUB_DIRS = {'F': 'Flexion_p', 'D': 'Deviation_p', 'P': 'Pronation_p'}

SELECTED_MUSCLES = [
    'Top_PT', 'Top_ECRL', 'Top_UI_UB5', 'Top_PQ', 
    'Top_FCU', 'Top_FDS4', 'Top_PL', 'Top_LU_RB5', 
    'Top_ECRB', 'Top_FCR', 'Top_ECU'
]

BATCH_SIZE = 16            
LEARNING_RATE = 0.001
NUM_EPOCHS = 200          
SEQ_LEN = 800            
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ================= 辅助类：标签归一化 =================
class TargetScaler:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, targets):
        self.mean = np.mean(targets, axis=0)
        self.scale = np.std(targets, axis=0) + 1e-6

    def transform(self, targets):
        return (targets - self.mean) / self.scale

    def inverse_transform(self, targets_norm):
        return targets_norm * self.scale + self.mean

# ================= 数据加载与处理 =================
def get_label_from_k_c(k_str, c_str):
    k = float(k_str) / 1000.0
    c = float(c_str) / 1000.0
    return np.array([k, c], dtype=np.float32)

def process_single_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    feature_list = []
    n_rows = len(df)
    
    for muscle in SELECTED_MUSCLES:
        if muscle in df.columns:
            col_data = df[muscle].values.astype(np.float32)
        else:
            col_data = np.zeros(n_rows, dtype=np.float32)
        feature_list.append(col_data)
    
    features = np.stack(feature_list, axis=1)
    
    if len(features) > SEQ_LEN:
        features = features[:SEQ_LEN, :]
    else:
        padding = np.zeros((SEQ_LEN - len(features), features.shape[1]), dtype=np.float32)
        features = np.vstack((features, padding))
        
    return features

def load_multimodal_data(root_dir):
    flex_dir = os.path.join(root_dir, SUB_DIRS['F'])
    dev_dir  = os.path.join(root_dir, SUB_DIRS['D'])
    pro_dir  = os.path.join(root_dir, SUB_DIRS['P'])
    
    if not os.path.exists(flex_dir):
        raise ValueError(f"找不到文件夹: {flex_dir}")

    flex_files = glob.glob(os.path.join(flex_dir, "*_F_p.csv"))
    
    data_list = []
    label_list = []
    
    print(f"Scanning {len(flex_files)} base files in Flexion folder...")
    
    for f_path in flex_files:
        try:
            filename = os.path.basename(f_path)
            parts = filename.split('_')
            k_str, c_str = parts[0], parts[1]
            
            d_filename = f"{k_str}_{c_str}_D_p.csv"
            p_filename = f"{k_str}_{c_str}_P_p.csv"
            
            d_path = os.path.join(dev_dir, d_filename)
            p_path = os.path.join(pro_dir, p_filename)
            
            if not os.path.exists(d_path) or not os.path.exists(p_path):
                continue
            
            feat_f = process_single_csv(f_path)
            feat_d = process_single_csv(d_path)
            feat_p = process_single_csv(p_path)
            
            combined_feat = np.concatenate([feat_f, feat_d, feat_p], axis=1)
            
            data_list.append(combined_feat)
            label_list.append(get_label_from_k_c(k_str, c_str))
            
        except Exception as e:
            print(f"Error processing set {filename}: {e}")
            continue
            
    print(f"Successfully loaded {len(data_list)} fused samples.")
    return np.array(data_list), np.array(label_list)

class MuscleForceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ================= BiLSTM 模型定义 =================
class KC_Predictor_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, num_layers=2):
        super(KC_Predictor_BiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True 
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), 
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)

# ================= 评估函数 =================
def evaluate_performance(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ================= 主程序 (含测速) =================
def train_and_test():
    try:
        X_all, y_all_raw = load_multimodal_data(DATA_ROOT)
    except Exception as e:
        print(f"Fatal Error: {e}")
        return

    if len(X_all) == 0:
        print("No valid data loaded.")
        return

    N, L, F = X_all.shape
    print(f"Final Input Shape: {X_all.shape} (Features={F})")
    
    # 1. Input 归一化
    X_reshaped = X_all.reshape(-1, F)
    input_scaler = StandardScaler()
    X_scaled = input_scaler.fit_transform(X_reshaped).reshape(N, L, F)
    
    # 2. Target 归一化
    target_scaler = TargetScaler()
    target_scaler.fit(y_all_raw)
    y_all_norm = target_scaler.transform(y_all_raw)
    
    # 3. 拆分
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all_norm, test_size=0.1, random_state=42)
    
    train_dataset = MuscleForceDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(MuscleForceDataset(X_test, y_test), batch_size=1, shuffle=False)

    # --- 初始化 BiLSTM ---
    model = KC_Predictor_BiLSTM(input_dim=F, hidden_dim=128, output_dim=2).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_history = []
    epoch_times = []
    
    print("\n" + "="*60)
    print(f"Start Training (BiLSTM Fusion) on {DEVICE}")
    print(f"Total Training Samples: {len(train_dataset)}")
    print("="*60)
    
    total_train_start = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        
        # [Speed] 计时开始
        epoch_start = time.time()
        
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # [Speed] 计时结束
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)
        
        # 计算当前速度 (seq/s)
        current_speed = len(train_dataset) / epoch_duration
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.6f} ")

    total_train_end = time.time()
    total_duration = total_train_end - total_train_start
    avg_epoch_time = np.mean(epoch_times)
    avg_speed = len(train_dataset) / avg_epoch_time

    # ================= 训练速度报告 =================
    print("\n" + "="*60)
    print("           Training Speed Benchmark Report")
    print("="*60)
    print(f"Model Architecture:      Bi-LSTM")
    print(f"Device:                  {DEVICE}")
    print(f"Batch Size:              {BATCH_SIZE}")
    print(f"Total Training Time:     {total_duration:.2f} seconds")
    print(f"Average Time per Epoch:  {avg_epoch_time:.4f} seconds")
    print(f"Average Training Speed:  {avg_speed:.2f} sequences/second")
    print("="*60 + "\n")

    # ================= 测试与评估 =================
    print("\nRunning Final Evaluation...")
    model.eval()
    
    preds_norm_list = []
    gts_norm_list = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            preds_norm_list.append(pred.cpu().numpy())
            gts_norm_list.append(y.cpu().numpy())
            
    preds_norm = np.concatenate(preds_norm_list)
    gts_norm = np.concatenate(gts_norm_list)
    
    # 反归一化
    preds = target_scaler.inverse_transform(preds_norm)
    gts = target_scaler.inverse_transform(gts_norm)
    preds = np.maximum(0, preds)

    print("\n" + "="*60)
    print(f"{'Sample':<8} | {'True K':<10} {'Pred K':<10} | {'True C':<10} {'Pred C':<10}")
    print("-" * 60)
    for i in range(len(preds)):
        t_k, t_c = gts[i]
        p_k, p_c = preds[i]
        print(f"{i:<8} | {t_k:<10.4f} {p_k:<10.4f} | {t_c:<10.4f} {p_c:<10.4f}")
    print("="*60)

    rmse_k, mae_k, r2_k = evaluate_performance(gts[:, 0], preds[:, 0])
    rmse_c, mae_c, r2_c = evaluate_performance(gts[:, 1], preds[:, 1])

    print("\n" + "="*60)
    print("           BiLSTM 综合性能评估报告")
    print("="*60)
    print(f"{'Metric':<10} | {'Parameter K':<15} | {'Parameter C':<15}")
    print("-" * 60)
    print(f"{'RMSE':<10} | {rmse_k:<15.6f} | {rmse_c:<15.6f}")
    print(f"{'MAE':<10} | {mae_k:<15.6f} | {mae_c:<15.6f}")
    print(f"{'R^2':<10} | {r2_k:<15.4f} | {r2_c:<15.4f}")
    print("="*60 + "\n")

    # ================= 可视化 =================
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, 'b-', label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(gts[:, 0], preds[:, 0], c='b', alpha=0.6, label='Data')
    min_k, max_k = min(gts[:, 0].min(), preds[:, 0].min()), max(gts[:, 0].max(), preds[:, 0].max())
    plt.plot([min_k, max_k], [min_k, max_k], 'r--', label='Ideal')
    plt.title(f"K Value (BiLSTM)\n$R^2$={r2_k:.3f}")
    plt.xlabel("True K")
    plt.ylabel("Predicted K")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(gts[:, 1], preds[:, 1], c='g', alpha=0.6, label='Data')
    min_c, max_c = min(gts[:, 1].min(), preds[:, 1].min()), max(gts[:, 1].max(), preds[:, 1].max())
    plt.plot([min_c, max_c], [min_c, max_c], 'r--', label='Ideal')
    plt.title(f"C Value (BiLSTM)\n$R^2$={r2_c:.3f}")
    plt.xlabel("True C")
    plt.ylabel("Predicted C")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    if not os.path.exists("model"):
        os.makedirs("model")

    torch.save(model.state_dict(), "model/bilstm_fusion_model.pth")
    joblib.dump(input_scaler, 'model/bilstm_input_scaler.pkl')
    joblib.dump(target_scaler, 'model/bilstm_target_scaler.pkl')
    print("Models saved.")

if __name__ == '__main__':
    train_and_test()