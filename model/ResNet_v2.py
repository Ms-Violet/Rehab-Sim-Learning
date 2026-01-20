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

# ================= 配置参数 =================
DATA_ROOT = './data2'
SUB_DIRS = {'F': 'Flexion_p', 'D': 'Deviation_p', 'P': 'Pronation_p'}

SELECTED_MUSCLES = [
    'Top_PT', 'Top_ECRL', 'Top_UI_UB5', 'Top_PQ', 
    'Top_FCU', 'Top_FDS4', 'Top_PL', 'Top_LU_RB5', 
    'Top_ECRB', 'Top_FCR', 'Top_ECU'
]

# 每个自由度的特征数
N_FEATURES_PER_DOF = len(SELECTED_MUSCLES)

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

# ================= 数据加载 =================
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
    """
    为了方便归一化，我们依然先把数据合并加载，但在 Dataset 里再拆分
    """
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
            
            # 在特征维度拼接: (L, Features_F + Features_D + Features_P)
            combined_feat = np.concatenate([feat_f, feat_d, feat_p], axis=1)
            
            # 转置为 CNN 格式: (Channels, Length)
            combined_feat = combined_feat.transpose(1, 0)
            
            data_list.append(combined_feat)
            label_list.append(get_label_from_k_c(k_str, c_str))
            
        except Exception as e:
            print(f"Error processing set {filename}: {e}")
            continue
            
    print(f"Successfully loaded {len(data_list)} samples.")
    return np.array(data_list), np.array(label_list)

class SeparatedMuscleDataset(Dataset):
    """
    [修改] 策略2专用的 Dataset
    在 getitem 时，将拼接好的数据拆分成三个部分，分别返回
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.n_feat = N_FEATURES_PER_DOF # 比如 11

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        # X shape is (Total_Channels, L) e.g., (33, 1000)
        full_data = self.X[idx]
        
        # 拆分
        x_f = full_data[0 : self.n_feat, :]
        x_d = full_data[self.n_feat : 2*self.n_feat, :]
        x_p = full_data[2*self.n_feat : , :]
        
        return x_f, x_d, x_p, self.y[idx]

# ================= 策略2：多头 ResNet 模型 =================

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetBranch(nn.Module):
    """
    单个 ResNet 分支，只负责提取特征，没有最后的分类层
    """
    def __init__(self, in_channels):
        super(ResNetBranch, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(16, 16, stride=1)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)
        # 最后一层输出 64 通道
        self.layer4 = self._make_layer(64, 128, stride=2) 
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_channels, out_channels, stride):
        return ResBlock(in_channels, out_channels, stride)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1) # 输出维度 128
        return x

class LateFusionResNet(nn.Module):
    """
    主模型：包含3个分支 + 后期融合层
    """
    def __init__(self, feature_per_dof):
        super(LateFusionResNet, self).__init__()
        
        # 定义三个独立的分支，权重不共享
        self.branch_f = ResNetBranch(in_channels=feature_per_dof)
        self.branch_d = ResNetBranch(in_channels=feature_per_dof)
        self.branch_p = ResNetBranch(in_channels=feature_per_dof)
        
        # 融合层
        # 每个分支输出 128 维，共 3 个分支 -> 384 维
        fusion_dim = 128 * 3 
        
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2) # 输出 K, C
        )

    def forward(self, x_f, x_d, x_p):
        # 1. 独立提取特征
        feat_f = self.branch_f(x_f)
        feat_d = self.branch_d(x_d)
        feat_p = self.branch_p(x_p)
        
        # 2. 拼接 (Late Fusion)
        combined = torch.cat([feat_f, feat_d, feat_p], dim=1)
        
        # 3. 预测
        out = self.fc(combined)
        return out

# ================= 评估函数 =================
def evaluate_performance(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ================= 主程序 =================
def train_and_test():
    try:
        X_all, y_all_raw = load_multimodal_data(DATA_ROOT)
    except Exception as e:
        print(f"Fatal Error: {e}")
        return

    if len(X_all) == 0:
        print("No valid data loaded.")
        return

    N, C, L = X_all.shape
    print(f"Total Input Shape: {X_all.shape}. Will be split into 3 branches of {N_FEATURES_PER_DOF} channels.")
    
    # 1. Input 归一化 (整体归一化)
    X_temp = X_all.transpose(0, 2, 1).reshape(-1, C)
    input_scaler = StandardScaler()
    X_scaled = input_scaler.fit_transform(X_temp).reshape(N, L, C).transpose(0, 2, 1)
    
    # 2. Target 归一化
    target_scaler = TargetScaler()
    target_scaler.fit(y_all_raw)
    y_all_norm = target_scaler.transform(y_all_raw)
    
    # 3. 拆分
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all_norm, test_size=0.1, random_state=42)
    
    # 使用修改后的 Dataset
    train_loader = DataLoader(SeparatedMuscleDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SeparatedMuscleDataset(X_test, y_test), batch_size=1, shuffle=False)

    # --- 初始化 LateFusion 模型 ---
    model = LateFusionResNet(feature_per_dof=N_FEATURES_PER_DOF).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    loss_history = []
    
    print("\nStart Training (Strategy 2: Late Fusion ResNet)...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for x_f, x_d, x_p, y in train_loader:
            # 数据送入 GPU
            x_f, x_d, x_p, y = x_f.to(DEVICE), x_d.to(DEVICE), x_p.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            # 传入三个独立的输入
            pred = model(x_f, x_d, x_p)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # ================= 测试与评估 =================
    print("\nRunning Final Evaluation...")
    model.eval()
    
    preds_norm_list = []
    gts_norm_list = []
    
    with torch.no_grad():
        for x_f, x_d, x_p, y in test_loader:
            x_f, x_d, x_p, y = x_f.to(DEVICE), x_d.to(DEVICE), x_p.to(DEVICE), y.to(DEVICE)
            
            pred = model(x_f, x_d, x_p)
            
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
    print("           Late Fusion ResNet 综合性能评估报告")
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
    plt.title(f"K Value (Late Fusion)\n$R^2$={r2_k:.3f}")
    plt.xlabel("True K")
    plt.ylabel("Predicted K")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(gts[:, 1], preds[:, 1], c='orange', marker='^', alpha=0.6, label='Data')
    min_c, max_c = min(gts[:, 1].min(), preds[:, 1].min()), max(gts[:, 1].max(), preds[:, 1].max())
    plt.plot([min_c, max_c], [min_c, max_c], 'r--', label='Ideal')
    plt.title(f"C Value (Late Fusion)\n$R^2$={r2_c:.3f}")
    plt.xlabel("True C")
    plt.ylabel("Predicted C")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "model/resnet_late_fusion_model.pth")
    joblib.dump(input_scaler, 'model/resnet_lf_input_scaler.pkl')
    joblib.dump(target_scaler, 'model/resnet_lf_target_scaler.pkl')
    print("Models saved.")

if __name__ == '__main__':
    train_and_test()