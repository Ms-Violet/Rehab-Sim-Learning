"""
InceptionTime.py

功能:
  - 使用 InceptionTime 结构对肌肉信号序列进行回归，预测物理参数 K 与 C。
  - 读取 ./data2 下 F/D/P 三自由度数据，按文件名配对组成一个样本。

数据约定:
  - 文件名形如: {k}_{c}_{F|D|P}_p.csv；k、c 会除以 1000，并对标签做 log 变换用于训练。
  - 输入通道仅使用 SELECTED_MUSCLES 中列，缺失列补 0；序列长度 SEQ_LEN=800，不足补 0、超长截断。

输出:
  - 训练曲线与评估指标打印；可视化散点/损失曲线。
  - 模型权重保存到: model/fusion_model_final_10pct.pth (可能覆盖同名文件)。

运行:
  - 在项目根目录执行: python InceptionTime.py
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置参数
# ==========================================
DATA_ROOT = './data2' 
SUB_DIRS = {'F': 'Flexion_p', 'D': 'Deviation_p', 'P': 'Pronation_p'}

SELECTED_MUSCLES = [
    'Top_PT', 'Top_ECRL', 'Top_UI_UB5', 'Top_PQ', 
    'Top_FCU', 'Top_FDS4', 'Top_PL', 'Top_LU_RB5', 
    'Top_ECRB', 'Top_FCR', 'Top_ECU'
]

SEQ_LEN = 800      
BATCH_SIZE = 16
EPOCHS = 100        
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 数据处理
# ==========================================
def get_label_from_k_c(k_str, c_str):
    k = float(k_str) / 1000.0
    c = float(c_str) / 1000.0
    return np.array([np.log(k + 1e-8), np.log(c + 1e-8)], dtype=np.float32)

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

class MuscleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==========================================
# 3. 评估函数
# ==========================================
def evaluate_performance(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return rmse, mae, r2

# ==========================================
# 4. InceptionTime 模型
# ==========================================
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super(InceptionModule, self).__init__()
        self.use_bottleneck = in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        c_in = bottleneck_channels if self.use_bottleneck else in_channels
        self.conv_layers = nn.ModuleList()
        for k in kernel_sizes:
            self.conv_layers.append(nn.Conv1d(c_in, out_channels, kernel_size=k, padding=k // 2, bias=False))
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        if self.use_bottleneck: x = self.bottleneck(x)
        conv_outputs = [layer(x) for layer in self.conv_layers]
        pool_output = self.conv_pool(self.maxpool(input_tensor))
        cat_output = torch.cat(conv_outputs + [pool_output], dim=1)
        return self.act(self.bn(cat_output))

class ShortcutBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShortcutBlock, self).__init__()
        self.use_projection = (in_channels != out_channels)
        if self.use_projection:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x, residual):
        if self.use_projection: residual = self.bn(self.projection(residual))
        return x + residual

class InceptionTime(nn.Module):
    def __init__(self, input_dim, output_dim=2, num_blocks=3, use_residual=True):
        super(InceptionTime, self).__init__()
        self.num_blocks = num_blocks
        self.use_residual = use_residual
        in_channels = input_dim
        out_channels = 32
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        
        for i in range(num_blocks):
            module_out_dim = out_channels * 4 
            block = nn.Sequential(
                InceptionModule(in_channels if i == 0 else module_out_dim, out_channels),
                InceptionModule(module_out_dim, out_channels),
                InceptionModule(module_out_dim, out_channels) 
            )
            self.blocks.append(block)
            if self.use_residual:
                self.shortcuts.append(ShortcutBlock(in_channels if i == 0 else module_out_dim, module_out_dim))
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 4, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        residual = x
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
            if self.use_residual:
                x = self.shortcuts[i](x, residual)
                residual = x
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

# ==========================================
# 5. 主程序
# ==========================================
def main():
    if not os.path.exists(DATA_ROOT):
        print(f"Error: {DATA_ROOT} not found.")
        return

    # --- 1. 加载 ---
    try:
        X_all, y_all_log = load_multimodal_data(DATA_ROOT)
    except Exception as e:
        print(f"Fatal Error: {e}")
        return

    if len(X_all) == 0:
        print("No valid data loaded. Exiting.")
        return

    # --- 2. 归一化与拆分 ---
    N, L, F = X_all.shape
    print(f"Final Input Shape: {X_all.shape} (Features={F})")
    
    scaler = StandardScaler()
    X_reshaped = X_all.reshape(-1, F)
    X_scaled = scaler.fit_transform(X_reshaped).reshape(N, L, F)

    # [修改] test_size ��为 0.1
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all_log, test_size=0.1, random_state=42)
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    train_loader = DataLoader(MuscleDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(MuscleDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. 初始化 ---
    model = InceptionTime(input_dim=F, output_dim=2, num_blocks=3).to(DEVICE)
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. 训练 ---
    train_hist = []
    print("\nStart Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)
        train_hist.append(train_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f}")

    # --- 5. 最终评估与可视化 ---
    print("\nRunning Final Evaluation on Test Set...")
    model.eval()
    preds_log, gts_log = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            out = model(inputs)
            preds_log.append(out.cpu().numpy())
            gts_log.append(targets.numpy())
    
    # 还原数据： Log -> Real
    preds = np.exp(np.concatenate(preds_log))
    gts = np.exp(np.concatenate(gts_log))

    # [新增] 打印详细预测表
    print("\n" + "="*60)
    print(f"{'Sample':<8} | {'True K':<10} {'Pred K':<10} | {'True C':<10} {'Pred C':<10}")
    print("-" * 60)
    for i in range(len(preds)):
        t_k, t_c = gts[i]
        p_k, p_c = preds[i]
        print(f"{i:<8} | {t_k:<10.4f} {p_k:<10.4f} | {t_c:<10.4f} {p_c:<10.4f}")
    print("="*60)

    # 计算指标
    rmse_k, mae_k, r2_k = evaluate_performance(gts[:, 0], preds[:, 0])
    rmse_c, mae_c, r2_c = evaluate_performance(gts[:, 1], preds[:, 1])

    print("\n" + "="*60)
    print("           综合性能评估报告")
    print("="*60)
    print(f"{'Metric':<10} | {'Parameter K':<15} | {'Parameter C':<15}")
    print("-" * 60)
    print(f"{'RMSE':<10} | {rmse_k:<15.6f} | {rmse_c:<15.6f}")
    print(f"{'MAE':<10} | {mae_k:<15.6f} | {mae_c:<15.6f}")
    print(f"{'R^2':<10} | {r2_k:<15.4f} | {r2_c:<15.4f}")
    print("="*60 + "\n")

    # --- 可视化 ---
    plt.figure(figsize=(16, 6))
    
    # 图1: Train Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_hist, label='Train Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.legend()
    plt.grid(alpha=0.3)

    # 图2: K值评估
    plt.subplot(1, 3, 2)
    plt.scatter(gts[:, 0], preds[:, 0], c='b', alpha=0.6, label='Data')
    min_k, max_k = min(gts[:, 0].min(), preds[:, 0].min()), max(gts[:, 0].max(), preds[:, 0].max())
    plt.plot([min_k, max_k], [min_k, max_k], 'r--', label='Ideal')
    plt.title(f'K Value\n$R^2$={r2_k:.3f}')
    plt.xlabel('True K')
    plt.ylabel('Predicted K')
    plt.legend()
    plt.grid(alpha=0.3)

    # 图3: C值评估
    plt.subplot(1, 3, 3)
    plt.scatter(gts[:, 1], preds[:, 1], c='orange', marker='^', alpha=0.6, label='Data')
    min_c, max_c = min(gts[:, 1].min(), preds[:, 1].min()), max(gts[:, 1].max(), preds[:, 1].max())
    plt.plot([min_c, max_c], [min_c, max_c], 'r--', label='Ideal')
    plt.title(f'C Value\n$R^2$={r2_c:.3f}')
    plt.xlabel('True C')
    plt.ylabel('Predicted C')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    torch.save(model.state_dict(), "model/fusion_model_final_10pct.pth")
    print("Model saved.")

if __name__ == "__main__":
    main()
