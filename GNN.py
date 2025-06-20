import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import r2_score

adj_df = pd.read_csv('Adjacency matrix1.csv', index_col=0)
adj_matrix = adj_df.values

edge_index = []
edge_weight = []
num_nodes = adj_matrix.shape[0]
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j and not np.isinf(adj_matrix[i, j]):
            edge_index.append([i, j])
            edge_weight.append(1.0 / adj_matrix[i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight, dtype=torch.float)

# 3. 读取交通流量 CSV 文件，假设命名为 "traffic.csv"，第一列为索引，第二列为交通流量
traffic_df = pd.read_csv('AADT_recoded.csv', index_col=0)
# traffic 流量（目标值），转为 tensor，shape 为 [num_nodes]
traffic = torch.tensor(traffic_df.iloc[:, 0].values, dtype=torch.float)

# 提取目标值（交通流量）作为 y
traffic = torch.tensor(traffic_df.iloc[:, 0].values, dtype=torch.float)

# ✅ 4. 提取节点特征（现在包含4个额外特征：经度纬度、速度、人口）
# 第三列到第六列对应的列名可能是：["Latitude", "Longitude", "Speed", "Population2021"]
# 如果你不确定列名，可以用 print(traffic_df.columns) 检查
feature_cols = traffic_df.columns[1:7]  # 取第2到第5列（从0开始数）
print(feature_cols)
node_features = torch.tensor(traffic_df[feature_cols].values, dtype=torch.float)

# ⚠️（可选）对特征进行归一化处理（建议做，避免不同量级影响训练）
node_features = (node_features - node_features.mean(dim=0)) / node_features.std(dim=0)

# 5. 构造图数据对象
data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, y=traffic)

# 6. 划分训练和验证集
indices = torch.randperm(num_nodes)
train_size = int(num_nodes * 0.7)
train_idx = indices[:train_size]
val_idx = indices[train_size:]


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)

        self.res1 = nn.Linear(in_channels, hidden_channels)  # 用于conv1残差
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x_ = self.res1(x)  # 映射以便加残差
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x + x_)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_ = x
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x + x_)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_ = x
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x + x_)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv4(x, edge_index, edge_weight)
        return x


in_channels = node_features.shape[1]  # 现在是 4
hidden_channels = 16
out_channels = 1
model = GCN(in_channels, hidden_channels, out_channels)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):  # 避免 sqrt(0)
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = RMSELoss()

train_losses = []
val_losses = []
train_r2_scores = []
val_r2_scores = []

num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    train_loss = criterion(out[train_idx].squeeze(), data.y[train_idx])
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data)
        val_loss = criterion(out[val_idx].squeeze(), data.y[val_idx])
        # 计算 R²
        pred = out.squeeze().cpu().numpy()
        y_true = data.y.cpu().numpy()
        train_r2 = r2_score(y_true[train_idx], pred[train_idx])
        val_r2 = r2_score(y_true[val_idx], pred[val_idx])
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    train_r2_scores.append(train_r2)
    val_r2_scores.append(val_r2)

    print(f"Epoch {epoch + 1}: Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}, "
          f"Train R² = {train_r2:.4f}, Val R² = {val_r2:.4f}")

from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred, eps=1e-6):
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

model.eval()
with torch.no_grad():
    predictions = model(data).squeeze().cpu().numpy()
    true_values = data.y.cpu().numpy()

    # 分别取训练集和验证集
    train_true = true_values[train_idx]
    train_pred = predictions[train_idx]
    val_true = true_values[val_idx]
    val_pred = predictions[val_idx]

    # R²
    train_r2 = r2_score(train_true, train_pred)
    val_r2 = r2_score(val_true, val_pred)

    # MAE
    train_mae = mean_absolute_error(train_true, train_pred)
    val_mae = mean_absolute_error(val_true, val_pred)

    # MAPE
    train_mape = mean_absolute_percentage_error(train_true, train_pred)
    val_mape = mean_absolute_percentage_error(val_true, val_pred)

    # MSE
    train_mse = mean_squared_error(train_true, train_pred)
    val_mse = mean_squared_error(val_true, val_pred)

    # RMSE
    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)

print("🔍 最终评估指标：")
print(f"Train R²:  {train_r2:.4f} | Val R²:  {val_r2:.4f}")
print(f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")
print(f"Train MAPE: {train_mape:.2f}%  | Val MAPE: {val_mape:.2f}%")
print(f"Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}")
print(f"Train RMSE:{train_rmse:.4f} | Val RMSE:{val_rmse:.4f}")


import matplotlib.pyplot as plt
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制 R² 曲线
plt.figure()
plt.plot(train_r2_scores, label='Train R²')
plt.plot(val_r2_scores, label='Validation R²')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.legend()
plt.show()