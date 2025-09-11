# 简单设计一个ANN
# 就这样：
"""
class ANN(nn.Module):
    def __init__(self, input_size=784, hidden_size=15, output_size=10):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x  # 不加 softmax，CrossEntropyLoss 会处理
"""
# 要求二分类
# DATASET我们就用make_moons，调噪声咱就不干了

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义ANN模型类
class ANN(nn.Module):
    def __init__(self, input_size=2, hidden_size=15, output_size=1):
        super(ANN, self).__init__()
        # make_moons数据集是2维特征，所以input_size设为2
        # 二分类任务，output_size设为1
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x

# 准备数据集
def prepare_data():
    # 生成1000个样本的make_moons数据集
    X, y = make_moons(n_samples=1000, random_state=42)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)  # 添加维度以匹配模型输出 这个说到底还是数据集的问题
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    return X_train, X_test, y_train, y_test

# 训练模型
def train_model(model, X_train, y_train, X_test, y_test, epochs=1000, learning_rate=0.1):
    criterion = nn.BCEWithLogitsLoss()  # 结合了sigmoid和BCELoss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录训练损失
        train_losses.append(loss.item())
        
        # 在测试集上评估
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
        
        # 每100个epoch打印一次
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
    
    return train_losses, test_losses

# 评估模型性能
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        # 应用sigmoid并将输出转换为类别（0或1）
        predicted = torch.round(torch.sigmoid(outputs))
        
        # 计算准确率
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Accuracy on test set: {accuracy * 100:.2f}%')
    
    return accuracy

# 绘制决策边界
def plot_decision_boundary(model, X, y):
    model.eval()
    
    # 设置绘图范围
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # 生成网格点
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 转换为张量并预测
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        Z = model(grid)
        Z = torch.sigmoid(Z).numpy()
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.title('ANN Decision Boundary on make_moons Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('decision_boundary.png')
    plt.show()

# 绘制损失曲线
def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

# 主函数
def main():
    # 准备数据
    X_train, X_test, y_train, y_test = prepare_data()
    
    # 创建模型
    model = ANN(input_size=2, hidden_size=15, output_size=1)
    
    # 训练模型
    print("Starting training...")
    train_losses, test_losses = train_model(
        model, X_train, y_train, X_test, y_test, epochs=1000, learning_rate=0.1
    )
    
    # 评估模型
    accuracy = evaluate_model(model, X_test, y_test)
    
    # 绘制结果
    plot_losses(train_losses, test_losses)
    # 合并训练和测试数据用于绘图
    X_combined = np.vstack((X_train.numpy(), X_test.numpy()))
    y_combined = np.vstack((y_train.numpy(), y_test.numpy())).ravel()
    plot_decision_boundary(model, X_combined, y_combined)
    
    # 保存模型
    torch.save(model.state_dict(), 'ann_model.pth')
    print("Model saved as 'ann_model.pth'")

if __name__ == "__main__":
    main()
