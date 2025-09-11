# 简单设计一个ANN
# 就这样：（nn版）
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
# 纯np实现
import numpy as np


class ANN:
    def __init__(self, input_size=784, hidden_size=15, output_size=1):
        # 随机初始化权重（He初始化）
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        """sigmoid激活函数"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """sigmoid函数的导数"""
        return x * (1 - x)
    
    def forward(self, x):
        """前向传播"""
        # 展平输入（如果需要）
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        
        # 保存中间结果，用于反向传播
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # 二分类使用sigmoid
        
        return self.a2
    
    def compute_loss(self, y_pred, y_true):
        """计算二分类交叉熵损失 这是标准做法"""
        # 避免log(0)的情况
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # 交叉熵损失
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, x, y_true, learning_rate=0.01):
        """反向传播算法更新权重"""
        # 确保输入被展平
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        
        m = x.shape[0]  # 样本数量
        
        # 计算输出层的误差
        dz2 = self.a2 - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 计算隐藏层的误差
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 更新权重和偏置
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X_train, y_train, epochs=1000, batch_size=64, learning_rate=0.01):
        """训练模型"""
        # 确保y_train是二维数组
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        losses = []
        
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0
            # 分批训练
            for i in range(0, X_train.shape[0], batch_size):
                # 获取批次数据
                end = min(i + batch_size, X_train.shape[0])
                batch_X = X_train_shuffled[i:end]
                batch_y = y_train_shuffled[i:end]
                
                # 前向传播
                y_pred = self.forward(batch_X)
                
                # 计算损失
                loss = self.compute_loss(y_pred, batch_y)
                epoch_loss += loss * (end - i)  # 加权累加损失
                
                # 反向传播更新权重
                self.backward(batch_X, batch_y, learning_rate)
            
            # 计算平均损失
            epoch_loss /= X_train.shape[0]
            losses.append(epoch_loss)
            
            # 每100个epoch打印一次损失
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
        
        return losses
    
    def predict(self, x, threshold=0.5):
        """使用训练好的模型进行预测"""
        # 前向传播获取概率
        y_pred = self.forward(x)
        # 根据阈值转换为类别
        return (y_pred >= threshold).astype(int)


def prepare_data():
    """准备数据集"""
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    
    # 生成1000个样本的make_moons数据集
    X, y = make_moons(n_samples=1000, random_state=42)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 转换为numpy数组并调整y的形状
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).reshape(-1, 1)  # 添加维度以匹配模型输出
    y_test = np.array(y_test).reshape(-1, 1)
    
    return X_train, X_test, y_train, y_test

# 简单测试示例
if __name__ == "__main__":
    # 准备数据集（与DL_torch.py使用相同的数据集）
    X_train, X_test, y_train, y_test = prepare_data()
    
    # 创建并训练模型
    model = ANN(input_size=2, hidden_size=5, output_size=1)
    print("开始训练模型...")
    losses = model.train(X_train, y_train, epochs=500, batch_size=32, learning_rate=0.1)
    
    # 评估模型性能
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        # 计算准确率
        accuracy = np.mean(y_pred == y_test)
        print(f'Accuracy on test set: {accuracy * 100:.2f}%')
        return accuracy
    
    # 绘制决策边界
    def plot_decision_boundary(model, X, y):
        # 设置绘图范围
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        # 生成网格点
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # 预测网格点的类别
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
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
    def plot_losses(losses):
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_curve.png')
        plt.show()
    
    # 简单可视化（如果安装了matplotlib）
    try:
        import matplotlib.pyplot as plt
        
        # 评估模型
        accuracy = evaluate_model(model, X_test, y_test)
        
        # 绘制结果
        plot_losses(losses)
        
        # 合并训练和测试数据用于绘图
        X_combined = np.vstack((X_train, X_test))
        y_combined = np.vstack((y_train, y_test)).ravel()
        plot_decision_boundary(model, X_combined, y_combined)
        
    except ImportError:
        print("无法导入matplotlib库，无法进行可视化")

