// IO部分有点不太好搞，所以这里我写了训练的矩阵操作库，写了训练的辅助函数和ANN定义，比较合适的方式是pybind到py进行训练
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

namespace Mat {
    inline Matrix zeros(int r, int c) { return Matrix(r, Vector(c, 0.0)); }

    inline Matrix randn(int r, int c, double scale = 1.0) {
        Matrix m(r, Vector(c));
        std::mt19937 gen(42);
        std::normal_distribution<double> d(0.0, 1.0);
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
                m[i][j] = d(gen) * scale;
        return m;
    }

    inline Matrix multiply(const Matrix& a, const Matrix& b) {
        Matrix c(a.size(), Vector(b[0].size(), 0.0));
        for (size_t i = 0; i < a.size(); ++i)
            for (size_t k = 0; k < b.size(); ++k)
                for (size_t j = 0; j < b[0].size(); ++j)
                    c[i][j] += a[i][k] * b[k][j];
        return c;
    }

    inline Matrix add(const Matrix& a, const Matrix& b) {
        Matrix c = a;
        for (size_t i = 0; i < c.size(); ++i)
            for (size_t j = 0; j < c[0].size(); ++j)
                c[i][j] += b[i][j];
        return c;
    }

    inline Matrix subtract(const Matrix& a, const Matrix& b) {
        Matrix c = a;
        for (size_t i = 0; i < c.size(); ++i)
            for (size_t j = 0; j < c[0].size(); ++j)
                c[i][j] -= b[i][j];
        return c;
    }

    inline Matrix T(const Matrix& a) {   // 转置
        Matrix c(a[0].size(), Vector(a.size()));
        for (size_t i = 0; i < a.size(); ++i)
            for (size_t j = 0; j < a[0].size(); ++j)
                c[j][i] = a[i][j];
        return c;
    }

    inline Matrix rowMean(const Matrix& a) {   // 关键：把 (batch, cols) 压成 (1, cols)
        Matrix mu(1, Vector(a[0].size(), 0.0));
        for (size_t j = 0; j < a[0].size(); ++j) {
            double s = 0.0;
            for (size_t i = 0; i < a.size(); ++i) s += a[i][j];
            mu[0][j] = s / a.size();
        }
        return mu;
    }

    inline Matrix apply(const Matrix& a, double (*f)(double)) {
        Matrix c = a;
        for (auto& row : c)
            for (double& v : row) v = f(v);
        return c;
    }
}

namespace Act {
    inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    inline double dsigmoid(double x) { double s = sigmoid(x); return s * (1.0 - s); }
    inline Matrix dsigmoidM(const Matrix& x) {
        Matrix c = x;
        for (auto& row : c)
            for (double& v : row) v = dsigmoid(v);
        return c;
    }
}

// 人工神经网络(Artificial Neural Network)类
// 实现了一个简单的前馈神经网络，包含一个输入层、一个隐藏层和一个输出层
class ANN {
    // 网络架构参数
    int in;    // 输入层神经元数量
    int hid;   // 隐藏层神经元数量
    int out;   // 输出层神经元数量
    
    // 模型参数（权重和偏置）
    Matrix W1; // 输入层到隐藏层的权重矩阵
    Matrix b1; // 隐藏层的偏
    Matrix W2; // 隐藏层到输出层的权重矩阵
    Matrix b2; // 输出层的偏
    
    // 用于反向传播的中间计算结果
    Matrix z1; // 隐藏层的加权输入
    Matrix a1; // 隐藏层的激活值
    Matrix z2; // 输出层的加权输入
    Matrix a2; // 输出层的激活值

public:
    ANN(int input_size, int hidden_size, int output_size)
        : in(input_size), hid(hidden_size), out(output_size) {
        double s1 = std::sqrt(2.0 / in), s2 = std::sqrt(2.0 / hid);
        W1 = Mat::randn(in, hid, s1);
        b1 = Mat::zeros(1, hid);
        W2 = Mat::randn(hid, out, s2);
        b2 = Mat::zeros(1, out);
    }

    // 前向传播函数，计算网络的输出
    // x: 输入数据矩阵，形状为(batch_size, input_size)
    // 返回: 网络的输出预测，形状为(batch_size, output_size)
    Matrix forward(const Matrix& x) {
        // 计算隐藏层的加权输入
        z1 = Mat::add(Mat::multiply(x, W1), b1);
        // 应用sigmoid激活函数得到隐藏层的输出
        a1 = Mat::apply(z1, Act::sigmoid);
        // 计算输出层的加权输入
        z2 = Mat::add(Mat::multiply(a1, W2), b2);
        // 应用sigmoid激活函数得到最终输出
        a2 = Mat::apply(z2, Act::sigmoid);
        return a2;
    }

    // 计算交叉熵损失
    // y_pred: 模型的预测输出
    // y_true: 真实标签
    // 返回: 平均交叉熵损失值
    double loss(const Matrix& y_pred, const Matrix& y_true) {
        const double eps = 1e-10;
        double l = 0.0;
        for (size_t i = 0; i < y_pred.size(); ++i)
            for (size_t j = 0; j < y_pred[0].size(); ++j) {
                double p = std::max(eps, std::min(1.0 - eps, y_pred[i][j]));
                l -= y_true[i][j] * std::log(p) + (1.0 - y_true[i][j]) * std::log(1.0 - p);
            }
        return l / (y_pred.size() * y_pred[0].size());
    }

    // 反向传播算法，更新模型参数
    // x: 输入数据
    // y_true: 真实标签
    // lr: 学习率，控制参数更新的步长
    void backward(const Matrix& x, const Matrix& y_true, double lr = 0.01) {
        int m = x.size();
        Matrix dz2 = Mat::subtract(a2, y_true);                          // (m, out)
        Matrix dW2 = Mat::scalarMultiply(Mat::multiply(Mat::T(a1), dz2), 1.0 / m);
        Matrix db2 = Mat::rowMean(dz2);                                  // (1, out)

        Matrix da1 = Mat::multiply(dz2, Mat::T(W2));                     // (m, hid)
        Matrix dz1 = Mat::elementWiseMultiply(da1, Act::dsigmoidM(z1));  // (m, hid)
        Matrix dW1 = Mat::scalarMultiply(Mat::multiply(Mat::T(x), dz1), 1.0 / m);
        Matrix db1 = Mat::rowMean(dz1);                                  // (1, hid)

        W2 = Mat::subtract(W2, Mat::scalarMultiply(dW2, lr));
        b2 = Mat::subtract(b2, Mat::scalarMultiply(db2, lr));
        W1 = Mat::subtract(W1, Mat::scalarMultiply(dW1, lr));
        b1 = Mat::subtract(b1, Mat::scalarMultiply(db1, lr));
    }

    // 训练模型
    // X: 训练数据
    // y: 训练标签
    // epochs: 训练的轮数
    // batch_size: 批次大小
    // lr: 学习率
    // 返回: 每轮训练的损失值数组
    std::vector<double> train(const Matrix& X, const Matrix& y,
                              int epochs = 1000, int batch_size = 32, double lr = 0.01) {
        std::vector<double> losses;
        std::vector<int> idx(X.size());
        std::iota(idx.begin(), idx.end(), 0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(idx.begin(), idx.end(), std::mt19937(epoch));
            double epoch_loss = 0.0;

            for (int i = 0; i < X.size(); i += batch_size) {
                int end = std::min(i + batch_size, static_cast<int>(X.size()));
                Matrix batch_X(end - i, Vector(X[0].size()));
                Matrix batch_y(end - i, Vector(y[0].size()));
                for (int j = i; j < end; ++j) {
                    batch_X[j - i] = X[idx[j]];
                    batch_y[j - i] = y[idx[j]];
                }

                Matrix y_pred = forward(batch_X);
                epoch_loss += loss(y_pred, batch_y) * (end - i);
                backward(batch_X, batch_y, lr);
            }
            epoch_loss /= X.size();
            losses.push_back(epoch_loss);
            if ((epoch + 1) % 100 == 0)
                std::cout << "Epoch " << epoch + 1 << "/" << epochs
                          << "  loss=" << epoch_loss << '\n';
        }
        return losses;
    }

    // 使用训练好的模型进行预测
    // x: 输入数据
    // threshold: 分类阈值，大于等于此值预测为类别1，否则为类别0
    // 返回: 预测的类别标签
    Matrix predict(const Matrix& x, double threshold = 0.5) {
        Matrix prob = forward(x);
        Matrix out = prob;
        for (auto& row : out)
            for (double& v : row) v = (v >= threshold) ? 1.0 : 0.0;
        return out;
    }

    // 计算模型的准确率
    // y_pred: 模型的预测输出
    // y_true: 真实标签
    // 返回: 准确率，范围在0到1之间
    double accuracy(const Matrix& y_pred, const Matrix& y_true) {
        int correct = 0, total = 0;
        for (size_t i = 0; i < y_pred.size(); ++i)
            for (size_t j = 0; j < y_pred[0].size(); ++j) {
                if (std::abs(y_pred[i][j] - y_true[i][j]) < 1e-6) ++correct;
                ++total;
            }
        return total ? static_cast<double>(correct) / total : 0.0;
    }
};
