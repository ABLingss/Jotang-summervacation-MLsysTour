计算图
AI模型或者网络，往往是由若干基本块/层组成，例如我们最基础的AI模型 - 多层感知机模型（MLP）

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model class.
    Defines a neural network with four linear layers and sigmoid activations.
    """

    def __init__(self) -> object:
        super().__init__()
        # Define model layers
        self.layer0 = nn.Linear(8, 8, bias=True)
        self.layer1 = nn.Linear(8, 4, bias=True)
        self.layer2 = nn.Linear(4, 2, bias=True)
        self.layer3 = nn.Linear(2, 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after forward pass.
        """
        x = self.layer0(x)
        x = torch.sigmoid(x)
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        x = self.layer3(x)
        return x
该MLP模型是由四个Linear层跟Sigmoid组成，Linear层的计算可以视作是一个矩阵乘法，例如nn.Linear(8, 8, bias =True)数学描述的是

其中$$\mathbb{R}^8$$表示一个维度为8的向量，$$\mathbb{R}^{8\times8}$$表示一个8乘8的矩阵，大家可以以“xxx的形式化描述是什么？”的promote向AI提问以获取某个模型的数学描述


1570×966 72.5 KB
我们无需关心为什么如此定义的MLP模型能够对图形进行分类，就像我们学习分子时无需关心为什么分子总是在进行无规则的布朗运动，也许这的确能够被解释，但是这不是我们所要关心的问题。我们需要关心的只有MLP是由交错的矩阵乘法和Sigmoid函数构成，它的输入是一个向量，输出也是一个向量。

为了提升对于模型计算描述的泛化性，我们引入了计算图的这个概念，正如算法题往往会给题目包装一个生活情景，我们拨开包装发现实际是某一个算法问题。我们也可以想象模型计算结构其实是对一个图的包装，图中的节点描述的是一种计算，例如矩阵乘法。我们将模型以图的形式来描述，更利于我们对其进行分析和变换。

具体到上述例子而言，torch贴心地为我们提供了将一个模型以ONNX格式的计算图形式进行导出的工具

ONNX ONNX | About 1 格式是一种常用的跨平台的网络模型表示格式，方便跨平台部署和调试。需要注意的是实际上ONNX只是计算图的一个实例化表示，由计算图描述的网络模型除了ONNX格式还可以有多种其他格式的实例化格式表示，只需要语意上是等价的即可。

ONNX能够将网络模型表示为一个计算图，图中的节点是计算原语（primitive），基础的计算原语（primitive）可以表达复杂的网络结构。
```py
model = MLP()
example_x = torch.randn(97, 8, dtype=torch.float32)
torch.onnx.export(model, example_x, "mlp.onnx", opset_version=12)
```
需要注意的是，在导出过程中，torch.onnx.export 实际上是在torch层面对该模型执行了一次，并且捕获执行时的流程，我们往往将这个过程称之为trace，然后根据执行时捕获的信息构造计算图；基于这个前提，如果你在torch网络中根据条件分支判断来执行某段计算内容时，很有可能只能捕获到实际执行分支的内容。

换句话说，trace过程对于模型动态性的支持是比较有限的，无论是动态的执行模型代码还是形状动态的输入。

初学者不必过分在意这部分。

利用网页版 Netron 2 打开mlp.onnx即可视化查看ONNX网络模型


1802×1684 179 KB
模型网络的一次执行(forward)可以看作是输入onnx::Gemm_0，该输入流入第一个Gemm节点进行计算，计算结果作为后续Sigmoid节点的输入，进行计算，如此反复直至输出节点。

虽然上述只介绍了MLP这种简单模型的计算图表示，但是计算图的表达能力远不止于此，复杂的模型结构如Transformer结构仍可以通过计算图来表示。

为了简化本节对计算图讨论，本节只涉及网络模型前向传播(forward)的计算部分，与我们常说的模型推理过程联系比较紧密，即模型权重是已经被训练好的数据，不涉及梯度的计算和对模型内权重矩阵等信息的更新。

如果是要对模型进行训练和微调，则还需要考虑反向传播(backward)计算对于模型权重矩阵等内容的更新，涉及更复杂一些的机器学习基础知识。关于这部分内容建议参考李沐动手学机器学习的 5.3.1. Forward Propagation ¶ 3小节及其相关学习视频。

计算图优化
抽象出计算图的最大的好处之一就是能够方便的对计算进行描述，我们能够针对性地对计算进行分析/优化和适配各种类型的硬件，使其能够更高效的运行。

我们以上节MLP模型计算图的一个子图来介绍计算图层面优化的一个经典例子 — 算子融合


472×512 41 KB
假设我们在部署时将上述的子图使用python来实现并且使用python解释器运行，其等价于下述代码
```py
# m = 97 , n = k =8
for i in range(m):          
    for j in range(n):      
        for a in range(k):  
            result[i][j] += x[i][a] * weight[a][j]
        result[i][j] += bias[j] # A
for i in range(m):
    for j in range(n):
        result[i][j] = 1 / (1 + math.exp(-M[i][j])) #B
```
第一个计算节点matmul计算出中间结果result(shape=[97,8])#A写回在内存中，第二个计算节点sigmoid将result(shape=[97,8])#A的每个数读出、计算最后再写回。result中每个元素都经历了如下步骤


1182×416 23.4 KB
result 从内存中读取和向写入了两次，这样的实现显然是很慢的，相比读取寄存器的，向内存读取操作是非常耗费时间的。

通过与result计算模式的观察我们很容易就可以发现这两个嵌套循环是可以被合并的，
```py
# m = 97 , n = k =8
for i in range(m):          
    for j in range(n):      
        for a in range(k):  
            result[i][j] += x[i][a] * weight[a][j]
        result[i][j] += bias[j] # A
        result[i][j] = 1 / (1 + math.exp(-M[i][j])) #B
```
保证结果正确的前提下，向内存的读写次数直接减半了，更多的操作都发生在寄存器层面，大大提升了运行时的性能。


1170×382 20.7 KB
上述例子通过简单的分析就能大大提升运行时的性能，这就是计算图层面的优化，这种优化手段是架构无关的，因为在冯诺伊曼体系结构中，不管面向的的硬件是什么，都能够从减少全局内存的读取中受益。

为了降低AI开发人员的心智负担和计算图描述的可拓展性，我们往往是提供的计算原语都是torch.matmul、torch.sigmoid这种单算子，而不会是一个名为torch.matmul-sigmoid的融合算子，即使这样做能够显著提升性能，这种优化机会很多时候是需要由优化框架自动发掘和完成。

自动算子优化
除了上述介绍的算子融合这样在图层面具有代表性的优化，还有算子层面的优化，即针对计算图中的特定（融合）算子如何在目标架构生成高性能的实现。假设我们面向的执行硬件是cpu且不调用高性能算子库的情况下，reduce_sum算子使用c语言描述编译出的代码会比python解释执行快非常多

# python
```py
def reduce_sum(array):
    sum = 0.0
    for num in array:
        sum += num
    return sum          
```

# C
```c
float reduce_sum(float* array, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum;
}
```
很大一部分原因在于c编译器在满足正确性的前提下自动地对代码进行更多的优化，针对reduce_sum算子，这种优化机会来自于c语言编译器对目标架构的了解，包括但不限于

了解目标架构硬件特性生成高效的汇编指令，例如向量化指令

分析目标架构的存储结构，优化内存访问顺序以更好的利用多级缓存

甚至有激进的编译器能够比编译出在多颗CPU物理核心上运行的代码。

例如编译器可能能够将上述C代码通过循环向量化优化变换(clang -O3 -ffast-math)为
```cpp
// Assume that size is divisible by 8
float reduce_sum_avx(float* array, int size) {
    // 确保数组是 32 字节对齐（AVX 要求对齐加载以获得最佳性能）
    if ((uintptr_t)array % 32 != 0) {
        printf("Warning: Array is not 32-byte aligned. Performance may be reduced.\n");
    }

    __m256 sum_vec = _mm256_setzero_ps();  // 初始化 8 个 float = 0

    // 主循环：每次处理 8 个 float
    for (int i = 0; i < size; i += 8) {
        __m256 data = _mm256_load_ps(&array[i]);  // 对齐加载 8 个 float
        sum_vec = _mm256_add_ps(sum_vec, data);  // 8 个 float 并行相加
    }

    // 横向求和：8 个 float -> 1 个 sum
    float sum = horizontal_sum_avx(sum_vec);
    return sum;
}

// 辅助函数：AVX 寄存器横向求和（8 个 float -> 1 个 float）
float horizontal_sum_avx(__m256 vec) {
    // 1. 将 8 个 float 分成 2 组 4 个，分别求和
    __m128 low = _mm256_extractf128_ps(vec, 0);  // 低 128 位（4 个 float）
    __m128 high = _mm256_extractf128_ps(vec, 1); // 高 128 位（4 个 float）
    __m128 sum128 = _mm_add_ps(low, high);       // 4 + 4 = 4 个 float

    // 2. 继续横向求和（4 -> 2 -> 1）
    sum128 = _mm_hadd_ps(sum128, sum128);  // 相邻相加 (a0+a1, a2+a3, a0+a1, a2+a3)
    sum128 = _mm_hadd_ps(sum128, sum128);  // 再次相加 (a0+a1+a2+a3, ...)

    // 3. 提取最终结果
    float sum;
    _mm_store_ss(&sum, sum128);  // 存储最低 32 位（即总和）
    return sum;
}
```
可以简单理解一条向量化指令(_mm256_add_ps)等价八条普通的标量指令(float add)，但是执行的速度可能是一样快的，利用向量化指令便能够提升程序的性能。

笔者在调试测试不同c语言编译器编译matmul生成代码的性能时惊讶发现gcc居然比clang编译出的代码快十多倍，经过对汇编的分析发现gcc编译出的代码能够更好地利用cpu缓存。这意味着同样的代码，你仅仅换一个编译器就能让你的代码快上10多倍。

手工算子优化及算子库
当然，编译器再牛在很多特定的情况下还是无法比拟经过专家优化的代码，这种差距来自各个方面

编译器为了保证其通用性在部分情况会做保守地优化

部分优化分析复杂度过高或者无法收敛

多数时无法获取执行时的profiling信息来反馈优化

无法感知代码的语义

难以生成多线程代码

以现实世界应用非常广泛的的y轴stencil为例子，其形式化描述如下


1534×612 39.4 KB
CPP代码描述如下
```cpp
#include <algorithm>
#include <immintrin.h>
#include <stdbool.h>

#define X 10240
#define Y 10240
#define IterNum 10000000
#define BLUR_SIZE 4
int image[X * Y];
int image_blur_origin[X * Y];
int image_blur_tile[X * Y];

void blur_y(int const image[X * Y], int image_blur[X * Y]) {
  for (int j = 0; j < Y; j++) {
    for (int i = 0; i < X; i++) {
      int sum = 0;
      for (int t = -BLUR_SIZE; t <= BLUR_SIZE; t++) {
        if (i + t >= 0 && i + t < X) {
          sum += image[(i + t) * Y + j];
        }
      }
      image_blur[i * Y + j] = sum / (2 * BLUR_SIZE + 1);
    }
  }
}
算子库的编写者可能会采取对外层循环进行分块变换的方式对上述代码进行优化

void blur_y_tile(int const image[X * Y], int image_blur[X * Y]) {
  for (int j = 0; j < Y; j += 16) {
    for (int i = 0; i < X; i++) {
      for (int k = 0; k < std::min(16, Y - j); k++) {
        int sum = 0;
        for (int t = -BLUR_SIZE; t <= BLUR_SIZE; t++) {
          if (i + t >= 0 && i + t < X) {
            sum += image[(i + t) * Y + (j + k)];
          }
        }
        image_blur[i * Y + (j + k)] = sum / (2 * BLUR_SIZE + 1);
      }
    }
  }
}
```
在我的个人PC下进行测试（clang -O3），保证结果相同的情况下，仅仅通过分块就让执行时间降低了一半以上。

[ RUN      ] clang.blur_origin
[       OK ] clang.blur_origin (2513 ms)
[ RUN      ] clang.blur_tile
[       OK ] clang.blur_tile (960 ms)
循环分块可以视为专家针对目标平台对算法进行手工优化的一个典型示例，除此之外还有各种各样的优化手段。

算子库的编写是一个人力密集型的工作，并且泛化性很有可能不好，表现在针对不同类的硬件平台、同一种硬件平台的不同代产品可能都需要专门进行适配；除此之外，极致性能优化的代码可能甚至需要内嵌汇编代码来达到更细致的控制，其代码可读性近乎没有，难以调试，依赖领域专家的经验。

该节所提到的手工算子优化更多的是为了极致压榨硬件平台的性能，与同学们在大一时算法课学到的算法设计有所区别。“手工算子优化”一般来说无法带来像冒泡排序到快排这种数量级的性能优化。

手写以及编译从来不是互斥的关系，在传统编译器时代，编译器+math库高性能是很经典的搭配。

面向CPU有名的线性代数算子库可以参考 Eigen 1，针对GPU的并行计算参考并行计算的章节。

问题
将 llama 3b 模型导出为onnx格式并用 onnx runtime 进行部署

profiling算子融合所带来的性能受益

结合汇编代码，尝试解释为什么gcc编译出的matmul算子性能是clang的十倍

解释为什么对循环进行分块可以带来性能的巨大提升

除了循环分块，还有哪些优化常见的优化手段，请使用代码举例说明；并且尝试将这些优化手段应用于blur算子，并且附上性能数据

异步内存拷贝相比同步拷贝有什么优势

调研如何分析一个矩阵乘法在单核心CPU上实现的的理论性能

请使用多线程在CPU上实现矩阵乘法，详细介绍在数据通信、计算任务划分等方面的优化

分析下述代码，foo和bar函数哪个运行更快，为什么？如何优化foo函数……
```c++
#include <cstdint>
#include <iostream>
#include <thread>

struct data {
  int a;
  int b;
};

void add_a(data &global_data) {
  for (int i = 0; i < 500000000; ++i) {
    global_data.a++;
  }
}

void add_b(data &global_data) {
  for (int i = 0; i < 500000000; ++i) {
    global_data.b++;
  }
}

void foo() {
  data data_x;
  std::thread t1([&data_x]() { add_a(data_x); });
  std::thread t2([&data_x]() { add_b(data_x); });

  t1.join();
  t2.join();
  uint64_t sum = data_x.a + data_x.b;
  std::cout << "Sum foo : " << sum << std::endl;
}

void bar() {
  data data_y;
  add_a(data_y);
  add_b(data_y);

  uint64_t sum = data_y.a + data_y.b;
  std::cout << "Sum bar : " << sum << std::endl;
}
```
调研如何分析一个矩阵乘法在单核心CPU上实现的的理论性能