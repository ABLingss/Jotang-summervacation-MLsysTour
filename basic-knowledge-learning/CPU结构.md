## CPU 结构

### 不同的内存层级间的特点（register，L1 cache，Memory….）

我们需要知道有那些内存层级，我们阅读了[cache高速缓存](https://zhuanlan.zhihu.com/p/482651908)，了解到了CPU的内存层级，分别是寄存器，L1 cache，L2 cache，Memory。

```plaintext
CPU寄存器--L1 cache--L2--L3--主存memory--磁盘等大容量存储器
```

计算机在运行程序时首先将程序从磁盘读取到主存，然后CPU按规则从主存中取出指令、数据并执行指令

> 计算机在运行程序时首先将程序从磁盘读取到主存，然后CPU按规则从主存中取出指令、数据并执行指令，但是直接从主存（一般用DRAM制成）中读写是很慢的，所以我们引入了cache。
> 在执行程序前，首先会试图把要用到的指令、数据从主存移到cache中，然后在执行程序时直接访问cache。如果指令、数据在cache中，那么我们能很快地读取出来，这称为“命中（hit）”；如果指令、数据不在cache中，我们仍旧要从主存中拿指令、数据，这称为“不命中（miss）”。命中率对于cache而言是很重要的。
> 现代处理器一般有三层cache，分别称为L1 cache、L2 cache、L3 cache。L1 cache离CPU核最近，存储信息的读取速度接近CPU核的工作速度，容量较小，一般分成I-cache和D-cache两块，分别存储指令和数据；L2 cache比L1更远，速度慢一些，但是容量更大，不分I-cache和D-cache；L3更慢、更大，现在流行多核处理器，L3一般由多个处理器核共享，而L1、L2是单核私有的。

- 那么我们就可以简单地描述整个CPU处理的流程和每个层级的特点
  - 寄存器：寄存器是CPU内部的高速存储单元，用于存储当前正在执行的指令和数据。寄存器的访问速度非常快，通常在一个时钟周期内完成。
  - L1 cache：L1 cache是CPU内部的一级缓存，用于存储最近使用的指令和数据。L1 cache的访问速度比寄存器慢，但是比L2 cache快。
  - L2 cache：L2 cache是CPU内部的二级缓存，用于存储最近使用的指令和数据。L2 cache的访问速度比L1 cache慢，但是比L3 cache快。
  - L3 cache：L3 cache是CPU内部的三级缓存，用于存储最近使用的指令和数据。L3 cache的访问速度比L2 cache慢，但是比主存快。
  - 主存：主存是CPU外部的大容量存储设备，用于存储所有的指令和数据。主存的访问速度比L3 cache慢，但是比磁盘快。
  - 磁盘：磁盘是CPU外部的大容量存储设备，用于存储所有的指令和数据。磁盘的访问速度比主存慢。

> 我们可以看到，不同的内存层级之间的访问速度是不同的，而访问速度越快的内存层级，成本也越高。所以我们在设计内存层级时，需要平衡访问速度和成本。

### 对于 CPU 来说计算一个 a[0:31] = b[0:31] + d[0:31] 的过程

- 首先解释一下题目： a[0:31] 是 Python 中列表/字符串/元组的语法，表示从序列 a 中取从索引 0 到索引 30 的子序列（不包括索引 31）。

- 接下来，我详细阅读了 [cache高速缓存](https://zhuanlan.zhihu.com/p/482651908) 2 cache的组成和 3 cache的写入方式 总结了CPU的整个计算推理过程，具体名词请再看此知乎文档

> cache容量较小，所以数据需要按照一定的规则从主存映射到cache。一般把主存和cache分割成一定大小的块，这个块在主存中称为data block，在cache中称为cache line。举个例子，块大小为1024个字节，那么data block和cache line都是1024个字节。当把主存和cache分割好之后，我们就可以把data block放到cache line中，而这个“放”的规则一般有三种，分别是“直接映射”、“组相联”和“全相联”。

- 直接映射：每个data block只能放到cache中的一个line中，这个line的选择是根据data block的地址计算得到的,采用“取模”的方式进行一对一映射。
- 组相联：
  为了避免直接映射带来的冲突问题（例如访问 0 号 block 和 8 号 block 时反复覆盖），引入了组相联。组相联的思路是：将 cache 分成多个 set，每个 set 中有若干个 line。一个 data block 可以放到对应 set 中的任意一个 line，而不是固定某一个位置。这样降低了冲突带来的 miss。
  在硬件实现上，组相联通常有 并行 和 串行 两种方式：

  - 并行实现：index 同时送到 tag RAM 和 data RAM，同时比较多个 tag，再通过多路选择器选中数据；速度较快，但功耗较大。

  - 串行实现：先比较 tag，确认命中的 cache line，再从 data RAM 中取出数据；延迟更高，但节省功耗。

- 全相联：
  全相联是极端情况，cache 只有一个 set，所有的 data block 都能放入任意 line。它的好处是几乎没有冲突 miss，但缺点是需要同时比较所有 line 的 tag，硬件代价高、延时大。


####　当 CPU 执行 a[0:31] = b[0:31] + d[0:31] 时，大致流程如下：

- 取数据

  CPU 首先会尝试从 cache 中读取 b[0:31] 和 d[0:31] 所在的 block。

  如果数据在 cache 中（hit），直接读取；如果不在（miss），则需要从主存将对应的 block 调入 cache line。

  主存到 cache 的映射由 cache 的组织方式（直接映射、组相联、全相联）决定。

- 执行运算

  CPU 将取到的数据装载到寄存器中（如寄存器 r1 ← b[i], r2 ← d[i]）。

  执行加法指令（r3 = r1 + r2）。

- 写回数据

  r3 的结果写入 cache 中对应的 a[i]。

  此时涉及到 cache 的 写策略：

  - 如果采用 写回 + 写分配：结果先写入 cache，并将该 cache line 标记为脏，只有当该 line 被替换时才写回主存。

  - 如果采用 写穿 + 写不分配：结果同时写入 cache 和主存，保证一致性，但延时更高。

- “写穿”“写不分配”组合的工作流程
![alt text](images/image.png) 
- 写回/写分配组合工作流程
![alt text](images/image1.png)

最终，a[0:31] 的所有计算结果都存放在 cache 和/或主存中，具体位置依赖于所采用的写入策略。
### 什么是进程和线程，多进程和 (单核/多核) 多线程的区别是什么？各有什么优缺点？

#### 进程（Process）

进程是操作系统进行资源分配和调度的基本单位，是程序在执行过程中的实例。每个进程都有自己独立的内存空间、文件描述符和系统资源。

**进程的主要特点：**
- 拥有独立的地址空间
- 进程间通信需要特殊的机制（如管道、消息队列、共享内存等）
- 进程切换开销较大
- 进程之间相互独立，一个进程的崩溃不会影响其他进程

#### 线程（Thread）

线程是进程内的一个执行单元，是CPU调度的基本单位。一个进程可以包含多个线程，这些线程共享进程的地址空间和系统资源，但有各自独立的程序计数器、寄存器和栈。

**线程的主要特点：**
- 线程共享所属进程的地址空间和资源
- 线程间通信相对简单，可以通过共享内存进行
- 线程切换开销较小
- 同一进程内的线程共享内存，一个线程的错误可能影响整个进程

#### 多进程与多线程的区别

| 特性 | 多进程 | 多线程 |
|------|--------|--------|
| 内存空间 | 独立 | 共享 |
| 通信方式 | 复杂（需要IPC机制） | 简单（共享内存） |
| 切换开销 | 大 | 小 |
| 资源占用 | 多 | 少 |
| 稳定性 | 高（一个进程崩溃不影响其他进程） | 低（一个线程崩溃可能导致整个进程崩溃） |
| 编程复杂度 | 较高 | 较低 |

#### 单核多线程与多核多线程的区别

- **单核多线程**：在单个CPU核心上运行多个线程，线程间通过时间分片的方式交替执行，宏观上并发，微观上串行。主要优势是提高I/O密集型任务的效率，避免CPU空闲。
- **多核多线程**：在多个CPU核心上同时运行多个线程，线程可以真正并行执行。主要优势是提高计算密集型任务的效率，充分利用多核处理器的性能。

#### 优缺点总结

**多进程的优点：**
- 稳定性高
- 可以充分利用多核处理器
- 安全性好，进程间隔离

**多进程的缺点：**
- 资源占用大
- 进程间通信复杂
- 创建和销毁开销大

**多线程的优点：**
- 资源占用小
- 线程间通信简单
- 创建和销毁开销小

**多线程的缺点：**
- 稳定性较低
- 需要处理线程同步问题
- 可能受到GIL（全局解释器锁）的限制（如Python）

### 如何使用 C++ 多线程/多进程加速这一个计算过程

假设我们要加速的计算过程是 `a[0:31] = b[0:31] + d[0:31]`，下面分别介绍如何使用C++的多线程和多进程来加速这个计算。

#### 使用 C++ 多线程加速

C++11引入了标准的线程库`std::thread`，我们可以使用它来创建多线程程序。

```cpp
#include <iostream>
#include <vector>
#include <thread>

// 定义数组大小
const int ARRAY_SIZE = 32;

// 计算函数，处理数组的一部分
void compute_partial(const std::vector<int>& b, const std::vector<int>& d, std::vector<int>& a, int start, int end) {
    for (int i = start; i < end; ++i) {
        a[i] = b[i] + d[i];
    }
}

int main() {
    // 初始化数组
    std::vector<int> b(ARRAY_SIZE, 1);  // 假设b数组的所有元素都是1
    std::vector<int> d(ARRAY_SIZE, 2);  // 假设d数组的所有元素都是2
    std::vector<int> a(ARRAY_SIZE, 0);  // 结果数组，初始化为0
    
    // 确定线程数和每个线程处理的元素数量
    int num_threads = std::thread::hardware_concurrency();  // 获取硬件支持的线程数
    if (num_threads == 0) num_threads = 4;  // 如果无法获取，则默认为4
    
    // 根据数组大小和线程数，调整实际使用的线程数
    int actual_threads = std::min(num_threads, ARRAY_SIZE);
    int elements_per_thread = ARRAY_SIZE / actual_threads;
    
    // 创建线程
    std::vector<std::thread> threads;
    for (int i = 0; i < actual_threads; ++i) {
        int start = i * elements_per_thread;
        int end = (i == actual_threads - 1) ? ARRAY_SIZE : (i + 1) * elements_per_thread;
        threads.emplace_back(compute_partial, std::ref(b), std::ref(d), std::ref(a), start, end);
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    // 输出结果
    std::cout << "Result: ";
    for (int val : a) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### 使用 C++ 多进程加速

在C++中，我们可以使用操作系统提供的API来创建多进程，如Unix/Linux系统的`fork()`函数或Windows系统的`CreateProcess()`函数。此外，C++17引入了`<filesystem>`库，可以更方便地进行文件操作。

以下是使用`fork()`创建多进程的示例：

```cpp
#include <iostream>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <fstream>

// 定义数组大小
const int ARRAY_SIZE = 32;

int main() {
    // 初始化数组
    std::vector<int> b(ARRAY_SIZE, 1);  // 假设b数组的所有元素都是1
    std::vector<int> d(ARRAY_SIZE, 2);  // 假设d数组的所有元素都是2
    std::vector<int> a(ARRAY_SIZE, 0);  // 结果数组，初始化为0
    
    // 确定进程数
    int num_processes = sysconf(_SC_NPROCESSORS_ONLN);  // 获取可用的CPU核心数
    if (num_processes <= 0) num_processes = 4;  // 如果无法获取，则默认为4
    
    // 根据数组大小和进程数，调整实际使用的进程数
    int actual_processes = std::min(num_processes, ARRAY_SIZE);
    int elements_per_process = ARRAY_SIZE / actual_processes;
    
    // 创建进程
    for (int i = 0; i < actual_processes; ++i) {
        pid_t pid = fork();
        
        if (pid == 0) {  // 子进程
            int start = i * elements_per_process;
            int end = (i == actual_processes - 1) ? ARRAY_SIZE : (i + 1) * elements_per_process;
            
            // 计算子进程负责的部分
            for (int j = start; j < end; ++j) {
                a[j] = b[j] + d[j];
            }
            
            // 将结果写入临时文件
            std::string filename = "temp_result_" + std::to_string(i) + ".dat";
            std::ofstream outfile(filename, std::ios::binary);
            outfile.write(reinterpret_cast<const char*>(&start), sizeof(start));
            outfile.write(reinterpret_cast<const char*>(&end), sizeof(end));
            outfile.write(reinterpret_cast<const char*>(&a[start]), (end - start) * sizeof(int));
            outfile.close();
            
            exit(0);  // 子进程退出
        }
    }
    
    // 等待所有子进程完成
    for (int i = 0; i < actual_processes; ++i) {
        wait(nullptr);
    }
    
    // 从临时文件中读取结果并合并
    for (int i = 0; i < actual_processes; ++i) {
        std::string filename = "temp_result_" + std::to_string(i) + ".dat";
        std::ifstream infile(filename, std::ios::binary);
        
        int start, end;
        infile.read(reinterpret_cast<char*>(&start), sizeof(start));
        infile.read(reinterpret_cast<char*>(&end), sizeof(end));
        infile.read(reinterpret_cast<char*>(&a[start]), (end - start) * sizeof(int));
        
        infile.close();
        remove(filename.c_str());  // 删除临时文件
    }
    
    // 输出结果
    std::cout << "Result: ";
    for (int val : a) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### 多线程 vs 多进程的选择

在实际应用中，我们需要根据具体情况选择使用多线程还是多进程：

- **CPU密集型任务**：如果任务主要是计算，且可以很好地并行化，两种方式都可以。在多核处理器上，多进程可能略占优势，因为避免了线程同步的开销。
- **I/O密集型任务**：如果任务涉及大量的I/O操作（如文件读写、网络通信等），多线程通常是更好的选择，因为线程切换的开销比进程小。
- **内存占用**：如果任务需要大量内存，多进程可能会占用更多的内存，因为每个进程都有自己独立的地址空间。
- **编程复杂度**：多线程的编程通常比多进程简单，特别是在需要共享数据时。
- **稳定性要求**：如果对稳定性要求很高，多进程可能更合适，因为一个进程的崩溃不会影响其他进程。
