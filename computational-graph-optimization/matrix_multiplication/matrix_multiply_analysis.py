#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
矩阵乘法理论性能分析工具

此脚本实现了以下功能：
1. 分析矩阵乘法在单核心CPU上的理论性能
2. 实现多种矩阵乘法优化算法
3. 计算理论性能极限和实际性能对比
4. 提供性能瓶颈分析

使用方法：
    python matrix_multiply_analysis.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# 配置参数
MATRIX_SIZES = [64, 128, 256, 512, 1024]  # 测试的矩阵大小
REPEATS = 5                               # 每个测试重复次数
WARMUP_ITERATIONS = 3                     # 预热迭代次数

# CPU特性参数（可以根据实际CPU修改）
# 这里使用典型的现代CPU参数作为示例
CPU_FREQUENCY_GHZ = 3.5                   # CPU频率（GHz）
L1_CACHE_SIZE_KB = 32                     # L1缓存大小（KB）
L2_CACHE_SIZE_KB = 256                    # L2缓存大小（KB）
L3_CACHE_SIZE_KB = 8192                   # L3缓存大小（KB）
MAX_FLOPS_PER_CYCLE = 16                  # 每周期最大浮点操作数（假设支持AVX2 256位指令）

# 初始化矩阵数据
def initialize_matrices(m, n, k):
    """初始化三个矩阵：A(m×k), B(k×n), C(m×n)"""
    A = np.random.rand(m, k).astype(np.float64)
    B = np.random.rand(k, n).astype(np.float64)
    C = np.zeros((m, n), dtype=np.float64)
    return A, B, C

# 朴素的矩阵乘法实现
def matrix_multiply_naive(A, B, C, m, n, k):
    """朴素的三重循环矩阵乘法实现"""
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i, j] += A[i, p] * B[p, j]

# 优化1: 矩阵转置，提高缓存利用率
def matrix_multiply_transposed(A, B, C, m, n, k):
    """通过矩阵转置优化缓存利用率的矩阵乘法"""
    # 转置矩阵B以提高缓存局部性
    B_transposed = B.T.copy()
    
    for i in range(m):
        for j in range(n):
            sum_val = 0.0
            for p in range(k):
                sum_val += A[i, p] * B_transposed[j, p]
            C[i, j] = sum_val

# 优化2: 循环分块（Blocking/Tiling）
def matrix_multiply_blocked(A, B, C, m, n, k, block_size=64):
    """使用循环分块技术优化的矩阵乘法"""
    # 转置矩阵B
    B_transposed = B.T.copy()
    
    # 按块处理矩阵
    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            for p in range(0, k, block_size):
                # 计算当前块的边界
                i_end = min(i + block_size, m)
                j_end = min(j + block_size, n)
                p_end = min(p + block_size, k)
                
                # 计算块内的矩阵乘法
                for ii in range(i, i_end):
                    for jj in range(j, j_end):
                        sum_val = 0.0
                        for pp in range(p, p_end):
                            sum_val += A[ii, pp] * B_transposed[jj, pp]
                        C[ii, jj] += sum_val

# 优化3: 使用NumPy内置函数（高度优化）
def matrix_multiply_numpy(A, B, C, m, n, k):
    """使用NumPy内置的矩阵乘法（高度优化）"""
    np.matmul(A, B, out=C)

# 测量函数执行时间
def measure_time(func, *args):
    """测量函数执行时间"""
    # 预热
    for _ in range(WARMUP_ITERATIONS):
        func(*args)
    
    # 多次测量取平均
    times = []
    for _ in range(REPEATS):
        # 重置结果矩阵
        if len(args) >= 3 and isinstance(args[2], np.ndarray):
            args[2].fill(0.0)
        
        # 测量执行时间
        start_time = time.time()
        func(*args)
        end_time = time.time()
        
        # 转换为毫秒
        times.append((end_time - start_time) * 1000)
    
    # 返回平均执行时间（毫秒）
    return sum(times) / len(times)

# 计算理论性能极限
def calculate_theoretical_flops(m, n, k):
    """计算矩阵乘法的理论浮点操作数"""
    # 矩阵乘法的浮点操作数：2*m*n*k （m*n个点，每个点需要k次乘法和k次加法）
    flops = 2 * m * n * k
    return flops

# 计算CPU理论最大性能
def calculate_cpu_max_performance():
    """计算CPU理论最大计算性能（FLOPS）"""
    # 计算CPU理论最大FLOPS
    # CPU频率(GHz) * 每周期浮点操作数
    max_flops = CPU_FREQUENCY_GHZ * 1e9 * MAX_FLOPS_PER_CYCLE
    return max_flops

# 计算内存带宽限制
def calculate_memory_bandwidth_limit(A, B, C):
    """计算受内存带宽限制的理论性能"""
    # 计算数据量（字节）
    # A: m*k*8字节, B: k*n*8字节, C: m*n*8字节
    data_size_bytes = (A.size + B.size + C.size) * A.itemsize
    
    # 假设内存带宽（这里使用典型值，实际应根据硬件测试）
    # 例如，DDR4内存大约有20-30 GB/s的带宽
    memory_bandwidth_gbs = 25.0  # GB/s
    
    # 计算受内存带宽限制的最小执行时间
    min_time_seconds = data_size_bytes / (memory_bandwidth_gbs * 1e9)
    
    return min_time_seconds * 1000  # 转换为毫秒

# 分析缓存使用情况
def analyze_cache_usage(m, n, k):
    """分析矩阵乘法的缓存使用情况"""
    # 计算各个矩阵的大小（字节）
    matrix_size_bytes = lambda rows, cols: rows * cols * 8  # 8字节/双精度浮点数
    
    A_size = matrix_size_bytes(m, k)
    B_size = matrix_size_bytes(k, n)
    C_size = matrix_size_bytes(m, n)
    
    # 转换为KB以便比较
    A_size_kb = A_size / 1024
    B_size_kb = B_size / 1024
    C_size_kb = C_size / 1024
    
    # 计算工作集大小（假设同时使用A和B的一部分以及C）
    # 这是一个简化的估计
    working_set_size_kb = (A_size_kb + B_size_kb + C_size_kb) / 2
    
    # 评估缓存适配情况
    cache_fit = {
        'l1_fit': working_set_size_kb <= L1_CACHE_SIZE_KB,
        'l2_fit': working_set_size_kb <= L2_CACHE_SIZE_KB,
        'l3_fit': working_set_size_kb <= L3_CACHE_SIZE_KB
    }
    
    return {
        'A_size_kb': A_size_kb,
        'B_size_kb': B_size_kb,
        'C_size_kb': C_size_kb,
        'working_set_size_kb': working_set_size_kb,
        'cache_fit': cache_fit
    }

# 运行性能分析
def run_performance_analysis():
    """运行矩阵乘法性能分析"""
    print("===== Matrix Multiplication Performance Analysis =====")
    
    # 显示CPU理论参数
    print(f"CPU Theoretical Parameters:")
    print(f"- Frequency: {CPU_FREQUENCY_GHZ} GHz")
    print(f"- L1 Cache: {L1_CACHE_SIZE_KB} KB")
    print(f"- L2 Cache: {L2_CACHE_SIZE_KB} KB")
    print(f"- L3 Cache: {L3_CACHE_SIZE_KB} KB")
    print(f"- Max FLOPs per cycle: {MAX_FLOPS_PER_CYCLE}")
    print(f"- Theoretical peak performance: {calculate_cpu_max_performance()/1e9:.2f} GFLOPS")
    
    # 存储所有结果
    all_results = []
    
    # 遍历不同的矩阵大小
    for size in MATRIX_SIZES:
        m, n, k = size, size, size  # 使用方阵进行测试
        
        print(f"\nTesting matrix size: {m}x{k}x{n}")
        
        # 初始化矩阵
        A, B, C = initialize_matrices(m, n, k)
        
        # 计算理论浮点操作数
        theoretical_flops = calculate_theoretical_flops(m, n, k)
        print(f"Theoretical FLOPs: {theoretical_flops/1e6:.2f} MFLOPs")
        
        # 分析缓存使用情况
        cache_analysis = analyze_cache_usage(m, n, k)
        print(f"Cache analysis:")
        print(f"- A size: {cache_analysis['A_size_kb']:.2f} KB")
        print(f"- B size: {cache_analysis['B_size_kb']:.2f} KB")
        print(f"- C size: {cache_analysis['C_size_kb']:.2f} KB")
        print(f"- Working set size: {cache_analysis['working_set_size_kb']:.2f} KB")
        print(f"- Fits in L1 cache: {cache_analysis['cache_fit']['l1_fit']}")
        print(f"- Fits in L2 cache: {cache_analysis['cache_fit']['l2_fit']}")
        print(f"- Fits in L3 cache: {cache_analysis['cache_fit']['l3_fit']}")
        
        # 计算内存带宽限制下的最小时间
        memory_limited_time = calculate_memory_bandwidth_limit(A, B, C)
        print(f"Memory bandwidth limited time: {memory_limited_time:.2f} ms")
        
        # 计算计算限制下的最小时间
        cpu_max_flops = calculate_cpu_max_performance()
        compute_limited_time = (theoretical_flops / cpu_max_flops) * 1000  # 转换为毫秒
        print(f"Compute limited time: {compute_limited_time:.2f} ms")
        
        # 测量朴素实现的性能
        print("Testing naive implementation...")
        naive_time = measure_time(matrix_multiply_naive, A, B, C, m, n, k)
        naive_gflops = (theoretical_flops / 1e9) / (naive_time / 1000)  # GFLOPS
        print(f"Naive implementation time: {naive_time:.2f} ms ({naive_gflops:.2f} GFLOPS)")
        
        # 测量转置优化实现的性能
        print("Testing transposed implementation...")
        transposed_time = measure_time(matrix_multiply_transposed, A, B, C, m, n, k)
        transposed_gflops = (theoretical_flops / 1e9) / (transposed_time / 1000)  # GFLOPS
        print(f"Transposed implementation time: {transposed_time:.2f} ms ({transposed_gflops:.2f} GFLOPS)")
        
        # 测量分块优化实现的性能
        # 根据缓存大小选择合适的块大小
        if cache_analysis['cache_fit']['l1_fit']:
            block_size = 64
        elif cache_analysis['cache_fit']['l2_fit']:
            block_size = 128
        else:
            block_size = 256
        
        print(f"Testing blocked implementation with block size {block_size}...")
        blocked_time = measure_time(matrix_multiply_blocked, A, B, C, m, n, k, block_size)
        blocked_gflops = (theoretical_flops / 1e9) / (blocked_time / 1000)  # GFLOPS
        print(f"Blocked implementation time: {blocked_time:.2f} ms ({blocked_gflops:.2f} GFLOPS)")
        
        # 测量NumPy实现的性能
        print("Testing NumPy implementation...")
        numpy_time = measure_time(matrix_multiply_numpy, A, B, C, m, n, k)
        numpy_gflops = (theoretical_flops / 1e9) / (numpy_time / 1000)  # GFLOPS
        print(f"NumPy implementation time: {numpy_time:.2f} ms ({numpy_gflops:.2f} GFLOPS)")
        
        # 计算性能相对于理论极限的百分比
        # 实际性能受限于计算限制和内存带宽限制中的较小值
        theoretical_limit_time = min(compute_limited_time, memory_limited_time)
        
        # 记录结果
        all_results.append({
            'size': size,
            'theoretical_flops': theoretical_flops,
            'theoretical_limit_time': theoretical_limit_time,
            'memory_limited_time': memory_limited_time,
            'compute_limited_time': compute_limited_time,
            'naive_time': naive_time,
            'naive_gflops': naive_gflops,
            'transposed_time': transposed_time,
            'transposed_gflops': transposed_gflops,
            'blocked_time': blocked_time,
            'blocked_gflops': blocked_gflops,
            'numpy_time': numpy_time,
            'numpy_gflops': numpy_gflops,
            'cache_analysis': cache_analysis
        })
    
    # 生成性能报告
    generate_report(all_results)
    
    # 可视化性能结果
    visualize_results(all_results)

# 生成性能分析报告
def generate_report(all_results):
    """生成矩阵乘法性能分析报告"""
    print("\n===== Matrix Multiplication Performance Report =====")
    
    # 准备表格数据
    headers = [
        "Matrix Size",
        "Theoretical Limit (ms)",
        "Naive (ms)", "Naive (GFLOPS)",
        "Transposed (ms)", "Transposed (GFLOPS)",
        "Blocked (ms)", "Blocked (GFLOPS)",
        "NumPy (ms)", "NumPy (GFLOPS)"
    ]
    
    rows = []
    
    for result in all_results:
        rows.append([
            f"{result['size']}x{result['size']}x{result['size']}",
            f"{result['theoretical_limit_time']:.2f}",
            f"{result['naive_time']:.2f}",
            f"{result['naive_gflops']:.2f}",
            f"{result['transposed_time']:.2f}",
            f"{result['transposed_gflops']:.2f}",
            f"{result['blocked_time']:.2f}",
            f"{result['blocked_gflops']:.2f}",
            f"{result['numpy_time']:.2f}",
            f"{result['numpy_gflops']:.2f}"
        ])
    
    # 打印表格
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 分析结果
    print("\n===== Performance Analysis =====")
    
    # 找出最大的矩阵
    largest_result = max(all_results, key=lambda x: x['size'])
    
    print("\nKey Observations:")
    print("1. Performance scaling with matrix size:")
    for result in all_results:
        print(f"   - {result['size']}x{result['size']}x{result['size']}: NumPy achieves {result['numpy_gflops']:.2f} GFLOPS")
    
    print("\n2. Optimization effectiveness:")
    print(f"   - Transposed vs Naive: {largest_result['naive_time']/largest_result['transposed_time']:.2f}x speedup")
    print(f"   - Blocked vs Naive: {largest_result['naive_time']/largest_result['blocked_time']:.2f}x speedup")
    print(f"   - NumPy vs Naive: {largest_result['naive_time']/largest_result['numpy_time']:.2f}x speedup")
    
    print("\n3. Performance bottlenecks:")
    for result in all_results:
        # 判断是计算受限还是内存带宽受限
        if result['compute_limited_time'] < result['memory_limited_time']:
            bottleneck = "Compute-bound"
            achieved_percentage = (result['compute_limited_time'] / result['numpy_time']) * 100
        else:
            bottleneck = "Memory-bound"
            achieved_percentage = (result['memory_limited_time'] / result['numpy_time']) * 100
        
        print(f"   - {result['size']}x{result['size']}x{result['size']}: {bottleneck}, achieved {achieved_percentage:.1f}% of theoretical limit")
    
    print("\n4. Cache utilization insights:")
    for result in all_results:
        cache = result['cache_analysis']['cache_fit']
        print(f"   - {result['size']}x{result['size']}x{result['size']}: L1 fit={cache['l1_fit']}, L2 fit={cache['l2_fit']}, L3 fit={cache['l3_fit']}")
    
    # 总结提高矩阵乘法性能的方法
    print("\n===== How to Improve Matrix Multiplication Performance =====")
    print("1. Cache Optimization:")
    print("   - Use loop blocking/tiling to fit data into cache hierarchies")
    print("   - Transpose matrices to improve memory access patterns")
    print("   - Optimize loop order to maximize cache reuse")
    print("")
    print("2. Instruction Level Parallelism:")
    print("   - Use SIMD instructions (AVX, SSE) to process multiple data elements simultaneously")
    print("   - Apply loop unrolling to reduce loop overhead")
    print("   - Use compiler optimizations (-O3, -march=native)")
    print("")
    print("3. Algorithmic Improvements:")
    print("   - For very large matrices, consider Strassen's algorithm for O(n^2.807) complexity")
    print("   - Use blocked algorithms to optimize for cache")
    print("   - Consider using specialized libraries like BLAS, OpenBLAS, or Intel MKL")
    print("")
    print("4. Multi-threading and Parallelization:")
    print("   - Divide work across multiple CPU cores")
    print("   - Use thread pools and task-based parallelism")
    print("   - Consider distributed computing for extremely large matrices")

# 可视化性能结果
def visualize_results(all_results):
    """可视化矩阵乘法性能结果"""
    try:
        # 创建结果目录
        results_dir = "matrix_multiply_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 提取数据
        sizes = [f"{r['size']}x{r['size']}x{r['size']}" for r in all_results]
        naive_times = [r['naive_time'] for r in all_results]
        transposed_times = [r['transposed_time'] for r in all_results]
        blocked_times = [r['blocked_time'] for r in all_results]
        numpy_times = [r['numpy_time'] for r in all_results]
        theoretical_limits = [r['theoretical_limit_time'] for r in all_results]
        
        # 创建执行时间对比图表
        plt.figure(figsize=(12, 8))
        
        plt.plot(sizes, naive_times, marker='o', label='Naive')
        plt.plot(sizes, transposed_times, marker='s', label='Transposed')
        plt.plot(sizes, blocked_times, marker='^', label='Blocked')
        plt.plot(sizes, numpy_times, marker='d', label='NumPy')
        plt.plot(sizes, theoretical_limits, marker='x', linestyle='--', label='Theoretical Limit')
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Execution Time (ms)')
        plt.title('Matrix Multiplication Execution Time Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "matrix_multiply_time_comparison.png"))
        print("\nExecution time visualization saved as 'matrix_multiply_results/matrix_multiply_time_comparison.png'")
        
        # 创建GFLOPS对比图表
        naive_gflops = [r['naive_gflops'] for r in all_results]
        transposed_gflops = [r['transposed_gflops'] for r in all_results]
        blocked_gflops = [r['blocked_gflops'] for r in all_results]
        numpy_gflops = [r['numpy_gflops'] for r in all_results]
        peak_gflops = [calculate_cpu_max_performance()/1e9] * len(all_results)  # 理论峰值GFLOPS
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(sizes, naive_gflops, marker='o', label='Naive')
        plt.plot(sizes, transposed_gflops, marker='s', label='Transposed')
        plt.plot(sizes, blocked_gflops, marker='^', label='Blocked')
        plt.plot(sizes, numpy_gflops, marker='d', label='NumPy')
        plt.plot(sizes, peak_gflops, marker='x', linestyle='--', label='Theoretical Peak')
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Performance (GFLOPS)')
        plt.title('Matrix Multiplication Performance (GFLOPS) Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "matrix_multiply_gflops_comparison.png"))
        print("Performance (GFLOPS) visualization saved as 'matrix_multiply_results/matrix_multiply_gflops_comparison.png'")
        
    except Exception as e:
        print(f"Failed to generate visualizations: {str(e)}")

# 主函数
def main():
    """主函数，协调整个矩阵乘法性能分析流程"""
    # 运行性能分析
    run_performance_analysis()
    
    print("\n===== Matrix multiplication analysis completed =====")

if __name__ == "__main__":
    main()