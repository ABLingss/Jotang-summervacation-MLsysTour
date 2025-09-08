#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
内存访问模式分析和优化工具

此脚本实现了以下功能：
1. 分析不同内存访问模式对性能的影响
2. 实现多种内存优化技术
3. 比较顺序访问与随机访问的性能差异
4. 展示数据局部性优化效果
5. 提供缓存友好的数据结构和算法设计建议

使用方法：
    python memory_access_analysis.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# 配置参数
ARRAY_SIZES = [10**4, 10**5, 10**6, 10**7]  # 测试的数组大小
REPEATS = 5                               # 每个测试重复次数
WARMUP_ITERATIONS = 3                     # 预热迭代次数
BLOCK_SIZES = [32, 64, 128, 256, 512]     # 测试的块大小

# 初始化数据
def initialize_arrays(size, dtype=np.float64):
    """初始化测试用数组"""
    # 创建随机数组
    data = np.random.rand(size).astype(dtype)
    # 创建已排序索引数组（用于顺序访问）
    sequential_indices = np.arange(size).astype(np.int64)
    # 创建随机索引数组（用于随机访问）
    random_indices = np.random.permutation(size).astype(np.int64)
    # 创建部分有序的索引数组（用于测试空间局部性）
    partially_sorted_indices = create_partially_sorted_indices(size)
    
    return data, sequential_indices, random_indices, partially_sorted_indices

# 创建部分有序的索引数组
def create_partially_sorted_indices(size, block_size=1000):
    """创建部分有序的索引数组，模拟不同程度的数据局部性"""
    indices = np.arange(size)
    blocks = [indices[i:i+block_size] for i in range(0, size, block_size)]
    np.random.shuffle(blocks)
    return np.concatenate(blocks)

# 测量函数执行时间
def measure_time(func, *args):
    """测量函数执行时间"""
    # 预热
    for _ in range(WARMUP_ITERATIONS):
        func(*args)
    
    # 多次测量取平均
    times = []
    for _ in range(REPEATS):
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    # 返回平均执行时间和结果
    avg_time = sum(times) / len(times)
    return avg_time, result

# 1. 不同内存访问模式的性能比较

# 顺序访问数组
def sequential_access(data):
    """顺序访问数组元素"""
    sum_val = 0.0
    for i in range(len(data)):
        sum_val += data[i]
    return sum_val

# 随机访问数组
def random_access(data, indices):
    """随机访问数组元素"""
    sum_val = 0.0
    for i in range(len(indices)):
        sum_val += data[indices[i]]
    return sum_val

# 跨步访问数组
def strided_access(data, stride=1):
    """以特定步长访问数组元素"""
    sum_val = 0.0
    for i in range(0, len(data), stride):
        sum_val += data[i]
    return sum_val

# 2. 缓存优化技术演示

# 普通的矩阵按行访问
def matrix_row_access(matrix):
    """按行访问矩阵元素"""
    rows, cols = matrix.shape
    sum_val = 0.0
    for i in range(rows):
        for j in range(cols):
            sum_val += matrix[i, j]
    return sum_val

# 普通的矩阵按列访问
def matrix_col_access(matrix):
    """按列访问矩阵元素"""
    rows, cols = matrix.shape
    sum_val = 0.0
    for j in range(cols):
        for i in range(rows):
            sum_val += matrix[i, j]
    return sum_val

# 使用循环分块优化的矩阵访问
def matrix_blocked_access(matrix, block_size=64):
    """使用循环分块技术优化矩阵访问"""
    rows, cols = matrix.shape
    sum_val = 0.0
    
    # 按块处理矩阵
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # 计算当前块的边界
            i_end = min(i + block_size, rows)
            j_end = min(j + block_size, cols)
            
            # 块内访问（按行优先）
            for bi in range(i, i_end):
                for bj in range(j, j_end):
                    sum_val += matrix[bi, bj]
    
    return sum_val

# 3. 数据局部性优化

# 计算一维卷积（朴素实现）
def convolution_naive(data, kernel):
    """朴素的一维卷积实现"""
    result = np.zeros(len(data) - len(kernel) + 1)
    for i in range(len(result)):
        for j in range(len(kernel)):
            result[i] += data[i + j] * kernel[j]
    return result

# 计算一维卷积（优化数据局部性）
def convolution_optimized(data, kernel):
    """优化数据局部性的一维卷积实现"""
    result = np.zeros(len(data) - len(kernel) + 1)
    # 反转卷积核以提高缓存利用率（在实际应用中可能不需要）
    kernel_reversed = kernel[::-1]
    
    for i in range(len(result)):
        sum_val = 0.0
        for j in range(len(kernel_reversed)):
            sum_val += data[i + j] * kernel_reversed[j]
        result[i] = sum_val
    
    return result

# 4. 内存重用策略

# 无内存重用的向量运算
def vector_operation_no_reuse(a, b):
    """无内存重用的向量运算"""
    c = np.zeros_like(a)
    d = np.zeros_like(a)
    e = np.zeros_like(a)
    
    # 分别计算，没有利用中间结果
    for i in range(len(a)):
        c[i] = a[i] + b[i]
    for i in range(len(a)):
        d[i] = a[i] * b[i]
    for i in range(len(a)):
        e[i] = c[i] + d[i]
    
    return e

# 有内存重用的向量运算
def vector_operation_with_reuse(a, b):
    """有内存重用的向量运算"""
    e = np.zeros_like(a)
    
    # 一次遍历，重用中间结果，减少内存访问
    for i in range(len(a)):
        c_i = a[i] + b[i]
        d_i = a[i] * b[i]
        e[i] = c_i + d_i
    
    return e

# 运行内存访问模式分析
def run_memory_access_analysis():
    """运行内存访问模式分析"""
    print("===== Memory Access Pattern Analysis =====")
    
    # 1. 不同内存访问模式的性能比较
    print("\n=== 1. Sequential vs Random Access Performance ===")
    
    sequential_results = []
    random_results = []
    partial_sorted_results = []
    
    for size in ARRAY_SIZES:
        print(f"\nTesting array size: {size:,}")
        
        # 初始化数据
        data, seq_indices, rand_indices, part_sorted_indices = initialize_arrays(size)
        
        # 测试顺序访问
        seq_time, seq_sum = measure_time(sequential_access, data)
        print(f"Sequential access time: {seq_time:.2f} ms")
        
        # 测试随机访问
        rand_time, rand_sum = measure_time(random_access, data, rand_indices)
        print(f"Random access time: {rand_time:.2f} ms")
        print(f"Sequential vs Random speedup: {rand_time/seq_time:.2f}x")
        
        # 测试部分有序访问
        part_sorted_time, part_sorted_sum = measure_time(random_access, data, part_sorted_indices)
        print(f"Partially sorted access time: {part_sorted_time:.2f} ms")
        print(f"Sequential vs Partially sorted speedup: {part_sorted_time/seq_time:.2f}x")
        
        # 计算带宽（MB/s）
        data_size_mb = size * data.itemsize / (1024 * 1024)
        seq_bandwidth = data_size_mb / (seq_time / 1000)  # MB/s
        rand_bandwidth = data_size_mb / (rand_time / 1000)  # MB/s
        part_sorted_bandwidth = data_size_mb / (part_sorted_time / 1000)  # MB/s
        
        print(f"Sequential bandwidth: {seq_bandwidth:.2f} MB/s")
        print(f"Random bandwidth: {rand_bandwidth:.2f} MB/s")
        print(f"Partially sorted bandwidth: {part_sorted_bandwidth:.2f} MB/s")
        
        # 记录结果
        sequential_results.append((size, seq_time, seq_bandwidth))
        random_results.append((size, rand_time, rand_bandwidth))
        partial_sorted_results.append((size, part_sorted_time, part_sorted_bandwidth))
    
    # 2. 步长对内存访问性能的影响
    print("\n=== 2. Effect of Stride on Memory Access Performance ===")
    
    stride_results = []
    base_size = 10**7  # 使用固定大小的数组测试步长
    
    # 初始化数据
    data, _, _, _ = initialize_arrays(base_size)
    
    # 测试不同步长
    strides = [1, 2, 4, 8, 16, 32, 64, 128]
    
    for stride in strides:
        stride_time, _ = measure_time(strided_access, data, stride)
        
        # 计算实际访问的元素数量和带宽
        elements_accessed = len(data) // stride
        data_size_mb = elements_accessed * data.itemsize / (1024 * 1024)
        bandwidth = data_size_mb / (stride_time / 1000)  # MB/s
        
        print(f"Stride {stride}: time = {stride_time:.2f} ms, bandwidth = {bandwidth:.2f} MB/s")
        stride_results.append((stride, stride_time, bandwidth))
    
    # 3. 矩阵访问模式（行优先vs列优先）
    print("\n=== 3. Matrix Access Patterns (Row-major vs Column-major) ===")
    
    matrix_sizes = [1000, 2000, 3000, 4000]
    row_access_results = []
    col_access_results = []
    blocked_access_results = []
    
    for size in matrix_sizes:
        print(f"\nTesting matrix size: {size}x{size}")
        
        # 创建矩阵
        matrix = np.random.rand(size, size)
        
        # 测试行优先访问
        row_time, _ = measure_time(matrix_row_access, matrix)
        print(f"Row-major access time: {row_time:.2f} ms")
        
        # 测试列优先访问
        col_time, _ = measure_time(matrix_col_access, matrix)
        print(f"Column-major access time: {col_time:.2f} ms")
        print(f"Row vs Column speedup: {col_time/row_time:.2f}x")
        
        # 测试块访问（使用默认块大小）
        blocked_time, _ = measure_time(matrix_blocked_access, matrix)
        print(f"Blocked access time: {blocked_time:.2f} ms")
        
        # 记录结果
        row_access_results.append((size, row_time))
        col_access_results.append((size, col_time))
        blocked_access_results.append((size, blocked_time))
    
    # 4. 块大小对性能的影响
    print("\n=== 4. Effect of Block Size on Cache Performance ===")
    
    block_size_results = []
    fixed_matrix_size = 4000  # 使用固定大小的矩阵测试块大小
    
    # 创建矩阵
    matrix = np.random.rand(fixed_matrix_size, fixed_matrix_size)
    
    for block_size in BLOCK_SIZES:
        blocked_time, _ = measure_time(matrix_blocked_access, matrix, block_size)
        print(f"Block size {block_size}: time = {blocked_time:.2f} ms")
        block_size_results.append((block_size, blocked_time))
    
    # 5. 数据局部性优化
    print("\n=== 5. Data Locality Optimization ===")
    
    # 创建测试数据和卷积核
    signal_size = 10**6
    kernel_size = 100
    
    signal = np.random.rand(signal_size)
    kernel = np.random.rand(kernel_size)
    
    # 测试朴素卷积
    naive_time, naive_result = measure_time(convolution_naive, signal, kernel)
    print(f"Naive convolution time: {naive_time:.2f} ms")
    
    # 测试优化后的卷积
    optimized_time, optimized_result = measure_time(convolution_optimized, signal, kernel)
    print(f"Optimized convolution time: {optimized_time:.2f} ms")
    print(f"Optimization speedup: {naive_time/optimized_time:.2f}x")
    
    # 验证结果正确性
    results_match = np.allclose(naive_result, optimized_result)
    print(f"Results match: {results_match}")
    
    # 6. 内存重用策略
    print("\n=== 6. Memory Reuse Strategies ===")
    
    vector_size = 10**7
    a = np.random.rand(vector_size)
    b = np.random.rand(vector_size)
    
    # 测试无内存重用的实现
    no_reuse_time, no_reuse_result = measure_time(vector_operation_no_reuse, a, b)
    print(f"No memory reuse time: {no_reuse_time:.2f} ms")
    
    # 测试有内存重用的实现
    with_reuse_time, with_reuse_result = measure_time(vector_operation_with_reuse, a, b)
    print(f"With memory reuse time: {with_reuse_time:.2f} ms")
    print(f"Memory reuse speedup: {no_reuse_time/with_reuse_time:.2f}x")
    
    # 验证结果正确性
    reuse_results_match = np.allclose(no_reuse_result, with_reuse_result)
    print(f"Results match: {reuse_results_match}")
    
    # 生成性能报告
    generate_report(
        sequential_results, random_results, partial_sorted_results,
        stride_results,
        row_access_results, col_access_results, blocked_access_results,
        block_size_results,
        naive_time, optimized_time,
        no_reuse_time, with_reuse_time
    )
    
    # 可视化性能结果
    visualize_results(
        sequential_results, random_results, partial_sorted_results,
        stride_results,
        row_access_results, col_access_results, blocked_access_results,
        block_size_results
    )

# 生成内存访问模式分析报告
def generate_report(
    sequential_results, random_results, partial_sorted_results,
    stride_results,
    row_access_results, col_access_results, blocked_access_results,
    block_size_results,
    naive_conv_time, optimized_conv_time,
    no_reuse_time, with_reuse_time
):
    """生成内存访问模式分析报告"""
    print("\n===== Memory Access Pattern Analysis Report =====")
    
    # 1. 顺序 vs 随机访问性能比较
    print("\n=== 1. Sequential vs Random Access Performance ===")
    headers = ["Array Size", "Sequential (ms)", "Random (ms)", "Partial Sorted (ms)", "Seq/Random Ratio", "Seq/Part Sorted Ratio"]
    
    rows = []
    for i in range(len(sequential_results)):
        size, seq_time, _ = sequential_results[i]
        _, rand_time, _ = random_results[i]
        _, part_sorted_time, _ = partial_sorted_results[i]
        
        rows.append([
            f"{size:,}",
            f"{seq_time:.2f}",
            f"{rand_time:.2f}",
            f"{part_sorted_time:.2f}",
            f"{rand_time/seq_time:.2f}x",
            f"{part_sorted_time/seq_time:.2f}x"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 2. 步长对内存访问性能的影响
    print("\n=== 2. Effect of Stride on Memory Access Performance ===")
    headers = ["Stride", "Time (ms)", "Bandwidth (MB/s)", "Relative Performance"]
    
    rows = []
    base_bandwidth = stride_results[0][2]  # 步长为1时的带宽
    
    for stride, time_val, bandwidth in stride_results:
        rel_performance = bandwidth / base_bandwidth * 100  # 相对于步长为1的性能百分比
        rows.append([
            stride,
            f"{time_val:.2f}",
            f"{bandwidth:.2f}",
            f"{rel_performance:.1f}%"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 3. 矩阵访问模式性能比较
    print("\n=== 3. Matrix Access Patterns Performance ===")
    headers = ["Matrix Size", "Row-major (ms)", "Column-major (ms)", "Blocked (ms)", "Row/Col Ratio", "Row/Block Ratio"]
    
    rows = []
    for i in range(len(row_access_results)):
        size, row_time = row_access_results[i]
        _, col_time = col_access_results[i]
        _, blocked_time = blocked_access_results[i]
        
        rows.append([
            f"{size}x{size}",
            f"{row_time:.2f}",
            f"{col_time:.2f}",
            f"{blocked_time:.2f}",
            f"{col_time/row_time:.2f}x",
            f"{blocked_time/row_time:.2f}x"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 4. 块大小对性能的影响
    print("\n=== 4. Effect of Block Size on Performance ===")
    headers = ["Block Size", "Time (ms)", "Relative Performance"]
    
    rows = []
    # 找到最佳性能（最小时间）
    min_time = min([t for _, t in block_size_results])
    
    for block_size, time_val in block_size_results:
        rel_performance = min_time / time_val * 100  # 相对于最佳性能的百分比
        rows.append([
            block_size,
            f"{time_val:.2f}",
            f"{rel_performance:.1f}%"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 5. 数据局部性优化结果
    print("\n=== 5. Data Locality Optimization Results ===")
    print(f"Naive convolution time: {naive_conv_time:.2f} ms")
    print(f"Optimized convolution time: {optimized_conv_time:.2f} ms")
    print(f"Optimization speedup: {naive_conv_time/optimized_conv_time:.2f}x")
    
    # 6. 内存重用策略结果
    print("\n=== 6. Memory Reuse Strategy Results ===")
    print(f"No memory reuse time: {no_reuse_time:.2f} ms")
    print(f"With memory reuse time: {with_reuse_time:.2f} ms")
    print(f"Memory reuse speedup: {no_reuse_time/with_reuse_time:.2f}x")
    
    # 总结内存优化建议
    print("\n===== Memory Optimization Recommendations =====")
    print("1. Memory Access Patterns:")
    print("   - Favor sequential access over random access whenever possible")
    print("   - Arrange data structures to optimize for the access patterns of your algorithms")
    print("   - Use row-major order (C-style) for row-wise operations and column-major order (Fortran-style) for column-wise operations")
    print("")
    print("2. Cache Optimization:")
    print("   - Implement loop blocking (tiling) to improve cache hit rates")
    print("   - Choose block sizes that fit well with your processor's cache hierarchy")
    print("   - Minimize cache line evictions by reusing data in the same cache line")
    print("")
    print("3. Data Locality:")
    print("   - Exploit spatial locality by accessing data that is close together in memory")
    print("   - Exploit temporal locality by reusing data that was recently accessed")
    print("   - Consider data transformation techniques to improve locality")
    print("")
    print("4. Memory Reuse:")
    print("   - Reuse intermediate results to minimize memory traffic")
    print("   - Use scratch space instead of creating new arrays when possible")
    print("   - Minimize the number of passes over large datasets")
    print("")
    print("5. Data Structure Design:")
    print("   - Align data structures to cache lines for better performance")
    print("   - Consider using SoA (Structure of Arrays) vs AoS (Array of Structures) based on access patterns")
    print("   - Use appropriate data types to reduce memory footprint")
    print("")
    print("6. Compiler Optimizations:")
    print("   - Enable compiler optimizations (-O3, -march=native)")
    print("   - Use compiler directives to guide memory access optimizations")
    print("   - Consider profile-guided optimization (PGO)")

# 可视化内存访问模式分析结果
def visualize_results(
    sequential_results, random_results, partial_sorted_results,
    stride_results,
    row_access_results, col_access_results, blocked_access_results,
    block_size_results
):
    """可视化内存访问模式分析结果"""
    try:
        # 创建结果目录
        results_dir = "memory_access_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. 顺序 vs 随机访问性能对比
        sizes = [f"{r[0]:,}" for r in sequential_results]
        seq_times = [r[1] for r in sequential_results]
        rand_times = [r[1] for r in random_results]
        part_sorted_times = [r[1] for r in partial_sorted_results]
        
        plt.figure(figsize=(12, 8))
        plt.plot(sizes, seq_times, marker='o', label='Sequential')
        plt.plot(sizes, rand_times, marker='s', label='Random')
        plt.plot(sizes, part_sorted_times, marker='^', label='Partially Sorted')
        
        plt.xlabel('Array Size')
        plt.ylabel('Execution Time (ms)')
        plt.title('Sequential vs Random vs Partially Sorted Access Performance')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "access_pattern_comparison.png"))
        print("\nAccess pattern comparison visualization saved as 'memory_access_results/access_pattern_comparison.png'")
        
        # 2. 带宽对比
        seq_bandwidths = [r[2] for r in sequential_results]
        rand_bandwidths = [r[2] for r in random_results]
        part_sorted_bandwidths = [r[2] for r in partial_sorted_results]
        
        plt.figure(figsize=(12, 8))
        plt.plot(sizes, seq_bandwidths, marker='o', label='Sequential')
        plt.plot(sizes, rand_bandwidths, marker='s', label='Random')
        plt.plot(sizes, part_sorted_bandwidths, marker='^', label='Partially Sorted')
        
        plt.xlabel('Array Size')
        plt.ylabel('Bandwidth (MB/s)')
        plt.title('Memory Bandwidth Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "bandwidth_comparison.png"))
        print("Bandwidth comparison visualization saved as 'memory_access_results/bandwidth_comparison.png'")
        
        # 3. 步长对性能的影响
        strides = [r[0] for r in stride_results]
        stride_bandwidths = [r[2] for r in stride_results]
        
        plt.figure(figsize=(12, 8))
        plt.plot(strides, stride_bandwidths, marker='o', linestyle='-')
        
        plt.xlabel('Stride')
        plt.ylabel('Bandwidth (MB/s)')
        plt.title('Effect of Stride on Memory Bandwidth')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "stride_effect.png"))
        print("Stride effect visualization saved as 'memory_access_results/stride_effect.png'")
        
        # 4. 矩阵访问模式对比
        matrix_sizes = [f"{r[0]}x{r[0]}" for r in row_access_results]
        row_times = [r[1] for r in row_access_results]
        col_times = [r[1] for r in col_access_results]
        blocked_times = [r[1] for r in blocked_access_results]
        
        plt.figure(figsize=(12, 8))
        plt.plot(matrix_sizes, row_times, marker='o', label='Row-major')
        plt.plot(matrix_sizes, col_times, marker='s', label='Column-major')
        plt.plot(matrix_sizes, blocked_times, marker='^', label='Blocked')
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Execution Time (ms)')
        plt.title('Matrix Access Pattern Performance')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "matrix_access_comparison.png"))
        print("Matrix access comparison visualization saved as 'memory_access_results/matrix_access_comparison.png'")
        
        # 5. 块大小对性能的影响
        block_sizes = [r[0] for r in block_size_results]
        block_times = [r[1] for r in block_size_results]
        
        plt.figure(figsize=(12, 8))
        plt.plot(block_sizes, block_times, marker='o', linestyle='-')
        
        plt.xlabel('Block Size')
        plt.ylabel('Execution Time (ms)')
        plt.title('Effect of Block Size on Performance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "block_size_effect.png"))
        print("Block size effect visualization saved as 'memory_access_results/block_size_effect.png'")
        
    except Exception as e:
        print(f"Failed to generate visualizations: {str(e)}")

# 主函数
def main():
    """主函数，协调整个内存访问模式分析流程"""
    # 运行内存访问模式分析
    run_memory_access_analysis()
    
    print("\n===== Memory access pattern analysis completed =====")

if __name__ == "__main__":
    main()