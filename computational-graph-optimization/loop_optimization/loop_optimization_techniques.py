#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
循环优化技术实现与性能分析

此脚本实现了以下功能：
1. 实现并比较多种循环优化技术
2. 详细解释循环分块(Loop Tiling)的性能提升原理
3. 提供不同优化技术的性能数据对比
4. 可视化各种优化技术的性能差异

使用方法：
    python loop_optimization_techniques.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# 配置参数
IMAGE_SIZE = 4096  # 图像大小
BLUR_SIZE = 4      # 模糊半径
TILE_SIZE = 16     # 分块大小
REPEATS = 5        # 每个测试重复次数
WARMUP_ITERATIONS = 3  # 预热迭代次数

# 初始化图像数据
def initialize_image(size):
    """初始化测试用的图像数据"""
    # 创建随机图像数据
    image = np.random.randint(0, 256, size=(size, size), dtype=np.int32)
    return image

# 原始模糊函数（未优化）
def blur_original(image, result, size, blur_size):
    """原始的Y轴模糊实现（未优化）"""
    for j in range(size):
        for i in range(size):
            sum_val = 0
            count = 0
            for t in range(-blur_size, blur_size + 1):
                if 0 <= i + t < size:
                    sum_val += image[i + t, j]
                    count += 1
            result[i, j] = sum_val // count

# 循环分块优化的模糊函数
def blur_tiled(image, result, size, blur_size, tile_size):
    """使用循环分块优化的Y轴模糊实现"""
    # 外层循环按分块处理
    for j in range(0, size, tile_size):
        # 计算当前块的实际大小
        current_tile_size = min(tile_size, size - j)
        
        for i in range(size):
            for k in range(current_tile_size):
                # 计算实际的列索引
                actual_j = j + k
                
                sum_val = 0
                count = 0
                for t in range(-blur_size, blur_size + 1):
                    if 0 <= i + t < size:
                        sum_val += image[i + t, actual_j]
                        count += 1
                result[i, actual_j] = sum_val // count

# 循环交换优化的模糊函数
def blur_loop_swap(image, result, size, blur_size):
    """使用循环交换优化的Y轴模糊实现"""
    # 交换i和j循环的顺序，提高缓存局部性
    for i in range(size):
        for j in range(size):
            sum_val = 0
            count = 0
            for t in range(-blur_size, blur_size + 1):
                if 0 <= i + t < size:
                    sum_val += image[i + t, j]
                    count += 1
            result[i, j] = sum_val // count

# 循环展开优化的模糊函数
def blur_unrolled(image, result, size, blur_size):
    """使用循环展开优化的Y轴模糊实现"""
    # 针对内层循环进行部分展开
    for j in range(size):
        for i in range(0, size, 4):  # 每次处理4个元素
            # 确保不越界
            for ii in range(min(4, size - i)):
                actual_i = i + ii
                sum_val = 0
                count = 0
                for t in range(-blur_size, blur_size + 1):
                    if 0 <= actual_i + t < size:
                        sum_val += image[actual_i + t, j]
                        count += 1
                result[actual_i, j] = sum_val // count

# 循环分块+展开组合优化的模糊函数
def blur_tiled_unrolled(image, result, size, blur_size, tile_size):
    """结合循环分块和循环展开的优化实现"""
    for j in range(0, size, tile_size):
        current_tile_size = min(tile_size, size - j)
        
        for i in range(0, size, 4):  # 循环展开
            for k in range(current_tile_size):
                actual_j = j + k
                
                # 处理4个连续的i值
                for ii in range(min(4, size - i)):
                    actual_i = i + ii
                    sum_val = 0
                    count = 0
                    for t in range(-blur_size, blur_size + 1):
                        if 0 <= actual_i + t < size:
                            sum_val += image[actual_i + t, actual_j]
                            count += 1
                    result[actual_i, actual_j] = sum_val // count

# 边界条件优化的模糊函数
def blur_boundary_optimized(image, result, size, blur_size):
    """优化边界条件检查的Y轴模糊实现"""
    # 单独处理内部区域（无需边界检查）
    for j in range(size):
        # 内部区域
        for i in range(blur_size, size - blur_size):
            sum_val = 0
            for t in range(-blur_size, blur_size + 1):
                sum_val += image[i + t, j]
            result[i, j] = sum_val // (2 * blur_size + 1)
        
        # 处理左侧边界
        for i in range(0, blur_size):
            sum_val = 0
            count = 0
            for t in range(-blur_size, blur_size + 1):
                if 0 <= i + t < size:
                    sum_val += image[i + t, j]
                    count += 1
            result[i, j] = sum_val // count
        
        # 处理右侧边界
        for i in range(size - blur_size, size):
            sum_val = 0
            count = 0
            for t in range(-blur_size, blur_size + 1):
                if 0 <= i + t < size:
                    sum_val += image[i + t, j]
                    count += 1
            result[i, j] = sum_val // count

# 使用前缀和优化的模糊函数
def blur_prefix_sum(image, result, size, blur_size):
    """使用前缀和技术优化的Y轴模糊实现"""
    # 为每一列计算前缀和
    prefix_sums = np.zeros((size, size), dtype=np.int64)
    
    # 计算每一列的前缀和
    for j in range(size):
        prefix_sums[0, j] = image[0, j]
        for i in range(1, size):
            prefix_sums[i, j] = prefix_sums[i-1, j] + image[i, j]
    
    # 使用前缀和计算模糊值
    window_size = 2 * blur_size + 1
    for j in range(size):
        for i in range(size):
            # 计算窗口的上下边界
            top = max(0, i - blur_size)
            bottom = min(size - 1, i + blur_size)
            
            # 使用前缀和计算窗口内的总和
            if top == 0:
                sum_val = prefix_sums[bottom, j]
            else:
                sum_val = prefix_sums[bottom, j] - prefix_sums[top - 1, j]
            
            # 计算实际的窗口大小
            actual_window_size = bottom - top + 1
            result[i, j] = sum_val // actual_window_size

# 测量函数执行时间
def measure_time(func, *args):
    """测量函数执行时间"""
    # 执行函数并测量时间
    start_time = time.time()
    func(*args)
    end_time = time.time()
    
    # 返回执行时间（毫秒）
    return (end_time - start_time) * 1000

# 验证结果正确性
def verify_results(result1, result2, size):
    """验证两个结果是否一致"""
    # 计算最大差异
    max_diff = np.max(np.abs(result1 - result2))
    # 计算总差异
    total_diff = np.sum(np.abs(result1 - result2))
    
    return {
        'max_diff': max_diff,
        'total_diff': total_diff,
        'is_correct': max_diff < 1e-6  # 设置一个小的阈值
    }

# 运行性能测试
def run_performance_tests():
    """运行各种循环优化技术的性能测试"""
    print("===== Loop Optimization Techniques Performance Test =====")
    
    # 初始化图像数据
    print(f"Initializing image data ({IMAGE_SIZE}x{IMAGE_SIZE})...")
    image = initialize_image(IMAGE_SIZE)
    
    # 创建结果存储数组
    results = {
        'original': np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32),
        'tiled': np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32),
        'loop_swap': np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32),
        'unrolled': np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32),
        'tiled_unrolled': np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32),
        'boundary_optimized': np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32),
        'prefix_sum': np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
    }
    
    # 定义优化技术列表
    optimization_techniques = [
        {
            'name': 'Original',
            'function': blur_original,
            'args': [image, results['original'], IMAGE_SIZE, BLUR_SIZE],
            'key': 'original'
        },
        {
            'name': 'Loop Tiling',
            'function': blur_tiled,
            'args': [image, results['tiled'], IMAGE_SIZE, BLUR_SIZE, TILE_SIZE],
            'key': 'tiled'
        },
        {
            'name': 'Loop Swap',
            'function': blur_loop_swap,
            'args': [image, results['loop_swap'], IMAGE_SIZE, BLUR_SIZE],
            'key': 'loop_swap'
        },
        {
            'name': 'Loop Unrolling',
            'function': blur_unrolled,
            'args': [image, results['unrolled'], IMAGE_SIZE, BLUR_SIZE],
            'key': 'unrolled'
        },
        {
            'name': 'Tiling + Unrolling',
            'function': blur_tiled_unrolled,
            'args': [image, results['tiled_unrolled'], IMAGE_SIZE, BLUR_SIZE, TILE_SIZE],
            'key': 'tiled_unrolled'
        },
        {
            'name': 'Boundary Optimization',
            'function': blur_boundary_optimized,
            'args': [image, results['boundary_optimized'], IMAGE_SIZE, BLUR_SIZE],
            'key': 'boundary_optimized'
        },
        {
            'name': 'Prefix Sum',
            'function': blur_prefix_sum,
            'args': [image, results['prefix_sum'], IMAGE_SIZE, BLUR_SIZE],
            'key': 'prefix_sum'
        }
    ]
    
    # 预热
    print(f"Warming up ({WARMUP_ITERATIONS} iterations)...")
    for _ in range(WARMUP_ITERATIONS):
        for technique in optimization_techniques:
            technique['function'](*technique['args'])
    
    # 运行性能测试
    performance_results = []
    
    print(f"Running performance tests ({REPEATS} repetitions)...")
    for technique in optimization_techniques:
        print(f"Testing {technique['name']}...")
        
        # 多次运行并取平均值
        times = []
        for i in range(REPEATS):
            print(f"  Run {i+1}/{REPEATS}")
            # 每次运行前清空结果数组
            results[technique['key']].fill(0)
            # 测量执行时间
            exec_time = measure_time(technique['function'], *technique['args'])
            times.append(exec_time)
        
        # 计算平均时间
        avg_time = sum(times) / len(times)
        
        # 记录结果
        performance_results.append({
            'name': technique['name'],
            'avg_time': avg_time,
            'key': technique['key']
        })
        
        print(f"  Average time: {avg_time:.4f} ms")
    
    # 验证结果正确性
    print("\n===== Verifying Results =====")
    verification_results = {}
    
    # 以原始实现为基准进行验证
    base_result = results['original']
    
    for technique in performance_results:
        if technique['key'] != 'original':  # 跳过与自己比较
            verify_result = verify_results(base_result, results[technique['key']], IMAGE_SIZE)
            verification_results[technique['name']] = verify_result
            
            print(f"{technique['name']} vs Original:")
            print(f"  Max difference: {verify_result['max_diff']}")
            print(f"  Total difference: {verify_result['total_diff']}")
            print(f"  Results match: {verify_result['is_correct']}")
    
    # 生成性能报告
    generate_report(performance_results, verification_results)
    
    # 可视化性能结果
    visualize_results(performance_results)

# 生成性能报告
def generate_report(performance_results, verification_results):
    """生成性能分析报告"""
    print("\n===== Loop Optimization Performance Report =====")
    
    # 按执行时间排序
    performance_results.sort(key=lambda x: x['avg_time'])
    
    # 准备表格数据
    headers = ["Rank", "Optimization Technique", "Execution Time (ms)", "Speedup vs Original"]
    rows = []
    
    # 找到原始实现的执行时间
    original_time = next((r['avg_time'] for r in performance_results if r['key'] == 'original'), 1)
    
    for i, result in enumerate(performance_results):
        speedup = original_time / result['avg_time'] if result['avg_time'] > 0 else float('inf')
        rows.append([
            i + 1,
            result['name'],
            f"{result['avg_time']:.4f}",
            f"{speedup:.2f}x"
        ])
    
    # 打印表格
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 解释循环分块的性能提升原理
    print("\n===== Why Loop Tiling Improves Performance =====")
    print("1. Cache Locality Optimization:")
    print("   - Loop tiling improves spatial locality by working on small blocks of data that fit in cache")
    print("   - Reduces cache misses by reusing data that's already in the cache")
    print("   - Minimizes the number of times data needs to be fetched from main memory")
    print("")
    print("2. Memory Hierarchy Utilization:")
    print("   - Better utilization of L1, L2, and L3 caches")
    print("   - Smaller working set size fits in faster cache levels")
    print("   - Reduces cache evictions and thrashing")
    print("")
    print("3. Reduced Memory Traffic:")
    print("   - Decreases the number of memory accesses")
    print("   - Improves memory bandwidth utilization")
    print("   - Particularly effective for algorithms with low computational intensity")
    print("")
    print("4. Compiler Optimization Opportunities:")
    print("   - Enables better register allocation")
    print("   - Creates more opportunities for instruction-level parallelism")
    print("   - Facilitates better loop unrolling and vectorization")
    
    # 分析各种优化技术的效果
    print("\n===== Analysis of Optimization Techniques =====")
    
    # 循环分块分析
    tiling_result = next((r for r in performance_results if 'Tiling' in r['name']), None)
    if tiling_result:
        print("\nLoop Tiling:")
        print("- Breaks large loops into smaller 'tiles' that fit in cache")
        print(f"- Achieved speedup of {original_time/tiling_result['avg_time']:.2f}x compared to original")
        print("- Most effective when working set size exceeds cache capacity")
    
    # 循环交换分析
    loop_swap_result = next((r for r in performance_results if 'Loop Swap' in r['name']), None)
    if loop_swap_result:
        print("\nLoop Swap:")
        print("- Changes loop order to improve spatial locality")
        print("- Optimizes memory access patterns to follow row-major or column-major order")
        print(f"- Achieved speedup of {original_time/loop_swap_result['avg_time']:.2f}x compared to original")
    
    # 循环展开分析
    unrolled_result = next((r for r in performance_results if 'Unrolling' in r['name'] and 'Tiling' not in r['name']), None)
    if unrolled_result:
        print("\nLoop Unrolling:")
        print("- Reduces loop control overhead by processing multiple elements per iteration")
        print("- Enables better instruction scheduling and reduces branch prediction failures")
        print(f"- Achieved speedup of {original_time/unrolled_result['avg_time']:.2f}x compared to original")
    
    # 组合优化分析
    combined_result = next((r for r in performance_results if 'Tiling + Unrolling' in r['name']), None)
    if combined_result:
        print("\nCombined Tiling + Unrolling:")
        print("- Leverages the benefits of both techniques")
        print("- Improves cache locality while reducing loop overhead")
        print(f"- Achieved speedup of {original_time/combined_result['avg_time']:.2f}x compared to original")
    
    # 边界条件优化分析
    boundary_result = next((r for r in performance_results if 'Boundary' in r['name']), None)
    if boundary_result:
        print("\nBoundary Condition Optimization:")
        print("- Removes conditional checks from inner loops for most of the data")
        print("- Processes boundary regions separately")
        print(f"- Achieved speedup of {original_time/boundary_result['avg_time']:.2f}x compared to original")
    
    # 前缀和优化分析
    prefix_sum_result = next((r for r in performance_results if 'Prefix Sum' in r['name']), None)
    if prefix_sum_result:
        print("\nPrefix Sum Optimization:")
        print("- Reduces time complexity from O(n^2 * k) to O(n^2)")
        print("- Eliminates redundant calculations by precomputing sums")
        print(f"- Achieved speedup of {original_time/prefix_sum_result['avg_time']:.2f}x compared to original")
        print("- Most effective when the stencil size (blur radius) is large")

# 可视化性能结果
def visualize_results(performance_results):
    """可视化各种优化技术的性能结果"""
    try:
        # 创建结果目录
        results_dir = "loop_optimization_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 按执行时间排序
        performance_results.sort(key=lambda x: x['avg_time'])
        
        # 提取数据
        names = [r['name'] for r in performance_results]
        times = [r['avg_time'] for r in performance_results]
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制条形图
        bars = plt.barh(names, times, color='skyblue')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 10, bar.get_y() + bar.get_height()/2, f'{width:.2f} ms', 
                     ha='left', va='center')
        
        # 添加标签和标题
        plt.xlabel('Execution Time (ms)')
        plt.ylabel('Optimization Technique')
        plt.title('Performance Comparison of Loop Optimization Techniques')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(results_dir, "loop_optimization_performance.png"))
        print("\nPerformance visualization saved as 'loop_optimization_results/loop_optimization_performance.png'")
        
    except Exception as e:
        print(f"Failed to generate visualization: {str(e)}")

# 主函数
def main():
    """主函数，协调整个循环优化测试流程"""
    # 运行性能测试
    run_performance_tests()
    
    print("\n===== Loop optimization testing completed =====")

if __name__ == "__main__":
    main()