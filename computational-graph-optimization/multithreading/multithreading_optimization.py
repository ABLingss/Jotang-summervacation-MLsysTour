#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多线程优化分析工具

此脚本实现了以下功能：
1. 分析不同并行策略对计算图性能的影响
2. 实现数据并行和任务并行的优化方法
3. 评估并行计算的加速比和效率
4. 研究线程数与性能的关系
5. 分析并行计算的瓶颈

使用方法：
    python multithreading_optimization.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import concurrent.futures
import threading
from queue import Queue

# 配置参数
ARRAY_SIZES = [10**6, 10**7, 10**8]       # 测试的数组大小
NUM_THREADS_OPTIONS = [1, 2, 4, 8, 16]    # 测试的线程数选项
REPEATS = 5                               # 每个测试重复次数
WARMUP_ITERATIONS = 3                     # 预热迭代次数

# 初始化数据
def initialize_data(size, dtype=np.float64):
    """初始化测试用数据"""
    return np.random.rand(size).astype(dtype)

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

# 1. 基本的单线程计算函数

# 单线程向量运算
def vector_operation_singlethreaded(data, operation='sum'):
    """单线程向量运算"""
    if operation == 'sum':
        return np.sum(data)
    elif operation == 'mean':
        return np.mean(data)
    elif operation == 'sqrt':
        return np.sqrt(data)
    elif operation == 'square':
        return np.square(data)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

# 单线程矩阵乘法
def matrix_multiply_singlethreaded(A, B):
    """单线程矩阵乘法"""
    return np.matmul(A, B)

# 2. 数据并行实现

# 数据并行向量运算（使用concurrent.futures）
def vector_operation_dataparallel(data, num_threads, operation='sum'):
    """数据并行向量运算"""
    # 将数据分成num_threads个部分
    chunk_size = len(data) // num_threads
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    # 确保分块数量等于线程数
    if len(chunks) > num_threads:
        # 合并最后两个块
        chunks[-2] = np.concatenate((chunks[-2], chunks[-1]))
        chunks.pop()
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务
        futures = [executor.submit(vector_operation_singlethreaded, chunk, operation) for chunk in chunks]
        
        # 收集结果
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # 合并结果
    if operation == 'sum':
        return sum(results)
    elif operation == 'mean':
        # 加权平均
        total = sum(results[i] * len(chunks[i]) for i in range(len(results)))
        return total / len(data)
    elif operation in ['sqrt', 'square']:
        return np.concatenate(results)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

# 数据并行矩阵乘法（按行分块）
def matrix_multiply_row_parallel(A, B, num_threads):
    """按行分块的并行矩阵乘法"""
    m, k = A.shape
    k, n = B.shape
    result = np.zeros((m, n))
    
    # 将矩阵A按行分块
    chunk_size = m // num_threads
    
    # 定义每个线程的工作函数
    def multiply_chunk(start_row, end_row):
        """计算矩阵块的乘积"""
        local_result = np.matmul(A[start_row:end_row, :], B)
        result[start_row:end_row, :] = local_result
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务
        futures = []
        for i in range(num_threads):
            start_row = i * chunk_size
            # 最后一个线程处理剩余的所有行
            end_row = m if i == num_threads - 1 else (i + 1) * chunk_size
            futures.append(executor.submit(multiply_chunk, start_row, end_row))
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 确保没有异常
    
    return result

# 3. 任务并行实现

# 任务并行工作负载
class TaskParallelWorkload:
    """任务并行工作负载管理器"""
    def __init__(self, data, num_tasks):
        self.data = data
        self.num_tasks = num_tasks
        self.results = [None] * num_tasks
        self.task_queue = Queue()
        
        # 填充任务队列
        chunk_size = len(data) // num_tasks
        for i in range(num_tasks):
            start = i * chunk_size
            end = len(data) if i == num_tasks - 1 else (i + 1) * chunk_size
            self.task_queue.put((i, start, end))
    
    def worker(self):
        """工作线程函数"""
        while not self.task_queue.empty():
            try:
                task_id, start, end = self.task_queue.get_nowait()
                # 执行任务（这里示例为计算子数组的和）
                self.results[task_id] = np.sum(self.data[start:end])
                self.task_queue.task_done()
            except Exception:
                break
    
    def execute(self, num_threads):
        """执行任务并行计算"""
        threads = []
        
        # 创建并启动工作线程
        for _ in range(num_threads):
            thread = threading.Thread(target=self.worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 合并结果
        return sum(self.results)

# 4. 混合并行策略

# 混合数据并行和任务并行的矩阵运算
class HybridParallelMatrixOperations:
    """混合并行矩阵运算"""
    def __init__(self, matrix):
        self.matrix = matrix
        self.rows, self.cols = matrix.shape
        self.result = np.zeros_like(matrix)
    
    def row_operation(self, start_row, end_row):
        """对矩阵的指定行执行操作"""
        for i in range(start_row, end_row):
            # 这里示例为计算每行的平方和平方根
            row_sum = np.sum(self.matrix[i, :] ** 2)
            self.result[i, :] = self.matrix[i, :] * np.sqrt(row_sum)
    
    def execute(self, num_threads):
        """执行混合并行计算"""
        # 将矩阵按行分块
        chunk_size = self.rows // num_threads
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交任务
            futures = []
            for i in range(num_threads):
                start_row = i * chunk_size
                # 最后一个线程处理剩余的所有行
                end_row = self.rows if i == num_threads - 1 else (i + 1) * chunk_size
                futures.append(executor.submit(self.row_operation, start_row, end_row))
            
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 确保没有异常
        
        return self.result

# 5. 线程安全的数据结构

# 线程安全的累加器
class ThreadSafeAccumulator:
    """线程安全的累加器"""
    def __init__(self):
        self.value = 0.0
        self.lock = threading.Lock()
    
    def add(self, delta):
        """线程安全地增加累加器的值"""
        with self.lock:
            self.value += delta
    
    def get(self):
        """获取累加器的当前值"""
        with self.lock:
            return self.value

# 使用线程安全累加器的并行求和
def parallel_sum_with_accumulator(data, num_threads):
    """使用线程安全累加器的并行求和"""
    accumulator = ThreadSafeAccumulator()
    
    # 将数据分成num_threads个部分
    chunk_size = len(data) // num_threads
    
    # 定义每个线程的工作函数
    def sum_chunk(start, end):
        """计算数据块的和并累加到线程安全累加器"""
        chunk_sum = np.sum(data[start:end])
        accumulator.add(chunk_sum)
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务
        futures = []
        for i in range(num_threads):
            start = i * chunk_size
            # 最后一个线程处理剩余的所有数据
            end = len(data) if i == num_threads - 1 else (i + 1) * chunk_size
            futures.append(executor.submit(sum_chunk, start, end))
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 确保没有异常
    
    return accumulator.get()

# 运行多线程优化分析
def run_multithreading_analysis():
    """运行多线程优化分析"""
    print("===== Multithreading Optimization Analysis =====")
    
    # 1. 不同线程数对向量运算性能的影响
    print("\n=== 1. Effect of Thread Count on Vector Operation Performance ===")
    
    vector_results = []
    base_size = ARRAY_SIZES[-1]  # 使用最大的数组进行线程数测试
    
    # 初始化数据
    data = initialize_data(base_size)
    
    # 先测量单线程性能（作为基准）
    single_thread_time, single_thread_result = measure_time(vector_operation_singlethreaded, data, 'sum')
    print(f"Single-threaded time: {single_thread_time:.2f} ms")
    
    # 测试不同线程数
    for num_threads in NUM_THREADS_OPTIONS[1:]:  # 跳过已经测试过的单线程
        thread_time, thread_result = measure_time(vector_operation_dataparallel, data, num_threads, 'sum')
        
        # 计算加速比和效率
        speedup = single_thread_time / thread_time
        efficiency = speedup / num_threads * 100  # 百分比
        
        # 验证结果正确性
        results_match = np.isclose(thread_result, single_thread_result)
        
        print(f"{num_threads} threads time: {thread_time:.2f} ms, speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%")
        print(f"Results match: {results_match}")
        
        # 记录结果
        vector_results.append((num_threads, thread_time, speedup, efficiency))
    
    # 2. 数据大小对并行性能的影响
    print("\n=== 2. Effect of Data Size on Parallel Performance ===")
    
    size_results = []
    fixed_num_threads = 4  # 使用固定的线程数测试不同数据大小
    
    for size in ARRAY_SIZES:
        print(f"\nTesting array size: {size:,}")
        
        # 初始化数据
        data = initialize_data(size)
        
        # 测量单线程性能
        single_thread_time, single_thread_result = measure_time(vector_operation_singlethreaded, data, 'sum')
        print(f"Single-threaded time: {single_thread_time:.2f} ms")
        
        # 测量多线程性能
        multi_thread_time, multi_thread_result = measure_time(vector_operation_dataparallel, data, fixed_num_threads, 'sum')
        
        # 计算加速比和效率
        speedup = single_thread_time / multi_thread_time
        efficiency = speedup / fixed_num_threads * 100  # 百分比
        
        # 验证结果正确性
        results_match = np.isclose(multi_thread_result, single_thread_result)
        
        print(f"{fixed_num_threads} threads time: {multi_thread_time:.2f} ms, speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%")
        print(f"Results match: {results_match}")
        
        # 记录结果
        size_results.append((size, single_thread_time, multi_thread_time, speedup, efficiency))
    
    # 3. 并行矩阵乘法性能
    print("\n=== 3. Parallel Matrix Multiplication Performance ===")
    
    matrix_results = []
    matrix_sizes = [200, 400, 600, 800]  # 测试的矩阵大小
    fixed_num_threads = 4  # 使用固定的线程数
    
    for size in matrix_sizes:
        print(f"\nTesting matrix size: {size}x{size}")
        
        # 初始化矩阵
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        # 测量单线程性能
        single_thread_time, single_thread_result = measure_time(matrix_multiply_singlethreaded, A, B)
        print(f"Single-threaded time: {single_thread_time:.2f} ms")
        
        # 测量多线程性能
        multi_thread_time, multi_thread_result = measure_time(matrix_multiply_row_parallel, A, B, fixed_num_threads)
        
        # 计算加速比和效率
        speedup = single_thread_time / multi_thread_time
        efficiency = speedup / fixed_num_threads * 100  # 百分比
        
        # 验证结果正确性
        results_match = np.allclose(multi_thread_result, single_thread_result)
        
        print(f"{fixed_num_threads} threads time: {multi_thread_time:.2f} ms, speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%")
        print(f"Results match: {results_match}")
        
        # 记录结果
        matrix_results.append((size, single_thread_time, multi_thread_time, speedup, efficiency))
    
    # 4. 任务并行vs数据并行性能比较
    print("\n=== 4. Task Parallel vs Data Parallel Performance ===")
    
    parallel_type_results = []
    comparison_size = 10**7  # 比较的数据集大小
    fixed_num_threads = 4    # 使用固定的线程数
    num_tasks = 16           # 任务数量
    
    # 初始化数据
    data = initialize_data(comparison_size)
    
    # 测量数据并行性能
    data_parallel_time, data_parallel_result = measure_time(vector_operation_dataparallel, data, fixed_num_threads, 'sum')
    print(f"Data parallel time: {data_parallel_time:.2f} ms")
    
    # 测量任务并行性能
    task_workload = TaskParallelWorkload(data, num_tasks)
    task_parallel_time, _ = measure_time(task_workload.execute, fixed_num_threads)
    print(f"Task parallel time: {task_parallel_time:.2f} ms")
    
    # 计算性能比
    performance_ratio = data_parallel_time / task_parallel_time
    print(f"Data parallel vs Task parallel ratio: {performance_ratio:.2f}x")
    
    # 记录结果
    parallel_type_results.append((comparison_size, data_parallel_time, task_parallel_time, performance_ratio))
    
    # 5. 混合并行策略性能
    print("\n=== 5. Hybrid Parallel Strategy Performance ===")
    
    hybrid_results = []
    hybrid_matrix_size = 800  # 混合并行测试的矩阵大小
    
    # 初始化矩阵
    matrix = np.random.rand(hybrid_matrix_size, hybrid_matrix_size)
    
    # 测量单线程性能（作为基准）
    def singlethreaded_hybrid_operation(mat):
        """单线程执行混合操作"""
        result = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            row_sum = np.sum(mat[i, :] ** 2)
            result[i, :] = mat[i, :] * np.sqrt(row_sum)
        return result
    
    single_thread_time, single_thread_result = measure_time(singlethreaded_hybrid_operation, matrix)
    print(f"Single-threaded time: {single_thread_time:.2f} ms")
    
    # 测试不同线程数的混合并行性能
    for num_threads in NUM_THREADS_OPTIONS:
        if num_threads == 1:
            # 已经测量过单线程性能
            speedup = 1.0
            efficiency = 100.0
        else:
            hybrid_workload = HybridParallelMatrixOperations(matrix)
            multi_thread_time, multi_thread_result = measure_time(hybrid_workload.execute, num_threads)
            
            # 计算加速比和效率
            speedup = single_thread_time / multi_thread_time
            efficiency = speedup / num_threads * 100  # 百分比
            
            # 验证结果正确性
            results_match = np.allclose(multi_thread_result, single_thread_result)
            
            print(f"{num_threads} threads time: {multi_thread_time:.2f} ms, speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%")
            print(f"Results match: {results_match}")
        
        # 记录结果
        hybrid_results.append((num_threads, speedup, efficiency))
    
    # 6. 线程安全数据结构性能
    print("\n=== 6. Thread-Safe Data Structure Performance ===")
    
    threadsafe_results = []
    threadsafe_size = 10**7  # 测试大小
    
    # 初始化数据
    data = initialize_data(threadsafe_size)
    
    # 测量普通并行求和性能
    regular_time, regular_result = measure_time(vector_operation_dataparallel, data, fixed_num_threads, 'sum')
    print(f"Regular parallel sum time: {regular_time:.2f} ms")
    
    # 测量使用线程安全累加器的并行求和性能
    threadsafe_time, threadsafe_result = measure_time(parallel_sum_with_accumulator, data, fixed_num_threads)
    print(f"Thread-safe accumulator sum time: {threadsafe_time:.2f} ms")
    
    # 计算性能比
    performance_ratio = regular_time / threadsafe_time
    print(f"Regular vs Thread-safe ratio: {performance_ratio:.2f}x")
    
    # 验证结果正确性
    results_match = np.isclose(regular_result, threadsafe_result)
    print(f"Results match: {results_match}")
    
    # 记录结果
    threadsafe_results.append((threadsafe_size, regular_time, threadsafe_time, performance_ratio))
    
    # 生成性能报告
    generate_report(
        vector_results,
        size_results,
        matrix_results,
        parallel_type_results,
        hybrid_results,
        threadsafe_results,
        single_thread_time,  # 来自向量运算的单线程时间
        NUM_THREADS_OPTIONS
    )
    
    # 可视化性能结果
    visualize_results(
        vector_results,
        size_results,
        matrix_results,
        parallel_type_results,
        hybrid_results,
        threadsafe_results,
        NUM_THREADS_OPTIONS
    )

# 生成多线程优化分析报告
def generate_report(
    vector_results,
    size_results,
    matrix_results,
    parallel_type_results,
    hybrid_results,
    threadsafe_results,
    base_single_thread_time,
    num_threads_options
):
    """生成多线程优化分析报告"""
    print("\n===== Multithreading Optimization Analysis Report =====")
    
    # 1. 线程数对性能的影响
    print("\n=== 1. Effect of Thread Count on Performance ===")
    headers = ["Number of Threads", "Execution Time (ms)", "Speedup", "Efficiency (%)"]
    
    rows = [[1, f"{base_single_thread_time:.2f}", "1.00x", "100.0%"]]  # 添加单线程结果
    for num_threads, time_val, speedup, efficiency in vector_results:
        rows.append([
            num_threads,
            f"{time_val:.2f}",
            f"{speedup:.2f}x",
            f"{efficiency:.1f}%"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 2. 数据大小对并行性能的影响
    print("\n=== 2. Effect of Data Size on Parallel Performance ===")
    headers = ["Data Size", "Single-thread (ms)", "Multi-thread (ms)", "Speedup", "Efficiency (%)"]
    
    rows = []
    for size, single_time, multi_time, speedup, efficiency in size_results:
        rows.append([
            f"{size:,}",
            f"{single_time:.2f}",
            f"{multi_time:.2f}",
            f"{speedup:.2f}x",
            f"{efficiency:.1f}%"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 3. 并行矩阵乘法性能
    print("\n=== 3. Parallel Matrix Multiplication Performance ===")
    headers = ["Matrix Size", "Single-thread (ms)", "Multi-thread (ms)", "Speedup", "Efficiency (%)"]
    
    rows = []
    for size, single_time, multi_time, speedup, efficiency in matrix_results:
        rows.append([
            f"{size}x{size}",
            f"{single_time:.2f}",
            f"{multi_time:.2f}",
            f"{speedup:.2f}x",
            f"{efficiency:.1f}%"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 4. 任务并行vs数据并行性能比较
    print("\n=== 4. Task Parallel vs Data Parallel Performance ===")
    headers = ["Data Size", "Data Parallel (ms)", "Task Parallel (ms)", "Data/Task Ratio"]
    
    rows = []
    for size, data_time, task_time, ratio in parallel_type_results:
        rows.append([
            f"{size:,}",
            f"{data_time:.2f}",
            f"{task_time:.2f}",
            f"{ratio:.2f}x"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 5. 混合并行策略性能
    print("\n=== 5. Hybrid Parallel Strategy Performance ===")
    headers = ["Number of Threads", "Speedup", "Efficiency (%)"]
    
    rows = []
    for num_threads, speedup, efficiency in hybrid_results:
        rows.append([
            num_threads,
            f"{speedup:.2f}x",
            f"{efficiency:.1f}%"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 6. 线程安全数据结构性能
    print("\n=== 6. Thread-Safe Data Structure Performance ===")
    headers = ["Data Size", "Regular Parallel (ms)", "Thread-Safe (ms)", "Regular/Thread-Safe Ratio"]
    
    rows = []
    for size, regular_time, threadsafe_time, ratio in threadsafe_results:
        rows.append([
            f"{size:,}",
            f"{regular_time:.2f}",
            f"{threadsafe_time:.2f}",
            f"{ratio:.2f}x"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 总结多线程优化建议
    print("\n===== Multithreading Optimization Recommendations =====")
    print("1. Thread Count Selection:")
    print("   - Choose the number of threads based on the available CPU cores")
    print("   - Be aware of diminishing returns beyond a certain number of threads")
    print("   - Consider hyper-threading capabilities of your processor")
    print("")
    print("2. Parallelization Strategies:")
    print("   - Use data parallelism for large homogeneous datasets")
    print("   - Use task parallelism for collections of independent tasks")
    print("   - Consider hybrid approaches for complex workloads")
    print("")
    print("3. Load Balancing:")
    print("   - Ensure even distribution of work among threads")
    print("   - Consider dynamic load balancing for variable workloads")
    print("   - Monitor thread utilization to identify bottlenecks")
    print("")
    print("4. Memory Considerations:")
    print("   - Be mindful of memory bandwidth limitations in parallel computations")
    print("   - Minimize shared memory access to reduce contention")
    print("   - Consider data locality when partitioning workloads")
    print("")
    print("5. Thread Safety:")
    print("   - Use thread-safe data structures for shared data")
    print("   - Minimize the use of locks to reduce synchronization overhead")
    print("   - Consider lock-free algorithms for high-contention scenarios")
    print("")
    print("6. Performance Monitoring:")
    print("   - Measure speedup and efficiency to evaluate parallel performance")
    print("   - Identify serial bottlenecks in parallel code")
    print("   - Profile your application to find optimization opportunities")
    print("")
    print("7. Choosing the Right Tool:")
    print("   - Use high-level parallel libraries when possible")
    print("   - Consider specialized frameworks for specific workloads")
    print("   - Evaluate the trade-offs between simplicity and performance")

# 可视化多线程优化分析结果
def visualize_results(
    vector_results,
    size_results,
    matrix_results,
    parallel_type_results,
    hybrid_results,
    threadsafe_results,
    num_threads_options
):
    """可视化多线程优化分析结果"""
    try:
        # 创建结果目录
        results_dir = "multithreading_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. 线程数对加速比和效率的影响
        thread_counts = [1] + [r[0] for r in vector_results]  # 添加单线程
        speedups = [1.0] + [r[2] for r in vector_results]
        efficiencies = [100.0] + [r[3] for r in vector_results]
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        ax1.set_xlabel('Number of Threads')
        ax1.set_ylabel('Speedup', color='blue')
        ax1.plot(thread_counts, speedups, marker='o', color='blue', label='Speedup')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # 添加理想加速比参考线
        ideal_speedups = [min(tc, num_threads_options[-1]) for tc in thread_counts]  # 假设最多16核
        ax1.plot(thread_counts, ideal_speedups, linestyle='--', color='blue', alpha=0.5, label='Ideal Speedup')
        
        ax2 = ax1.twinx()  # 创建第二个y轴
        ax2.set_ylabel('Efficiency (%)', color='red')
        ax2.plot(thread_counts, efficiencies, marker='s', color='red', label='Efficiency')
        ax2.tick_params(axis='y', labelcolor='red')
        
        fig.tight_layout()
        plt.title('Effect of Thread Count on Speedup and Efficiency')
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(results_dir, "thread_count_effect.png"))
        print("\nThread count effect visualization saved as 'multithreading_results/thread_count_effect.png'")
        
        # 2. 数据大小对并行性能的影响
        sizes = [f"{r[0]:,}" for r in size_results]
        single_times = [r[1] for r in size_results]
        multi_times = [r[2] for r in size_results]
        speedups = [r[3] for r in size_results]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(sizes, single_times, marker='o', label='Single-threaded')
        plt.plot(sizes, multi_times, marker='s', label='Multi-threaded')
        plt.ylabel('Execution Time (ms)')
        plt.title('Effect of Data Size on Execution Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(2, 1, 2)
        plt.plot(sizes, speedups, marker='^', color='green')
        plt.xlabel('Data Size')
        plt.ylabel('Speedup')
        plt.title('Effect of Data Size on Speedup')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "data_size_effect.png"))
        print("Data size effect visualization saved as 'multithreading_results/data_size_effect.png'")
        
        # 3. 矩阵乘法并行性能
        matrix_sizes = [f"{r[0]}x{r[0]}" for r in matrix_results]
        single_times = [r[1] for r in matrix_results]
        multi_times = [r[2] for r in matrix_results]
        speedups = [r[3] for r in matrix_results]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(matrix_sizes, single_times, marker='o', label='Single-threaded')
        plt.plot(matrix_sizes, multi_times, marker='s', label='Multi-threaded')
        plt.ylabel('Execution Time (ms)')
        plt.title('Matrix Multiplication Execution Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(2, 1, 2)
        plt.plot(matrix_sizes, speedups, marker='^', color='green')
        plt.xlabel('Matrix Size')
        plt.ylabel('Speedup')
        plt.title('Matrix Multiplication Speedup')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "matrix_multiplication_performance.png"))
        print("Matrix multiplication performance visualization saved as 'multithreading_results/matrix_multiplication_performance.png'")
        
        # 4. 混合并行策略性能
        thread_counts = [r[0] for r in hybrid_results]
        speedups = [r[1] for r in hybrid_results]
        efficiencies = [r[2] for r in hybrid_results]
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        ax1.set_xlabel('Number of Threads')
        ax1.set_ylabel('Speedup', color='blue')
        ax1.plot(thread_counts, speedups, marker='o', color='blue', label='Speedup')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # 添加理想加速比参考线
        ideal_speedups = [min(tc, num_threads_options[-1]) for tc in thread_counts]
        ax1.plot(thread_counts, ideal_speedups, linestyle='--', color='blue', alpha=0.5, label='Ideal Speedup')
        
        ax2 = ax1.twinx()  # 创建第二个y轴
        ax2.set_ylabel('Efficiency (%)', color='red')
        ax2.plot(thread_counts, efficiencies, marker='s', color='red', label='Efficiency')
        ax2.tick_params(axis='y', labelcolor='red')
        
        fig.tight_layout()
        plt.title('Hybrid Parallel Strategy Performance')
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(results_dir, "hybrid_parallel_performance.png"))
        print("Hybrid parallel performance visualization saved as 'multithreading_results/hybrid_parallel_performance.png'")
        
    except Exception as e:
        print(f"Failed to generate visualizations: {str(e)}")

# 主函数
def main():
    """主函数，协调整个多线程优化分析流程"""
    # 运行多线程优化分析
    run_multithreading_analysis()
    
    print("\n===== Multithreading optimization analysis completed =====")

if __name__ == "__main__":
    main()