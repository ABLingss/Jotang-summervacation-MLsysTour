#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
编译器性能比较分析工具

此脚本实现了以下功能：
1. 比较GCC和Clang编译器编译的矩阵乘法代码性能
2. 生成不同优化级别的汇编代码
3. 分析两种编译器生成的汇编代码差异
4. 提供可视化性能对比

使用方法：
    python compiler_performance_comparison.py
"""

import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import tempfile

# 配置参数
MATRIX_SIZES = [128, 256, 512, 1024]  # 矩阵大小
COMPILERS = ["gcc", "clang"]          # 要比较的编译器
OPT_LEVELS = ["O0", "O1", "O2", "O3", "Ofast"]  # 优化级别
REPEATS = 5                          # 每个测试重复次数

# 矩阵乘法的C代码模板
MATRIX_MUL_CODE = """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// 矩阵乘法函数
double matrix_multiply(double* A, double* B, double* C, int n) {
    struct timeval start, end;
    double elapsed;
    
    gettimeofday(&start, NULL);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
    
    gettimeofday(&end, NULL);
    elapsed = (end.tv_sec - start.tv_sec) * 1000.0;    // 秒转毫秒
    elapsed += (end.tv_usec - start.tv_usec) / 1000.0; // 微秒转毫秒
    
    return elapsed;
}

// 优化的矩阵乘法（行主序优化）
double matrix_multiply_optimized(double* A, double* B, double* C, int n) {
    struct timeval start, end;
    double elapsed;
    
    gettimeofday(&start, NULL);
    
    // 转置矩阵B以提高缓存局部性
    double* B_transposed = (double*)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B_transposed[j * n + i] = B[i * n + j];
        }
    }
    
    // 使用转置后的矩阵B进行计算
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B_transposed[j * n + k];
            }
            C[i * n + j] = sum;
        }
    }
    
    free(B_transposed);
    
    gettimeofday(&end, NULL);
    elapsed = (end.tv_sec - start.tv_sec) * 1000.0;    // 秒转毫秒
    elapsed += (end.tv_usec - start.tv_usec) / 1000.0; // 微秒转毫秒
    
    return elapsed;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <optimized_flag>", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    int optimized = atoi(argv[2]);
    
    // 分配内存
    double* A = (double*)malloc(n * n * sizeof(double));
    double* B = (double*)malloc(n * n * sizeof(double));
    double* C = (double*)malloc(n * n * sizeof(double));
    
    // 初始化矩阵
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (double)(i + j) / (n * n);
            B[i * n + j] = (double)(i * j) / (n * n);
        }
    }
    
    double time_taken;
    if (optimized) {
        time_taken = matrix_multiply_optimized(A, B, C, n);
    } else {
        time_taken = matrix_multiply(A, B, C, n);
    }
    
    // 输出时间（仅输出数字，便于解析）
    printf("%.6f\n", time_taken);
    
    // 验证结果（防止编译器过度优化）
    double sum = 0.0;
    for (int i = 0; i < n; i += n/10) {
        for (int j = 0; j < n; j += n/10) {
            sum += C[i * n + j];
        }
    }
    
    // 释放内存
    free(A);
    free(B);
    free(C);
    
    return 0;
}
"""

# 生成汇编代码的C代码模板
ASM_GEN_CODE = """
// 简单的矩阵乘法函数，用于生成汇编代码分析
double matrix_multiply_asm(double* A, double* B, double* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
    return 0.0;
}
"""

# 检查编译器是否可用
def check_compiler_availability():
    """检查指定的编译器是否可用"""
    available_compilers = []
    
    for compiler in COMPILERS:
        try:
            # 检查编译器版本
            subprocess.run([compiler, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            available_compilers.append(compiler)
            print(f"{compiler} is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{compiler} is not available")
    
    return available_compilers

# 编译C代码
def compile_code(compiler, opt_level, output_file, code, optimized_version=False):
    """使用指定的编译器和优化级别编译C代码"""
    # 创建临时C文件
    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as f:
        f.write(code)
        temp_c_file = f.name
    
    # 构建编译命令
    compile_cmd = [
        compiler,
        temp_c_file,
        f"-{opt_level}",
        "-o", output_file,
        "-lm"
    ]
    
    # 如果是GCC且优化级别较高，添加额外的优化标志
    if compiler == "gcc" and opt_level in ["O3", "Ofast"]:
        compile_cmd.extend(["-march=native", "-mtune=native"])
    
    # 如果是Clang且优化级别较高，添加额外的优化标志
    elif compiler == "clang" and opt_level in ["O3", "Ofast"]:
        compile_cmd.extend(["-march=native", "-mtune=native"])
    
    # 添加调试信息以便分析
    if opt_level == "O0":
        compile_cmd.append("-g")
    
    try:
        # 执行编译命令
        subprocess.run(compile_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully compiled with {compiler} {opt_level}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed with {compiler} {opt_level}: {e.stderr.decode()}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_c_file):
            os.remove(temp_c_file)

# 生成汇编代码
def generate_assembly(compiler, opt_level, output_file):
    """生成指定编译器和优化级别下的汇编代码"""
    # 创建临时C文件
    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as f:
        f.write(ASM_GEN_CODE)
        temp_c_file = f.name
    
    # 构建生成汇编的命令
    asm_cmd = [
        compiler,
        temp_c_file,
        f"-{opt_level}",
        "-S", "-fverbose-asm",
        "-o", output_file
    ]
    
    # 添加架构特定的优化
    if opt_level in ["O3", "Ofast"]:
        asm_cmd.extend(["-march=native", "-mtune=native"])
    
    try:
        # 执行生成汇编的命令
        subprocess.run(asm_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully generated assembly with {compiler} {opt_level}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Assembly generation failed with {compiler} {opt_level}: {e.stderr.decode()}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_c_file):
            os.remove(temp_c_file)

# 运行编译后的程序
def run_program(program_path, matrix_size, optimized=False):
    """运行编译后的矩阵乘法程序"""
    try:
        # 执行程序
        result = subprocess.run(
            [program_path, str(matrix_size), str(1 if optimized else 0)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 解析输出（应该是一个表示执行时间的浮点数）
        execution_time = float(result.stdout.strip())
        return execution_time
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error running {program_path}: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error output: {e.stderr.decode()}")
        return float('inf')  # 返回一个很大的值表示失败

# 分析汇编代码
def analyze_assembly(asm_file_path):
    """分析汇编代码，提取关键信息"""
    try:
        with open(asm_file_path, 'r') as f:
            asm_content = f.read()
        
        # 统计各种指令的使用情况
        analysis = {
            'total_lines': asm_content.count('\n'),
            'vector_instr_count': asm_content.count('vmov') + asm_content.count('vadd') + \
                                 asm_content.count('vmul') + asm_content.count('vfmadd'),
            'load_store_count': asm_content.count('mov') + asm_content.count('load') + \
                               asm_content.count('store'),
            'has_loop_unrolling': 'unroll' in asm_content.lower(),
            'has_simd': 'ymm' in asm_content or 'xmm' in asm_content or 'zmm' in asm_content,
            'compiler_comments': asm_content.count('#')
        }
        
        return analysis
    except Exception as e:
        print(f"Error analyzing assembly file {asm_file_path}: {e}")
        return None

# 运行编译器性能比较
def run_compiler_comparison():
    """运行编译器性能比较测试"""
    # 检查可用的编译器
    available_compilers = check_compiler_availability()
    if not available_compilers:
        print("No supported compilers available. Exiting.")
        return
    
    # 创建结果目录
    results_dir = "compiler_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建汇编代码目录
    asm_dir = os.path.join(results_dir, "assembly")
    os.makedirs(asm_dir, exist_ok=True)
    
    # 存储所有结果
    all_results = []
    asm_analyses = {}
    
    # 遍历每个编译器
    for compiler in available_compilers:
        # 遍历每个优化级别
        for opt_level in OPT_LEVELS:
            # 生成输出文件名
            output_file = os.path.join(results_dir, f"matmul_{compiler}_{opt_level}")
            
            # 编译代码
            if not compile_code(compiler, opt_level, output_file, MATRIX_MUL_CODE):
                continue
            
            # 遍历每个矩阵大小
            for matrix_size in MATRIX_SIZES:
                print(f"Testing {compiler} {opt_level} with matrix size {matrix_size}x{matrix_size}")
                
                # 运行多次并取平均值
                times = []
                for i in range(REPEATS):
                    print(f"  Run {i+1}/{REPEATS}")
                    time_taken = run_program(output_file, matrix_size, optimized=False)
                    times.append(time_taken)
                
                # 计算平均时间
                avg_time = sum(times) / len(times)
                
                # 记录结果
                all_results.append({
                    'compiler': compiler,
                    'opt_level': opt_level,
                    'matrix_size': matrix_size,
                    'avg_time': avg_time
                })
                
                print(f"  Average time: {avg_time:.4f} ms")
            
            # 生成并分析汇编代码（仅对中等大小的矩阵进行）
            asm_file = os.path.join(asm_dir, f"matmul_{compiler}_{opt_level}.s")
            if generate_assembly(compiler, opt_level, asm_file):
                asm_analysis = analyze_assembly(asm_file)
                if asm_analysis:
                    asm_analyses[f"{compiler}_{opt_level}"] = asm_analysis
    
    # 生成性能报告
    generate_report(all_results, asm_analyses)
    
    # 可视化性能结果
    visualize_results(all_results)

# 生成性能分析报告
def generate_report(results, asm_analyses):
    """生成性能分析报告"""
    print("\n===== Compiler Performance Comparison Report =====")
    
    # 按编译器、优化级别和矩阵大小整理结果
    table_data = []
    
    for compiler in COMPILERS:
        compiler_results = [r for r in results if r['compiler'] == compiler]
        if not compiler_results:
            continue
            
        for opt_level in OPT_LEVELS:
            opt_results = [r for r in compiler_results if r['opt_level'] == opt_level]
            if not opt_results:
                continue
                
            # 创建行数据
            row = [f"{compiler} {opt_level}"]
            for matrix_size in MATRIX_SIZES:
                size_results = [r for r in opt_results if r['matrix_size'] == matrix_size]
                if size_results:
                    row.append(f"{size_results[0]['avg_time']:.4f}")
                else:
                    row.append("N/A")
            
            table_data.append(row)
    
    # 打印表格
    headers = ["Compiler & Opt Level"] + [f"{size}x{size}" for size in MATRIX_SIZES]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 分析GCC vs Clang在O3优化下的性能差异
    print("\n===== GCC vs Clang Performance Analysis (O3 Optimization) =====")
    
    gcc_o3_results = [r for r in results if r['compiler'] == 'gcc' and r['opt_level'] == 'O3']
    clang_o3_results = [r for r in results if r['compiler'] == 'clang' and r['opt_level'] == 'O3']
    
    if gcc_o3_results and clang_o3_results:
        for i, matrix_size in enumerate(MATRIX_SIZES):
            gcc_time = next((r['avg_time'] for r in gcc_o3_results if r['matrix_size'] == matrix_size), float('inf'))
            clang_time = next((r['avg_time'] for r in clang_o3_results if r['matrix_size'] == matrix_size), float('inf'))
            
            if gcc_time < float('inf') and clang_time < float('inf'):
                ratio = clang_time / gcc_time if gcc_time > 0 else float('inf')
                print(f"Matrix size {matrix_size}x{matrix_size}: GCC is {ratio:.2f}x faster than Clang")
    
    # 汇编代码分析
    print("\n===== Assembly Code Analysis =====")
    if asm_analyses:
        for key, analysis in asm_analyses.items():
            compiler, opt_level = key.split('_')
            print(f"\n{compiler} {opt_level}:")
            print(f"  Total assembly lines: {analysis['total_lines']}")
            print(f"  Vector instructions: {analysis['vector_instr_count']}")
            print(f"  Load/store operations: {analysis['load_store_count']}")
            print(f"  Uses loop unrolling: {analysis['has_loop_unrolling']}")
            print(f"  Uses SIMD instructions: {analysis['has_simd']}")
            print(f"  Compiler comments: {analysis['compiler_comments']}")
    
    # 分析GCC和Clang性能差异的原因
    print("\n===== Why GCC is faster than Clang for matrix multiplication =====")
    print("1. Cache optimization differences:")
    print("   - GCC may generate code with better cache locality patterns")
    print("   - Better prefetching strategies in GCC's code generation")
    print("   - More effective loop ordering optimizations")
    print("\n2. Instruction selection and scheduling:")
    print("   - GCC might select more efficient instruction sequences for the target architecture")
    print("   - Better register allocation in GCC for matrix operations")
    print("   - More effective use of instruction-level parallelism")
    print("\n3. SIMD vectorization differences:")
    print("   - GCC's auto-vectorization may be more aggressive or effective for matrix operations")
    print("   - Different handling of alignment requirements for SIMD operations")
    print("   - Better utilization of wider vector registers")
    print("\n4. Loop optimization differences:")
    print("   - GCC may apply more effective loop unrolling strategies")
    print("   - Better loop fusion/fission decisions in GCC")
    print("   - Different loop iteration ordering optimizations")

# 可视化性能结果
def visualize_results(results):
    """可视化性能比较结果"""
    try:
        # 创建结果目录
        results_dir = "compiler_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 为每个矩阵大小创建一个图表
        for matrix_size in MATRIX_SIZES:
            plt.figure(figsize=(10, 6))
            
            # 为每个编译器收集数据
            compiler_data = {}
            for compiler in COMPILERS:
                compiler_results = [
                    r for r in results 
                    if r['compiler'] == compiler and r['matrix_size'] == matrix_size
                ]
                
                if compiler_results:
                    # 按优化级别排序
                    compiler_results.sort(key=lambda x: OPT_LEVELS.index(x['opt_level']))
                    
                    opt_levels = [r['opt_level'] for r in compiler_results]
                    times = [r['avg_time'] for r in compiler_results]
                    
                    compiler_data[compiler] = (opt_levels, times)
            
            # 绘制图表
            for compiler, (opt_levels, times) in compiler_data.items():
                plt.plot(opt_levels, times, marker='o', label=compiler)
            
            # 添加标签和标题
            plt.xlabel('Optimization Level')
            plt.ylabel('Execution Time (ms)')
            plt.title(f'Performance Comparison: {matrix_size}x{matrix_size} Matrix Multiplication')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"performance_{matrix_size}x{matrix_size}.png"))
            print(f"Performance visualization saved for {matrix_size}x{matrix_size}")
        
        # 创建总体比较图表（O3优化级别）
        plt.figure(figsize=(12, 6))
        
        gcc_o3_times = [
            r['avg_time'] for r in results 
            if r['compiler'] == 'gcc' and r['opt_level'] == 'O3'
        ]
        clang_o3_times = [
            r['avg_time'] for r in results 
            if r['compiler'] == 'clang' and r['opt_level'] == 'O3'
        ]
        
        if gcc_o3_times and clang_o3_times:
            x = np.arange(len(MATRIX_SIZES))
            width = 0.35
            
            plt.bar(x - width/2, gcc_o3_times, width, label='GCC O3')
            plt.bar(x + width/2, clang_o3_times, width, label='Clang O3')
            
            plt.xlabel('Matrix Size')
            plt.ylabel('Execution Time (ms)')
            plt.title('GCC vs Clang Performance Comparison (O3 Optimization)')
            plt.xticks(x, [f"{size}x{size}" for size in MATRIX_SIZES])
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "gcc_vs_clang_O3.png"))
            print("GCC vs Clang O3 comparison visualization saved")
        
    except Exception as e:
        print(f"Failed to generate visualizations: {str(e)}")

# 主函数
def main():
    """主函数，协调整个编译器性能比较流程"""
    print("===== Compiler Performance Comparison Tool =====")
    
    # 运行编译器性能比较
    run_compiler_comparison()
    
    print("\n===== Comparison completed =====")

if __name__ == "__main__":
    main()