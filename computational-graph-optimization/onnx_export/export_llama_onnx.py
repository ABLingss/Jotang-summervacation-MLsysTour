#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将Llama-3B模型导出为ONNX格式并用ONNX Runtime进行部署

此脚本实现了以下功能：
1. 加载Llama-3B模型
2. 将模型导出为ONNX格式
3. 使用ONNX Runtime加载并运行导出的模型
4. 比较原始模型和ONNX模型的输出结果

使用方法：
    python export_llama_onnx.py
"""

import os
import time
import torch
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# 配置参数
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # 使用7B模型作为示例，3B可能需要自定义下载
ONNX_MODEL_PATH = "llama_model.onnx"
CACHE_DIR = "./cache"
MAX_SEQ_LEN = 64
BATCH_SIZE = 1

# 创建缓存目录
def create_cache_directory():
    """创建模型缓存目录"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"Created cache directory: {CACHE_DIR}")

# 加载模型和分词器
def load_model_and_tokenizer():
    """加载预训练的Llama模型和分词器"""
    print(f"Loading model: {MODEL_NAME}")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        use_fast=True
    )
    
    # 为了避免加载完整模型可能导致的内存问题，这里使用CPU并设置低内存模式
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    # 设置模型为评估模式
    model.eval()
    
    end_time = time.time()
    print(f"Model loaded in {end_time - start_time:.2f} seconds")
    
    return model, tokenizer

# 创建示例输入
def create_sample_input(tokenizer):
    """创建用于导出和测试的示例输入"""
    prompt = "Hello, my name is"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN
    )
    
    # 对于ONNX导出，我们需要创建动态轴以支持可变长度的输入
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    return input_ids, attention_mask, prompt

# 将模型导出为ONNX格式
def export_model_to_onnx(model, input_ids, attention_mask):
    """将PyTorch模型导出为ONNX格式"""
    print(f"Exporting model to ONNX: {ONNX_MODEL_PATH}")
    start_time = time.time()
    
    # 定义动态轴以支持可变长度的输入
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"}
    }
    
    # 导出模型
    with torch.no_grad():
        torch.onnx.export(
            model,
            (
                input_ids,
                attention_mask
            ),
            ONNX_MODEL_PATH,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )
    
    end_time = time.time()
    print(f"Model exported in {end_time - start_time:.2f} seconds")

# 使用ONNX Runtime运行模型
def run_onnx_model(input_ids, attention_mask):
    """使用ONNX Runtime加载并运行ONNX模型"""
    print(f"Running model with ONNX Runtime")
    
    # 将PyTorch张量转换为NumPy数组
    input_ids_np = input_ids.numpy()
    attention_mask_np = attention_mask.numpy()
    
    # 创建ONNX Runtime会话
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"]  # 可以根据需要更改为GPU提供程序
    )
    
    # 准备输入字典
    inputs = {
        "input_ids": input_ids_np,
        "attention_mask": attention_mask_np
    }
    
    # 测量推理时间
    start_time = time.time()
    outputs = session.run(None, inputs)
    end_time = time.time()
    
    print(f"ONNX Runtime inference completed in {end_time - start_time:.4f} seconds")
    
    return outputs[0], end_time - start_time

# 使用原始PyTorch模型运行
def run_pytorch_model(model, input_ids, attention_mask):
    """使用原始PyTorch模型运行推理"""
    print("Running model with PyTorch")
    
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        end_time = time.time()
    
    print(f"PyTorch inference completed in {end_time - start_time:.4f} seconds")
    
    return outputs.logits.cpu().numpy(), end_time - start_time

# 比较结果
def compare_results(pytorch_output, onnx_output, tokenizer, prompt):
    """比较PyTorch模型和ONNX模型的输出结果"""
    print("Comparing results between PyTorch and ONNX models")
    
    # 计算输出差异
    diff = np.abs(pytorch_output - onnx_output)
    print(f"Max difference: {np.max(diff):.6f}")
    print(f"Mean difference: {np.mean(diff):.6f}")
    
    # 生成文本以验证模型功能
    pytorch_predicted_ids = np.argmax(pytorch_output, axis=-1)
    onnx_predicted_ids = np.argmax(onnx_output, axis=-1)
    
    pytorch_text = tokenizer.decode(pytorch_predicted_ids[0], skip_special_tokens=True)
    onnx_text = tokenizer.decode(onnx_predicted_ids[0], skip_special_tokens=True)
    
    print(f"\nOriginal prompt: {prompt}")
    print(f"PyTorch generated: {pytorch_text}")
    print(f"ONNX generated: {onnx_text}")
    
    # 检查结果是否足够接近
    return np.max(diff) < 1e-3  # 设置一个合理的阈值

# 主函数
def main():
    """主函数，协调整个导出和测试流程"""
    print("===== Llama Model ONNX Export and Deployment ======")
    
    try:
        # 创建缓存目录
        create_cache_directory()
        
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer()
        
        # 创建示例输入
        input_ids, attention_mask, prompt = create_sample_input(tokenizer)
        
        # 导出模型为ONNX格式
        export_model_to_onnx(model, input_ids, attention_mask)
        
        # 使用PyTorch模型运行
        pytorch_output, pytorch_time = run_pytorch_model(model, input_ids, attention_mask)
        
        # 使用ONNX Runtime运行
        onnx_output, onnx_time = run_onnx_model(input_ids, attention_mask)
        
        # 比较结果
        results_match = compare_results(pytorch_output, onnx_output, tokenizer, prompt)
        
        # 计算加速比
        speedup = pytorch_time / onnx_time if onnx_time > 0 else float('inf')
        print(f"\nPerformance comparison:")
        print(f"PyTorch time: {pytorch_time:.4f} seconds")
        print(f"ONNX Runtime time: {onnx_time:.4f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        
        print("\n===== Process completed =====")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\n===== Process failed =====")

if __name__ == "__main__":
    main()