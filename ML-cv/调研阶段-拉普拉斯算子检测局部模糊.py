import cv2
import numpy as np
import matplotlib.pyplot as plt

def blur_mask_laplacian(image_path, block_size=64, threshold=100):
    """
    Laplacian 方差模糊检测，生成局部模糊 mask
    Args:
        image_path: 输入图像路径
        block_size: 检测窗口大小（越小越精细）
        threshold: 方差阈值（越大越严格，数值需调参）
    Returns:
        original: 原图
        mask: 模糊区域二值 mask
        heatmap: 方差可视化热力图
    """
    # 读取灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    # 存储方差图
    var_map = np.zeros((h // block_size, w // block_size), dtype=np.float32)

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            patch = img[i:i+block_size, j:j+block_size]
            lap = cv2.Laplacian(patch, cv2.CV_64F)
            var = lap.var()
            var_map[i//block_size, j//block_size] = var

    # 归一化到 0-255
    heatmap = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)

    # 阈值化得到 mask（模糊区域为1）
    mask = (heatmap < threshold).astype(np.uint8) * 255

    return img, mask, heatmap

def visualize(image_path):
    img, mask, heatmap = blur_mask_laplacian(image_path)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Blur Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Variance Heatmap")
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')

    plt.savefig("blur_detection_result_6_1.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    visualize("/home/abling/Compilers_learning/MLsys_learning_2025summervacation/JotangRecrument-main/JotangRecrument-main/ML/task_3/image_pairs/your_result/6.png")
