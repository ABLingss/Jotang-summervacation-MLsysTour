"""
思路如下：

先用你的 blur_mask_laplacian 得到 mask，模糊区域为白色。

原始图像和 mask 一起送进 DocRes 的 deblurring 模块：

只在模糊区域跑 deblurring，非模糊区域直接拷贝。

比如这样改：

from 调研阶段_拉普拉斯算子检测局部模糊 import blur_mask_laplacian

def selective_deblurring(model, img_path):
    # 1. Laplacian 生成 mask
    img, mask, heatmap = blur_mask_laplacian(img_path, block_size=64, threshold=80)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转成3通道方便融合

    # 2. 用DocRes跑deblurring
    _,_,_, deblurred = deblurring(model, img_path)

    # 3. 按mask融合：模糊区域用deblurred，清晰区域保持原图
    result = np.where(mask==255, deblurred, cv2.imread(img_path))

    return result, mask, heatmap


然后调用：

result, mask, heatmap = selective_deblurring(model, "xxx.png")
cv2.imwrite("restorted/xxx_selective.png", result)

"""