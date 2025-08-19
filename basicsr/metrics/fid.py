import numpy as np
import torch
from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_fid(img, img2, crop_border=0, input_order='HWC', **kwargs):
    """Calculate FID (Fréchet Inception Distance).
    
    Args:
        img (ndarray): Real images with range [0, 255].
        img2 (ndarray): Generated images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
        
    Returns:
        float: fid result.
    """
    try:
        from pytorch_fid import fid_score
    except ImportError:
        raise ImportError('Please install pytorch-fid: pip install pytorch-fid')
    
    # 重新排列图像顺序
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    
    # 裁剪边界
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    # 保存图像到临时文件（FID需要文件路径）
    import tempfile
    import os
    from PIL import Image
    
    with tempfile.TemporaryDirectory() as temp_dir:
        real_dir = os.path.join(temp_dir, 'real')
        fake_dir = os.path.join(temp_dir, 'fake')
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)
        
        # 保存图像
        for i in range(img.shape[0]):
            real_img = Image.fromarray(img[i].astype(np.uint8))
            fake_img = Image.fromarray(img2[i].astype(np.uint8))
            real_img.save(os.path.join(real_dir, f'{i}.png'))
            fake_img.save(os.path.join(fake_dir, f'{i}.png'))
        
        # 计算FID
        fid_value = fid_score.calculate_fid_given_paths([real_dir, fake_dir], 
                                                       batch_size=50, 
                                                       device='cuda' if torch.cuda.is_available() else 'cpu')
    
    return fid_value 