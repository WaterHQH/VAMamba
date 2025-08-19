from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .psnr_ssim import calculate_psnr, calculate_ssim, calculate_lpips, calculate_mae, calculate_mae_y

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_lpips', 'calculate_mae', 'calculate_mae_y']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
