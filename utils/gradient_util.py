import torch

def check_and_clean_gradients(params):
    non_finite_exist = False
    grad_cnt = 0
    non_finite_cnt = 0
    for param in params:
        if param.grad is not None:
            grad_cnt += 1
            # 检查梯度中是否有NaN或Inf
            grad = param.grad.data
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                non_finite_exist = True
                non_finite_cnt += 1
                # 将非有限梯度置零
                param.grad.data = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
    return non_finite_exist, grad_cnt, non_finite_cnt