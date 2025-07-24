import math

# 余弦学习率调度器
# 在训练过程中逐渐降低学习率，最终降到初始值的1/10
def get_lr(current_step, total_steps, lr):
    """使用余弦退火策略计算学习率
    Args:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 基础学习率
    Returns:
        当前步数对应的学习率
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))