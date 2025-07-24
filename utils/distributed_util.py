import os
import torch
import torch.distributed as dist

# 初始化分布式训练环境
def init_distributed_mode():
    rank = int(os.environ.get("RANK", -1))
    if rank == -1: return
    global ddp_local_rank, DEVICE

    # 全局进程编号
    rank = int(os.environ["RANK"])
    # 本地进程编号
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    # 总进程数
    world_size = int(os.environ["WORLD_SIZE"])
    # 设置当前进程使用的设备
    DEVICE = f"cuda:{ddp_local_rank}"
    # 初始化分布式进程组，使用NCCL或HCCL后端
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(DEVICE)
    return ddp_local_rank, DEVICE