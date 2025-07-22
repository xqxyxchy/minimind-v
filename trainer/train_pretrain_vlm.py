import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import time
import math
import warnings

warnings.filterwarnings('ignore')
import os
import sys
import torch
import torch.distributed as dist

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from model.model_vlm import MiniMindVLM, VLMConfig
from dataset.lm_dataset import VLMDataset
import torch_npu
from torch_npu.npu import amp # 导入AMP模块
from torch_npu.contrib import transfer_to_npu # 使能自动迁移

# 日志打印函数
# 在分布式训练时只在主进程(rank=0)上打印日志
def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

# 余弦学习率调度器
# 在训练过程中逐渐降低学习率，最终降到初始值的1/10
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    # 使用交叉熵损失函数，reduction='none'以便后续通过mask处理填充token
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    current_max_norm = args.grad_clip
    for step, (X, Y, loss_mask, pixel_values) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        pixel_values = pixel_values.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X, pixel_values=pixel_values)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            if args.grad_dynamic:
                # 监控梯度范数
                current_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=10,  # 临时设大值以测量真实范数
                    error_if_nonfinite=True
                )
                
                # 动态调整逻辑：若梯度持续过大则收紧
                if current_norm > 5 * args.grad_clip: 
                    new_max_norm = args.grad_clip * 0.8  # 缩小20%
                elif current_norm < 0.2 * args.grad_clip:
                    new_max_norm = args.grad_clip * 1.2  # 扩大20%
                else:
                    new_max_norm = current_norm

                current_max_norm = new_max_norm
                # 应用裁剪（实际训练时）
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=new_max_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Pre-Train MOE:{} Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min grad_norm:{:.3f}'.format(
                    args.use_moe,
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    current_max_norm)
                )

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                           "grad_norm": current_max_norm
                           })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if model_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_vlm_{model_config.hidden_size}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half() for k, v in clean_state_dict.items()}  # 半精度保存
            torch.save(clean_state_dict, ckp)
            model.train()

# 初始化模型和分词器
def init_model(model_config: VLMConfig):
    # 加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained('../model', use_fast=True)
    moe_path = '_moe' if model_config.use_moe else ''
    # 加载纯语言模型权重
    ckp = f'{args.input_dir}/{args.llm_prefix}_{model_config.hidden_size}{moe_path}.pth'
    model = MiniMindVLM(model_config, vision_model_path="../model/vision_model/clip-vit-base-patch16")
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    if args.only_vision_proj:
        # 冻结除 vision_proj 外的所有参数
        for name, param in model.named_parameters():
            if 'vision_proj' not in name:
                param.requires_grad = False

    Logger(f'VLM可训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    _, preprocess = model.vision_encoder, model.processor
    return model.to(args.device), tokenizer, preprocess

# 初始化分布式训练环境
def init_distributed_mode():
    if not ddp: return
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V Pretrain")
    
    # 基础训练参数
    parser.add_argument("--input_dir", type=str, help="输入目录")
    parser.add_argument("--llm_prefix", type=str, default="full_sft", help="语言模型前缀")
    parser.add_argument("--out_dir", type=str, default="../out", help="输出目录")
    parser.add_argument("--epochs", type=int, default=4, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="学习率")
    parser.add_argument("--beta1", type=float, default=0.95, help="动量系数β₁")
    parser.add_argument("--beta2", type=float, default=0.999, help="动量系数β₂")
    parser.add_argument("--eps", type=float, default=1e-8, help="数值稳定项ε")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="解耦权重衰减系数λ")
    parser.add_argument('--amsgrad', default=False, type=bool, help="是否启用AMSGrad变体")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="训练精度")

    # 日志和监控参数
    parser.add_argument("--use_wandb", default=False, action="store_true", help="是否使用wandb记录训练过程")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V", help="wandb项目名称")
    # parser.add_argument("--wandb_host", type=str, default=None, help="WandB host，用于无交互环境自动登录")
    # parser.add_argument("--wandb_api_key", type=str, default=None, help="WandB API Key，用于无交互环境自动登录")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")

    # 分布式训练参数
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载进程数")

    # 优化器参数
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_dynamic", action="store_true", help="是否使用动态伸缩梯度裁剪值阈值")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")

    # 模型参数
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Transformer层数")
    parser.add_argument('--max_seq_len', default=640, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=False, type=bool, help="是否使用MoE")
    parser.add_argument('--only_vision_proj', default=True, type=bool, help="是否只训练视觉层")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_data.jsonl", help="训练数据路径")
    parser.add_argument("--images_path", type=str, default="../dataset/pretrain_images", help="训练数据路径")
    args = parser.parse_args()

    model_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                             max_seq_len=args.max_seq_len)
    max_seq_len = model_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.input_dir is None:
        args.input_dir = args.save_dir
    elif not args.input_dir.strip():
        args.input_dir = args.save_dir

    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    ts=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H:%M:%S")
    args.wandb_run_name = f"{ts}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置自动混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    rank = int(os.environ.get("RANK", -1))
    ddp = rank != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or rank == 0):
        import wandb
        # # 安全考虑，外部使用wandb login命令交互式登陆
        # # 优先使用命令行参数，其次环境变量
        # whost = args.wandb_host or os.environ.get("WANDB_HOST", None)
        # api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY", None)
        # if whost is not None and api_key is not None:
        #     # 自动登录，适用于无交互环境
        #     wandb.login(host=whost,key=api_key)
        # elif whost is not None:
        #     # 自动登录，适用于无交互环境
        #     wandb.login(host=whost)
        # elif api_key is not None:
        #     # 自动登录，适用于无交互环境
        #     wandb.login(key=api_key)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer, preprocess = init_model(model_config)

    train_ds = VLMDataset(args.data_path, args.images_path, tokenizer, preprocess=preprocess,
                          image_special_token=model_config.image_special_token,
                          max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),                 # 待优化参数 (必选)
        lr=args.learning_rate,              # 学习率 η (默认1e-3)
        betas=(args.beta1, args.beta2),    # 动量系数 (β₁, β₂)
        eps=args.eps,                       # 数值稳定项 ε (默认1e-8)
        weight_decay=args.weight_decay,     # 解耦权重衰减系数 λ (关键改进)
        amsgrad=args.amsgrad                # 是否启用AMSGrad变体
    )

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)

    if ddp:
        dist.destroy_process_group()