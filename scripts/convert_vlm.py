import argparse
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from model.model_vlm import MiniMindVLM, VLMConfig
warnings.filterwarnings('ignore', category=UserWarning)

def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.bfloat16):
    VLMConfig.register_for_auto_class()
    MiniMindVLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindVLM(lm_config, vision_model_path="../model/vision_model/clip-vit-base-patch16")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    del lm_model.vision_encoder
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    print(f"模型已保存为 Transformers-MiniMind-V 格式: {transformers_path}")

def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniMind-V transformers")
    parser.add_argument("--model_dir", type=str, default="../out")
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--out_dir", type=str, default="../MiniMind2-V")
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: Pretrain模型，1: SFT模型，2: SFT-多图模型 (beta拓展)")
    args = parser.parse_args()
    
    lm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)

    modes = {0: 'pretrain_vlm', 1: 'sft_vlm', 2: 'sft_vlm_multi'}
    torch_path = f"{args.model_dir}/{modes[args.model_mode]}_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"

    transformers_path = args.out_dir

    convert_torch2transformers_minimind(torch_path, transformers_path)
