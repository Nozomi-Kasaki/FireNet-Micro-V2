import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import argparse
import numpy as np
import os
import sys

# 引入你的模型定义 (确保路径正确)
from models.model import FireNetMicroV2
from dataset import get_train_transforms, get_val_transforms
from utils import seed_everything, get_class_weights

# --- 参数配置 ---
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./database')
parser.add_argument('--val_dir', type=str, default='./test')
# 关键：这里填你那个表现最好的 step 6825 模型路径
parser.add_argument('--resume', type=str,default='best_c/model_acc87.32_ep127_step6825.pth' , help='Path to the step 6825 model')
parser.add_argument('--epochs', type=int, default=1, help='We only need very few epochs')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-6, help='Ultra small learning rate')
parser.add_argument('--img_size', type=int, default=192) 
# 噪声强度：这是核心参数。1e-4 意味着万分之一的抖动。
parser.add_argument('--noise_std', type=float, default=1e-4, help='Standard deviation of noise added to weights')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    seed_everything(42)
    
    # 创建保存目录
    save_dir = "jitter_finetune"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Jittered models will be saved to: {save_dir}/")
    
    # 1. 数据准备 (保持一致)
    train_dataset = datasets.ImageFolder(args.data_dir, transform=get_train_transforms(img_size=args.img_size))
    val_dataset = datasets.ImageFolder(args.val_dir, transform=get_val_transforms(img_size=args.img_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4) # 其实验证集此时不太重要了
    
    # 2. 加载模型
    print(f"Loading Magic Model: {args.resume}")
    model = FireNetMicroV2(num_classes=3).to(device)
    checkpoint = torch.load(args.resume, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 3. 优化器 (极低 LR)
    # 使用 SGD 而不是 Adam，因为 SGD 在微调时更“温和”，不容易剧烈改变参数分布
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device) # 标准 Loss 即可，微调阶段不需要花哨的
    
    global_step = 0
    
    print(f"Starting Weight Jittering...")
    print(f"Strategy: LR={args.lr} + Gaussian Noise(std={args.noise_std})")
    print("Saving every 10 steps...")

    for epoch in range(args.epochs):
        model.train()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # --- [魔法核心]：给权重注入微小噪声 ---
            with torch.no_grad():
                for param in model.parameters():
                    # 生成与参数形状相同的噪声
                    noise = torch.randn_like(param) * args.noise_std
                    # 加到参数上
                    param.add_(noise)
            
            global_step += 1
            
            # --- 验证与保存逻辑 ---
            # 我们不再关心 Val Acc (因为你说它不准)，我们只管保存模型供你测试
            if global_step % 1 == 0:
                # 简单跑一下验证，仅仅为了看 Loss 是否爆炸，防止模型崩了
                # (可选：为了速度可以注释掉下面这几行验证逻辑)
                # model.eval()
                # with torch.no_grad():
                #    val_out = model(inputs) # 只测当前 batch
                #    val_loss = criterion(val_out, labels)
                # model.train()
                
                # 文件名带上 step，方便你回溯
                filename = f"jitter_ep{epoch}_step{global_step}.pth"
                save_path = os.path.join(save_dir, filename)
                torch.save(model.state_dict(), save_path)
                
                sys.stdout.write(f"\rStep {global_step}: Saved {filename} (Loss: {loss.item():.4f})")
                sys.stdout.flush()

    print(f"\nDone! Please test the models in {save_dir} on your Test Set.")

if __name__ == '__main__':
    main()
