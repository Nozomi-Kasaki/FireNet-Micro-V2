import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import argparse
import numpy as np
import copy
import os
from models.model import FireNetMicroV2
from dataset import get_train_transforms, get_val_transforms
from utils import seed_everything, get_class_weights, IDX_TO_CLASS, CLASS_MAP
# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        # 如果输入是元组，提取分类输出
        if isinstance(inputs, tuple):
            inputs = inputs[0] # 假设分类输出是第一个元素
      
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()
# --- Mixup 工具 ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# --- 参数配置 ---
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./database')
parser.add_argument('--val_dir', type=str, default='./test')
parser.add_argument('--epochs', type=int, default=300, help='Total epochs')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-3, help='Initial learning rate')
parser.add_argument('--val_interval', type=int, default=10, help='Initial validation interval (steps)')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def validate(model, val_loader, criterion, prefix="Val"):
    """
    通用验证函数
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * 3
    class_total = [0] * 3
  
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
          
            # 如果输出是元组，提取分类输出
            if isinstance(outputs, tuple):
                outputs = outputs[0]
              
            loss = criterion(outputs, labels)
            total_loss += loss.item()
          
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
          
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item() if c.ndim > 0 else c.item()
                class_total[label] += 1
              
    acc = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
  
    # 打印精简信息
    print(f" [{prefix}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | ", end='')
    cls_names = ['Fire', 'No', 'Start'] # 确保顺序正确
    for i in range(3):
        if class_total[i] > 0:
            cls_acc = 100 * class_correct[i] / class_total[i]
            print(f"{cls_names[i]}: {cls_acc:.1f}% ", end='')
    print("")
  
    return acc
def main():
    seed_everything(42)
    save_dir = "best_c"
    os.makedirs(save_dir, exist_ok=True)
    print(f"模型将保存到: {save_dir}/")
  
    # 1. 数据准备
    train_dataset = datasets.ImageFolder(args.data_dir, transform=get_train_transforms())
    val_dataset = datasets.ImageFolder(args.val_dir, transform=get_val_transforms())
  
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
  
    # 2. 权重计算 (针对 No_fire 手动加权逻辑建议保留)
    base_weights = get_class_weights(train_dataset)
    print(f"Base Weights: {base_weights}")
  
    # 如果需要手动加权 no_fire (index 1)
    if 'no_fire' in train_dataset.class_to_idx:
        no_fire_idx = train_dataset.class_to_idx['no_fire']
        base_weights[no_fire_idx] *= 1.3
        print(f"Adjusted Weights (Boost No_Fire): {base_weights}")
  
    class_weights = base_weights.to(device)
    # 3. 模型初始化
    print("Initializing FireNetMicroV2...") # Corrected print statement
    model = FireNetMicroV2().to(device)
  
    # 4. 损失与优化
    criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
  
    # Cosine Schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
  
    # 5. 训练循环
    best_acc = 0.0
    global_step = 0
    high_alert_mode = False # 高能模式标志
    current_val_interval = args.val_interval
    print(f"Start Training: {args.epochs} Epochs. Mixup OFF after epoch {int(args.epochs*0.8)}.")
  
    for epoch in range(args.epochs):
        model.train()
      
        # Mixup Cooldown 逻辑
        use_mixup = True if epoch < int(args.epochs * 0.8) else False
      
        # 打印 Epoch 头部信息
        if epoch % 5 == 0 or high_alert_mode:
            status = "HIGH ALERT (Eval every step)" if high_alert_mode else f"Normal (Eval every {current_val_interval} steps)"
            print(f"\n--- Epoch {epoch+1}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.6f} | Mixup: {use_mixup} | Mode: {status} ---")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
              
                # --- Forward (Mixup or Normal) ---
            if use_mixup and np.random.rand() > 0.5:
                inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0, device=device)
                outputs = model(inputs_mixed)
                  
                    # 如果输出是元组，提取分类输出
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                      
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(inputs)
                  
                    # 如果输出是元组，提取分类输出
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                      
                loss = criterion(outputs, labels)
              
            loss.backward()
            optimizer.step()
          
            global_step += 1
          
            # --- 验证逻辑 ---
            # 如果进入高能模式，step % 1 == 0 (即每次都验证)
            # 否则按 current_val_interval 验证
            if global_step % current_val_interval == 0:
                val_acc = validate(model, val_loader, criterion, prefix=f"Step {global_step}")
                if val_acc > 87.0:
                    ckpt_name = f"model_acc{val_acc:.2f}_ep{epoch+1}_step{global_step}.pth"
                    save_path = os.path.join(save_dir, ckpt_name)
                    torch.save(model.state_dict(), save_path)
                    print(f" [Saved] {ckpt_name} (Acc > 87%)")
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_fire_model.pth"))
                    print(f" >>> 🏆 New Best Model! Acc: {best_acc:.2f}%")
                  
                    # 触发高能模式逻辑
                    if best_acc > 86.5 and not high_alert_mode:
                        high_alert_mode = True
                        current_val_interval = 1 # 强制改为每个 step 都验证
                        print("\n!!! 🚀 突破 86.5%！进入高能模式：每个 Step 都将进行评估以捕捉最佳模型！ !!!\n")
              
                model.train() # 切回训练模式
        # --- Epoch 结束: 更新 LR ---
        scheduler.step()
    print(f"Training Finished!")
    print(f"Best Acc: {best_acc:.2f}% (Saved to {save_dir}/best_fire_model.pth)")
if __name__ == '__main__':
    main()
