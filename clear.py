import os
import hashlib
from pathlib import Path

def calculate_md5(file_path):
    """计算文件的 MD5 哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def clean_duplicates(root_dir):
    """
    遍历目录，保留唯一的哈希值文件，删除重复项。
    返回：(保留数量, 删除数量)
    """
    unique_hashes = set()
    kept_files = 0
    deleted_files = 0
    
    # 遍历所有子文件夹 (fire, no_fire, start_fire)
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            # 过滤非图片文件 (根据需要调整后缀)
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
                
            file_path = os.path.join(root, filename)
            file_hash = calculate_md5(file_path)
            
            if file_hash in unique_hashes:
                # 发现重复，删除
                print(f"[Duplicate Removed] {file_path}")
                os.remove(file_path)
                deleted_files += 1
            else:
                # 记录新哈希
                unique_hashes.add(file_hash)
                kept_files += 1
                
    return kept_files, deleted_files

# --- 使用方法 ---
# 将下面的路径替换为你解压后的数据集路径
# 建议对 DB1, DB2, DB3 分别运行，或者合并后运行
dataset_path = "FIRE_DATABASE_1" 
kept, deleted = clean_duplicates(dataset_path)

print(f"处理完成: 保留了 {kept} 张图片, 删除了 {deleted} 张重复图片。")