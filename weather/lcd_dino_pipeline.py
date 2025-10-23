import os
import io
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# 加载 DINO ViT-Small 模型
dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
dino.eval()

# 图像预处理
transform = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

# 模块 1：CSV → 热图图像
def csv_to_heatmap(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.dropna(subset=['DATE', 'HourlyDryBulbTemperature'])
        df['hour'] = df['DATE'].dt.hour
        df['day'] = df['DATE'].dt.day
        pivot = df.pivot(index='day', columns='hour', values='HourlyDryBulbTemperature')

        # 可视化热图
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, cmap='coolwarm', cbar=False)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    except Exception as e:
        print(f"[Error in {csv_path}]: {e}")
        return None

# 模块 2+3：图像 → 特征
def extract_dino_features(img):
    with torch.no_grad():
        img_tensor = transform(img).unsqueeze(0)
        feats = dino.get_intermediate_layers(img_tensor, n=1)[0]
        cls_feat = feats[:, 0].squeeze(0).numpy()
        patch_feat = feats[:, 1:].squeeze(0).numpy()
        return cls_feat, patch_feat

# 主函数
def run_pipeline(csv_dir, img_dir, feat_dir):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    csv_paths = list(Path(csv_dir).glob("*.csv"))
    print(f"Found {len(csv_paths)} CSV files.")

    for csv_path in csv_paths:
        name = csv_path.stem
        print(f"Processing: {name}")
        img = csv_to_heatmap(csv_path)
        if img is None:
            continue
        img.save(f"{img_dir}/{name}.png")
        cls_feat, patch_feat = extract_dino_features(img)
        np.save(f"{feat_dir}/{name}_cls.npy", cls_feat)
        np.save(f"{feat_dir}/{name}_patch.npy", patch_feat)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DINO feature extraction on LCD data")
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory containing CSV files")
    parser.add_argument("--img_dir", type=str, required=False, help="Output directory for PNG images")
    parser.add_argument("--feat_dir", type=str, required=False, help="Output directory for .npy features")
    args = parser.parse_args()

    # 如果未提供，使用默认子目录
    img_dir = args.img_dir or os.path.join(args.csv_dir, "images")
    feat_dir = args.feat_dir or os.path.join(args.csv_dir, "features")

    run_pipeline(args.csv_dir, img_dir, feat_dir)

