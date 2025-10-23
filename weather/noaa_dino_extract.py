import os
import io
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# 加载 DINO ViT-small 模型
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

# 将 CSV 转为热图图像
def csv_to_heatmap_image(csv_path):
    df = pd.read_csv(csv_path)
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE', 'HourlyDryBulbTemperature'])
    df['hour'] = df['DATE'].dt.hour
    df['day'] = df['DATE'].dt.day
    pivot = df.pivot(index='day', columns='hour', values='HourlyDryBulbTemperature')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap='coolwarm', cbar=False)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# 提取 DINO 特征
def extract_dino_features(image):
    with torch.no_grad():
        img_tensor = transform(image).unsqueeze(0)
        feats = dino.get_intermediate_layers(img_tensor, n=1)[0]
        cls = feats[:, 0].squeeze(0).numpy()
        patch = feats[:, 1:].squeeze(0).numpy()
        return cls, patch

# 主程序
def run(csv_path, output_dir):
    name = Path(csv_path).stem
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing: {name}")
    img = csv_to_heatmap_image(csv_path)
    img.save(f"{output_dir}/{name}.png")

    cls_feat, patch_feat = extract_dino_features(img)
    np.save(f"{output_dir}/{name}_cls.npy", cls_feat)
    np.save(f"{output_dir}/{name}_patch.npy", patch_feat)
    print("✅ Finished. Saved to", output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to NOAA .csv file")
    parser.add_argument("--out", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    run(args.csv, args.out)
