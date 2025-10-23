import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns

# === 设置路径 ===
FEAT_DIR = "./output"
CLUSTER_NUM = 3

# === 加载 DINO 提取的 [CLS] 特征 ===
def load_features():
    cls_files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith("_cls.npy")])
    features = []
    names = []
    for f in cls_files:
        path = os.path.join(FEAT_DIR, f)
        features.append(np.load(path))
        names.append(f.replace("_cls.npy", ""))
    return np.stack(features), names

# === 可视化 t-SNE 降维 ===
def visualize_tsne(features, names, labels=None, title="t-SNE"):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="tab10")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1])
    for i, name in enumerate(names):
        plt.text(reduced[i, 0], reduced[i, 1], name, fontsize=8)
    plt.title(title)
    plt.show()

# === 聚类分析 ===
def cluster_and_plot(features, names):
    kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=0)
    clusters = kmeans.fit_predict(features)
    visualize_tsne(features, names, clusters, title="t-SNE + KMeans")
    return clusters

# === 模拟分类任务：我们手动打标签分类前10个为 0，其余为 1（示例）===
def run_prediction(features):
    y = np.array([0]*10 + [1]*(len(features)-10))  # 假设前10张图像属于类0，其余类1
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("=== 分类报告 ===")
    print(classification_report(y_test, y_pred))

# === 主程序 ===
if __name__ == "__main__":
    features, names = load_features()
    print(f"Loaded {len(features)} samples with dim {features.shape[1]}")
    cluster_and_plot(features, names)
    run_prediction(features)
