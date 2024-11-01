# main.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, rand_score, normalized_mutual_info_score, davies_bouldin_score

# Tạo thư mục 'images' nếu chưa tồn tại
os.makedirs("images", exist_ok=True)

# Tải dữ liệu IRIS
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y_true = data.target

# Sử dụng PCA để giảm số chiều về 2 cho việc vẽ biểu đồ
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Định nghĩa các số cụm cần thử nghiệm
n_clusters_options = [2, 3, 4, 5]

# Lưu các chỉ số đánh giá
f1_scores = []
rand_indices = []
nmi_scores = []
db_indices = []

# Vòng lặp qua từng số lượng cụm
for n_clusters in n_clusters_options:
    # Khởi tạo và huấn luyện mô hình KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    # Tính toán các chỉ số đánh giá
    f1 = f1_score(y_true, y_pred, average='macro')
    rand = rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    db = davies_bouldin_score(X, y_pred)
    
    # Thêm các chỉ số vào danh sách để vẽ biểu đồ
    f1_scores.append(f1)
    rand_indices.append(rand)
    nmi_scores.append(nmi)
    db_indices.append(db)
    
    # Vẽ biểu đồ phân cụm
    plt.figure()
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, cmap='viridis', s=50)
    plt.title(f"K-means với {n_clusters} cụm")
    plt.xlabel("Chiều dài đài hoa")
    plt.ylabel("Chiều rộng đài hoa")
    plt.savefig(f"images/clusters_{n_clusters}.png")
    plt.close()

# Vẽ biểu đồ các chỉ số đánh giá
plt.figure(figsize=(10, 6))

# Biểu đồ F1-score
plt.subplot(2, 2, 1)
plt.plot(n_clusters_options, f1_scores, marker='o', color='b')
plt.title("F1-score")
plt.xlabel("Số lượng cụm")
plt.ylabel("F1-score")

# Biểu đồ RAND index
plt.subplot(2, 2, 2)
plt.plot(n_clusters_options, rand_indices, marker='o', color='g')
plt.title("RAND Index")
plt.xlabel("Số lượng cụm")
plt.ylabel("RAND Index")

# Biểu đồ NMI
plt.subplot(2, 2, 3)
plt.plot(n_clusters_options, nmi_scores, marker='o', color='r')
plt.title("Normalized Mutual Information")
plt.xlabel("Số lượng cụm")
plt.ylabel("NMI")

# Biểu đồ Davies-Bouldin Index
plt.subplot(2, 2, 4)
plt.plot(n_clusters_options, db_indices, marker='o', color='m')
plt.title("Davies-Bouldin Index")
plt.xlabel("Số lượng cụm")
plt.ylabel("DB Index")

# Lưu biểu đồ các chỉ số đánh giá vào folder images
plt.tight_layout()
plt.savefig("images/evaluation_metrics.png")
plt.close()

print("Quá trình phân cụm hoàn tất và các biểu đồ đã được lưu vào thư mục 'images'.")
