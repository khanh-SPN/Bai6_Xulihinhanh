# Dự án Phân cụm K-means trên Dữ liệu IRIS

## Giới thiệu
Dự án này thực hiện phân cụm K-means trên bộ dữ liệu IRIS với các số cụm khác nhau (2, 3, 4 và 5) và đánh giá chất lượng phân cụm dựa trên các chỉ số:
- F1-score
- RAND Index
- Normalized Mutual Information (NMI)
- Davies-Bouldin (DB) Index

Kết quả phân cụm và biểu đồ đánh giá sẽ được lưu dưới dạng ảnh trong thư mục `images`.

Cài đặt
Cài đặt các thư viện cần thiết bằng lệnh sau:
 pip install pandas scikit-learn matplotlib

Hướng dẫn sử dụng
Chạy dự án bằng cách thực thi main.py trong thư mục src:
python src/main.py

Sau khi chạy xong, các hình ảnh sau sẽ được lưu trong thư mục images:
clusters_2.png, clusters_3.png, clusters_4.png, và clusters_5.png: Kết quả phân cụm tương ứng với 2, 3, 4 và 5 cụm.
evaluation_metrics.png: Biểu đồ tổng hợp các chỉ số F1-score, RAND Index, NMI và DB Index cho từng số cụm.
Quy trình và Kết quả
Phân cụm K-means: Tập tin main.py thực hiện phân cụm dữ liệu IRIS với số lượng cụm khác nhau (2, 3, 4 và 5). Mỗi kết quả phân cụm được hiển thị dưới dạng biểu đồ phân tán (sử dụng PCA để giảm chiều về 2) và được lưu vào thư mục images.
Chỉ số đánh giá: Với mỗi số lượng cụm, chương trình tính toán và lưu lại các chỉ số đánh giá sau:
F1-score: Đánh giá độ chính xác của phân cụm so với nhãn thực tế.
RAND Index: Đánh giá sự tương đồng giữa các cụm và nhãn thực tế.
NMI: Đo lường thông tin tương hỗ giữa các cụm dự đoán và nhãn thực tế.
Davies-Bouldin Index: Đánh giá độ chặt chẽ và tách biệt của các cụm.
Mỗi chỉ số này được vẽ trên biểu đồ theo số lượng cụm và được lưu dưới dạng evaluation_metrics.png trong thư mục images.


## Cấu trúc Dự án

```plaintext
kmeans_clustering_project/
│
├── src/
│   └── main.py              # Tập tin chính để phân cụm và lưu kết quả
│
├── images/                  # Thư mục chứa các ảnh kết quả
│   ├── clusters_2.png       # Kết quả phân cụm với 2 cụm
│   ├── clusters_3.png       # Kết quả phân cụm với 3 cụm
│   ├── clusters_4.png       # Kết quả phân cụm với 4 cụm
│   ├── clusters_5.png       # Kết quả phân cụm với 5 cụm
│   └── evaluation_metrics.png # Biểu đồ các chỉ số đánh giá theo số cụm
│
└── README.md                # Tài liệu hướng dẫn dự án
