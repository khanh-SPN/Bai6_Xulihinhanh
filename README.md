# Dự án K-means Clustering cho Dữ liệu IRIS

## Giới thiệu
Dự án này thực hiện phân cụm K-means trên bộ dữ liệu IRIS và đánh giá chất lượng phân cụm dựa trên các chỉ số: F1 Score, RAND Index, Normalized Mutual Information (NMI), và Davies-Bouldin Index.

## Cấu trúc Dự án
- `data/`: Chứa dữ liệu IRIS.
- `src/`: Chứa mã nguồn chính.
  - `main.py`: File chính để chạy phân cụm và đánh giá.
  - `clustering.py`: Triển khai thuật toán K-means.
  - `evaluation.py`: Các hàm đánh giá.
- `results/`: Lưu kết quả phân cụm và đánh giá.
- `README.md`: Hướng dẫn sử dụng.

## Cài đặt
Cài đặt các thư viện cần thiết:
```bash
pip install pandas scikit-learn
