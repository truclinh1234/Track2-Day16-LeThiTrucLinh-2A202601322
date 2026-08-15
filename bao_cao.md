# Báo cáo Lab 16 — LightGBM trên AWS CPU Instance

Hạ tầng triển khai bằng Terraform trên AWS (`us-east-1`), Compute Node dùng `t3.micro` (thay cho `t3.medium` mặc định do tài khoản bị giới hạn Free Tier, chỉ 1 vCPU khả dụng thực tế/1GB RAM). Dataset Credit Card Fraud Detection (284,807 dòng, 30 feature) load trong 2.51 giây, huấn luyện `LGBMClassifier` (300 cây, không dùng early stopping do tập validation quá ít mẫu fraud gây nhiễu) mất 11.05 giây — khá nhanh dù chạy trên instance nhỏ.

Model đạt AUC-ROC 0.893, cho thấy khả năng phân biệt fraud/non-fraud tốt dù chỉ dùng tham số mặc định, chưa tinh chỉnh sâu. Accuracy 99.09% nhìn có vẻ ấn tượng nhưng không phản ánh đúng chất lượng model do dataset cực kỳ mất cân bằng (chỉ 0.17% là fraud) — chỉ số đáng tin cậy hơn là recall 81.6% và precision 13.8%: model bắt được phần lớn giao dịch gian lận thật nhưng đánh đổi bằng khá nhiều cảnh báo giả, hệ quả trực tiếp của việc dùng `scale_pos_weight` cao (~577) để ưu tiên không bỏ sót fraud.

Về tốc độ inference, latency cho 1 dòng chỉ 1.23ms và throughput đạt ~58,000 dòng/giây khi dự đoán batch 1000 dòng — cho thấy dù chạy trên CPU instance nhỏ, LightGBM vẫn đủ nhanh để phục vụ inference thời gian thực cho các ứng dụng phát hiện gian lận quy mô vừa.
