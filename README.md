# Hướng dẫn Tạo Dữ Liệu Tổng hợp với GPT-OSS-120B

Dự án này cung cấp công cụ để tạo dữ liệu tổng hợp quy mô lớn sử dụng mô hình GPT-OSS-120B thông qua API.

## 🚀 Tính năng

- Tạo dữ liệu tổng hợp từ prompts có sẵn
- Hỗ trợ kết nối API GPT-OSS-120B
- Xử lý song song với nhiều instances
- Theo dõi tiến trình với WandB
- Lưu checkpoint định kỳ để tránh mất dữ liệu
- Tự động upload lên Hugging Face Hub

## 📁 Cấu trúc thư mục

```
synthetic_data_generation/
│
├── config/
│   └── gpt_oss_config.yaml          # File cấu hình API
│
├── scripts/
│   ├── run_generation.sh            # Script chạy generation
│   └── slurm/
│       └── submit_slurm_job.sh      # Script cho HPC Slurm
│
├── utils/
│   ├── __init__.py
│   ├── token_counter.py             # Tiện ích đếm token
│   └── data_processor.py            # Xử lý dữ liệu
│
├── generate_synthetic_gpt_oss.py    # Script chính
├── requirements.txt                 # dependencies
└── README.md                        # Hướng dẫn này
```

## ⚙️ Cài đặt

### 1. Clone và thiết lập môi trường

```bash
# Tạo thư mục dự án
mkdir synthetic_data_generation
cd synthetic_data_generation

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Thiết lập biến môi trường

Tạo file `.env` hoặc export các biến môi trường:

```bash
# API Keys
export API_KEY="your-gpt-oss-api-key-here"
export HF_TOKEN="your-huggingface-token-here"
export WANDB_API_KEY="your-wandb-api-key-here"

# Cấu hình đường dẫn
export CHECKPOINT_PATH="./synthetic_data"
export CONFIG_FILE="config/gpt_oss_config.yaml"
```

### 3. Cấu hình API endpoint

Chỉnh sửa file `config/gpt_oss_config.yaml`:

```yaml
model: "gpt-oss-120b"
api_base: "https://your-actual-gpt-oss-endpoint.com/v1"  # Thay bằng endpoint thực tế
api_key: "${API_KEY}"  # Sử dụng biến môi trường
instances: 2
per_instance_max_parallel_requests: 8
request_timeout: 120
max_retries: 6
retry_delay: 4
```

## 🏃‍♂️ Chạy Generation

### Cách 1: Sử dụng script (khuyến nghị)

```bash
bash scripts/run_generation.sh
```

### Cách 2: Chạy trực tiếp với Python

```bash
python generate_synthetic_gpt_oss.py \
    --prompts_dataset "HuggingFaceTB/cosmopedia-100k" \
    --prompt_column "prompt" \
    --max_samples 5000 \
    --checkpoint_interval 500 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.1 \
    --repo_id "your-username/synthetic_data_gpt_oss" \
    --wandb_username "your-wandb-username" \
    --config_file "config/gpt_oss_config.yaml"
```

### Cách 3: Chạy trên Slurm cluster

```bash
bash scripts/slurm/submit_slurm_job.sh
```

## 📊 Các tham số quan trọng

| Tham số | Mô tả | Giá trị mặc định |
|---------|-------|------------------|
| `--max_samples` | Số lượng mẫu cần generate (-1 để generate toàn bộ) | 1000 |
| `--max_new_tokens` | Số token tối đa cho mỗi generation | 2048 |
| `--temperature` | Độ sáng tạo của model | 0.7 |
| `--top_p` | Top-p sampling | 0.9 |
| `--checkpoint_interval` | Lưu checkpoint sau mỗi N mẫu | 100 |
| `--instances` | Số instances chạy song song | 2 |
| `--per_instance_max_parallel_requests` | Số request song song mỗi instance | 8 |

## 👀 Theo dõi tiến trình

### WandB Dashboard
Truy cập [wandb.ai](https://wandb.ai) để theo dõi:
- Tốc độ generation (tokens/giây)
- Tỷ lệ thành công/thất bại
- Chất lượng generation
- Sử dụng tài nguyên

### Checkpoint files
Dữ liệu được lưu tự động tại: `./synthetic_data/your-dataset-name/data/`

## 📦 Output

Dữ liệu được generate sẽ bao gồm:
- `prompt`: Prompt gốc
- `completion`: Văn bản được generate
- `token_length`: Độ dài token
- `model`: Tên model được sử dụng

Sau khi hoàn thành, dữ liệu sẽ được tự động upload lên Hugging Face Hub.

## 🛠️ Tùy chỉnh

### Thêm định dạng dữ liệu mới

Chỉnh sửa `utils/data_processor.py` để thêm định dạng output mới:

```python
def format_textbook(sample):
    """Định dạng cho textbook"""
    return f"# {sample['title']}\n\n{sample['content']}"

def format_qa(sample):
    """Định dạng cho Q&A"""
    return f"Q: {sample['question']}\nA: {sample['answer']}"
```

### Điều chỉnh prompt template

Sửa đổi hàm `process_text` trong script chính để thay đổi template prompt.

## ❓ Xử lý sự cố

### Lỗi kết nối API
- Kiểm tra API key và endpoint
- Đảm bảo network có thể truy cập endpoint
- Kiểm tra rate limits của API

### Lỗi memory
- Giảm `--per_instance_max_parallel_requests`
- Giảm `--max_new_tokens`
- Tăng `--checkpoint_interval`

### Lỗi timeout
- Tăng `--request_timeout` trong config
- Kiểm tra network stability

## 📈 Best Practices

1. **Bắt đầu nhỏ**: Chạy thử với `--max_samples 100` trước
2. **Theo dõi cost**: Giám sát token usage để kiểm soát chi phí
3. **Backup thường xuyên**: Sử dụng checkpoint để tránh mất dữ liệu
4. **Validate chất lượng**: Kiểm tra ngẫu nhiên các samples được generate

## 📝 License

Dự án này sử dụng MIT License.

## 🤝 Đóng góp

Đóng góp bằng cách:
1. Report bugs qua Issues
2. Gửi Pull Requests
3. Cải thiện documentation
4. Chia sẻ use cases và feedback



*Lưu ý: Đảm bảo bạn có quyền sử dụng API GPT-OSS-120B và tuân thủ các điều khoản sử dụng của nhà cung cấp.*