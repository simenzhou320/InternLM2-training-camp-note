# 使用 huggingface_hub 下载 InternLM2-Chat-7B 的 config.json 文件到本地
from huggingface_hub import hf_hub_download

# 指定模型和文件的名称
model_id = "internlm/internlm2-chat-7b"
filename = "config.json"

# 使用 hf_hub_download 函数下载文件
local_path = hf_hub_download(repo_id=model_id, filename=filename, local_dir='./')

print(f"文件已下载到：{local_path}")