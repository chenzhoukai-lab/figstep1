import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
# Qwen2.5-VL 专用加载类
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# ===================== 你的本地模型路径（无需修改） =====================
LOCAL_MODEL_PATH = "/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-3B-Instruct"
# =========================================================================

# 加载 Qwen2.5-VL 模型
print("加载本地 Qwen2.5-VL-3B 模型...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

# 加载路径
PROMPT_DIR = "/inspire/hdd/global_user/wenming-253108090054/czk/FigStep/dataset_m"
df = pd.read_csv("/inspire/hdd/global_user/wenming-253108090054/czk/FigStep/data/question/safebench.csv")
results = []

print("FigStep 攻击推理...")
for i in tqdm(range(len(df))):
    image = Image.open(f"{PROMPT_DIR}/image_{i}.png").convert("RGB")
    with open(f"{PROMPT_DIR}/prompt_{i}.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Qwen2.5-VL 官方输入格式
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text}
        ]}
    ]
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=False
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    results.append({
        "index": i,
        "question": df.iloc[i]["question"],
        "response": response
    })

pd.DataFrame(results).to_csv("figstep_qwen25vl_results.csv", index=False, encoding="utf-8")
print("✅ 推理完成！结果保存在 figstep_qwen25vl_results.csv")