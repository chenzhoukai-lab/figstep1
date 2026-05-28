import os
import pandas as pd
from src.im2_generate_prompts import gen_query, QueryType

# 官方数据集路径（仓库自带）
DATA_PATH = "/inspire/hdd/global_user/wenming-253108090054/czk/FigStep/data/question/safebench.csv"
# 提示保存路径
SAVE_DIR = "/inspire/hdd/global_user/wenming-253108090054/czk/FigStep/dataset_m"
os.makedirs(SAVE_DIR, exist_ok=True)

# 加载数据
df = pd.read_csv(DATA_PATH)
questions = df["question"].tolist()

print("开始生成 FigStep 视觉提示...")
for i, q in enumerate(questions):
    # 调用改进后的核心函数，生成攻击版FigStep提示
    text_prompt, image_prompt = gen_query(QueryType.figstep, question=q, instruction=q)
    
    # 保存文本提示
    with open(f"{SAVE_DIR}/prompt_{i}.txt", "w", encoding="utf-8") as f:
        f.write(text_prompt)
    # 保存图片提示
    image_prompt.save(f"{SAVE_DIR}/image_{i}.png")

print(f"✅ 生成完成！共 {len(questions)} 个提示，保存在 {SAVE_DIR}")