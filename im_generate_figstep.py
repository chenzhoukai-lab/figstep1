import os
import pandas as pd
from src.im_generate_prompts import gen_query, QueryType

# 500样本完整版数据集
DATA_PATH = "/inspire/hdd/global_user/wenming-253108090054/czk/FigStep/data/question/SafeBench-Tiny.csv"
SAVE_DIR = "dataset_m"
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
questions = df["question"].tolist()

print("开始生成 改进版FigStep 攻击素材（随机双Prompt）...")
for i, q in enumerate(questions):
    text_prompt, image_prompt = gen_query(QueryType.figstep, question=q, instruction=q)
    
    with open(f"{SAVE_DIR}/prompt_{i}.txt", "w", encoding="utf-8") as f:
        f.write(text_prompt)
    image_prompt.save(f"{SAVE_DIR}/image_{i}.png")

print(f"✅ 生成完成！共 {len(questions)} 个攻击素材")