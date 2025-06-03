import os
import random
import pandas as pd
from tqdm import tqdm

# --- STEP 1: Load graph embeddings ---
embedding_df = pd.read_csv("graphsage_embeddings.csv")
embedding_df["node_id"] = embedding_df["node_id"].astype(str)
embedding_node_ids = set(embedding_df["node_id"])

# --- STEP 2: Lấy mẫu ngẫu nhiên các bài báo từ folder năm ---
BASE_DIR = "./"  # hoặc đường dẫn gốc thư mục dữ liệu nếu khác
YEARS = [str(y) for y in range(1992, 2004)]  # 1992–2003

paper_ids = []

# Duyệt từng năm
for year in YEARS:
    year_dir = os.path.join(BASE_DIR, year)
    if not os.path.isdir(year_dir):
        continue
    for fname in os.listdir(year_dir):
        if fname.endswith(".abs"):
            paper_id = fname.replace(".abs", "")
            if paper_id in embedding_node_ids:  # Kiểm tra tồn tại trong embeddings
                paper_ids.append((paper_id, year))

# Shuffle và chọn 1000 mẫu
random.seed(42)
random.shuffle(paper_ids)
selected_papers = paper_ids[:5000]

# --- STEP 3: Hàm lấy abstract ---
def get_abstract(paper_id: str, year: str, base_dir: str) -> str:
    abs_path = os.path.join(base_dir, year, f"{paper_id}.abs")
    if not os.path.isfile(abs_path):
        return ""
    try:
        with open(abs_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        parts = content.split("\\\\")
        return parts[2].strip() if len(parts) >= 3 else ""
    except:
        return ""

# --- STEP 4: Trích abstract và ghép với embedding ---
records = []
for paper_id, year in tqdm(selected_papers, desc="Processing"):
    abstract = get_abstract(paper_id, year, BASE_DIR)
    embedding_row = embedding_df[embedding_df["node_id"] == paper_id]
    if embedding_row.empty:
        continue  # an toàn: bỏ nếu không khớp
    embedding_values = embedding_row.iloc[0].drop("node_id").to_dict()
    record = {
        "node_id": paper_id,
        "year": year,
        "abstract": abstract,
        **embedding_values
    }
    records.append(record)

# --- STEP 5: Xuất ra file ---
df_final = pd.DataFrame(records)
df_final.to_csv("graph_sample_5000_with_abstract.csv", index=False)
print(f"✅ Saved {len(df_final)} records to graph_sample_1000_with_abstract.csv")
