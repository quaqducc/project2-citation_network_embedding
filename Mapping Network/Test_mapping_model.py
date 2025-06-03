import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# Model Definitions
# ==========================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # x: (B, 1, D)
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = residual + attn_output

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x

class QuestionMapper(nn.Module):
    def __init__(self, input_dim=1024, output_dim=128, depth=4, dropout=0.3, num_heads=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU()
        )
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(dim=output_dim, num_heads=num_heads, dropout=dropout) for _ in range(depth)
        ])
        self.final_proj = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)  # (batch, 1, output_dim)
        x = self.transformer_blocks(x)
        x = x.squeeze(1)
        x = self.final_proj(x)
        return x

# ==========================
# Utility Functions
# ==========================
def compute_text_embeddings(text_list, encoder, tokenizer, device, batch_size=64):
    all_embeds = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = encoder(**encoded_input, return_dict=True)
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeds.append(cls_embeds)
    return np.vstack(all_embeds)

def normalize(vectors):
    vectors = np.array(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)

# ==========================
# Configuration
# ==========================
MODEL_PATH = "/kaggle/input/mapping_graph_embedding/pytorch/default/1/Mapping_e5r.pt" #https://www.kaggle.com/models/quangduc3122004/mapping_graph_embedding
ENCODER_NAME = "intfloat/e5-large-v2"
CSV_PATH = "/kaggle/input/question-and-abstract/combined_with_abstract.csv"  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Load Model and Encoder
# ==========================
print("📦 Loading tokenizer and encoder...")
tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
encoder = AutoModel.from_pretrained(ENCODER_NAME).to(DEVICE)
encoder.eval()

print("📦 Loading trained QuestionMapper...")
model = QuestionMapper(input_dim=1024, output_dim=128).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ==========================
# Input Question
# ==========================
questions = ("How do the studies of neutrino oscillations in the Sudbury Neutrino Observatory (SNO), "
                  "the strong coupling dynamics of the standard Higgs sector, and the evolution of color exchange "
                  "in QCD hard scattering collectively contribute to advancing our understanding of fundamental "
                  "particle interactions and their implications for experimental observations at high-energy "
                  "facilities like the LHC?")
# ==========================
# Encode + Predict
# ==========================
print("🔍 Encoding input question...")
q_vec = compute_text_embeddings(questions, encoder, tokenizer, device=DEVICE)
q_vec = normalize(q_vec)
q_tensor = torch.tensor(q_vec, dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    pred_vec = model(q_tensor)
    pred_vec = F.normalize(pred_vec, dim=1).cpu().numpy()  # (1, 128)

# ==========================
# Load Citation Graph Embeddings
# ==========================
print("📂 Loading citation graph vectors...")
df = pd.read_csv(CSV_PATH)

paper_ids = df['id'].tolist()
graph_vecs = df[[str(i) for i in range(128)]].values
graph_vecs = normalize(graph_vecs)

# ==========================
# Compute Cosine Similarity
# ==========================
print("📏 Calculating similarity to graph vectors...")
cos_sim = cosine_similarity(pred_vec, graph_vecs)[0]  # shape: (N,)
top_k = 5
top_indices = np.argsort(cos_sim)[::-1][:top_k]

print("\n🏆 Top 5 most similar papers:")
for rank, idx in enumerate(top_indices, start=1):
    print(f"{rank}. Paper ID: {paper_ids[idx]} | Cosine Sim: {cos_sim[idx]:.4f}")v