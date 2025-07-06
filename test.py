import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from model import PHQ9WithAttnPool
from PHQ9Dataset import PHQ9Dataset

# ① .pt 파일 로드
hs = torch.load("hs.pt", map_location="cpu")     # hs: (N, H) 형태의 Tensor
model = PHQ9WithAttnPool()
arr = hs.detach().cpu().numpy()
print("Shape:", arr.shape)

arr2d = arr.mean(axis=1)   # (N, H)

sim = cosine_similarity(arr2d)  # (N, N)
# 대각선 제외한 평균
N = sim.shape[0]
avg_sim = (sim.sum() - N) / (N*(N-1))
print(f"Average cosine similarity (off-diagonal): {avg_sim:.4f}")

# ④ 분포 시각화 (히스토그램)
import matplotlib.pyplot as plt
# off-diagonal 값만 추출
off_diag = sim[np.triu_indices(N, k=1)]
plt.hist(off_diag, bins=50)
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Hidden State Similarity Distribution")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from PHQ9Dataset import PHQ9Dataset


# 1) 레이블 리스트 추출
#    Dataset 객체라면 train_dataset["labels"], 
#    아니면 list-of-dicts라면 아래처럼
dataset = PHQ9Dataset("phq9_dataset.csv", model_name="mental/mental-bert-base-uncased")
labels = dataset.labels

# 2) DataFrame으로 변환
df = pd.DataFrame(labels, columns=["label"])

# 3) 값별 개수 집계
counts = df["label"].value_counts().sort_index()
print("Label counts:\n", counts)

# 4) 히스토그램 시각화
plt.hist(labels, bins=28, range=(-0.5, 27.5))
plt.xticks(range(0, 28))       # x축 눈금을 0~27로
plt.xlabel("Label Value")
plt.ylabel("Frequency")
plt.title("Label Distribution (by Integer)")
plt.show()

