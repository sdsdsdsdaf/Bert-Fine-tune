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
from datasets import load_dataset

ds = load_dataset("darssanle/GPT-4o-PHQ-9")
# 2) 구조 한눈에 확인
print(ds)                    # => DatasetDict(train: X examples)
print(ds["train"].features)  # => 컬럼 이름과 타입

# 3) 첫 번째 샘플 살펴보기
print(ds["train"][0])        # {'post_title': ..., 'post_text': ..., 'annotations': {...}}

# 4) 여러 개 미리보기
for row in ds["train"].select(range(3)):
    print(f"- {row['post_title'][:60]}...")

# 5) pandas DataFrame으로 변환해 테이블 형태로 보기 (작은 파일이므로 부담 없음)
df = ds["train"].to_pandas()
print(df.head())
