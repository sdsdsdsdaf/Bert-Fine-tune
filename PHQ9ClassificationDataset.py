import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

print(torch.__version__)

import psutil, os
proc = psutil.Process(os.getpid())
print([dll.path for dll in proc.memory_maps() if "omp" in dll.path.lower()])    

class PHQ9ClassificationDataset(Dataset):
    def __init__(self, data:pd.DataFrame, model_name:str, label_norm=False):
        super(PHQ9ClassificationDataset, self).__init__()

        if type(data) == str:
            data = pd.read_csv(data)
        # Define the severity level mapping
        severity_mapping = {
            "None-minimal": 0,
            "Mild": 1,
            "Moderate": 2,
            "Moderately severe": 3,
            "Severe": 4
        }
        # Apply the mapping to create a new label_id column
        data["label_id"] = data["c"].map(severity_mapping)
        label = data['label_id'].to_list()

        self.labels = label
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        sentences = []
        sep = self.tokenizer.sep_token  # "[SEP]"

        for i in range(len(label)):
            sentences.append((" " + sep + " ").join(data.iloc[i, 1:-3].tolist()))

        self.data = sentences
        self.max_len = 120


            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors=None,
            add_special_tokens=True
        )
        return {
            "input_ids": enc["input_ids"],            # List[int]
            "attention_mask": enc["attention_mask"],  # List[int]
            "labels": torch.tensor(label, dtype=torch.long)
        }
    
if __name__ == "__main__":

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 중복 로드 허용
    os.environ["OMP_NUM_THREADS"] = "1" 

    model_name = "mental/mental-bert-base-uncased"
    df = pd.read_csv("phq9_dataset.csv")

    dataset = PHQ9ClassificationDataset(df, model_name)
    severity_mapping = {
            "None-minimal": 0,
            "Mild": 1,
            "Moderate": 2,
            "Moderately severe": 3,
            "Severe": 4
        }


    # 절대 빈도
    print(df["label_id"].value_counts())

    # 상대 빈도(비율)
    print(df["label_id"].value_counts(normalize=True))

    print(dataset.labels)
    train_labels = dataset.labels
    

    import numpy as np
    labels = np.array(train_labels)
    print("레이블 최소–최대:", labels.min(), "~", labels.max())
    print("fpdlmf vudrbs: ", labels.mean())
    print("레이블 표준편차:", labels.std())

    labels = dataset.labels

    # 2) DataFrame으로 변환
    df = pd.DataFrame(labels, columns=["label"])

    

    print(f"Input data: {dataset.data[0]}")

    # 3) 값별 개수 집계
    counts = df["label"].value_counts().sort_index()
    print("Label counts:\n", counts)

    # 4) 히스토그램 시각화
    plt.hist(labels, bins=5, range=(-0.5, 5))
    plt.xticks(range(0, 5))       # x축 눈금을 0~27로
    plt.xlabel("Label Value")
    plt.ylabel("Frequency")
    plt.title("Label Distribution (by Integer)")
    plt.show()