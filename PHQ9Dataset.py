import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
    

class PHQ9Dataset(Dataset):
    def __init__(self, data:pd.DataFrame, model_name:str, label_norm=False):
        super(PHQ9Dataset, self).__init__()

        if type(data) == str:
            data = pd.read_csv(data)
        if label_norm:
            label = [x / 27.0 for x in data["PHQ-9 Score"].astype(float).tolist()]
        else:
            label = data['PHQ-9 Score'].to_list()

        self.labels = label
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        sentences = []
        sep = self.tokenizer.sep_token  # "[SEP]"

        for i in range(len(label)):
            sentences.append((" " + sep + " ").join(data.iloc[i, 1:-2].tolist()))

        self.data = sentences
        self.max_len = 120

        print(sentences[0])


            
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
            "labels": torch.tensor(label, dtype=torch.float)
        }
    
if __name__ == "__main__":
    model_name = "mental/mental-bert-base-uncased"
    df = pd.read_csv("phq9_dataset.csv")

    dataset = PHQ9Dataset(df, model_name)
    # 절대 빈도
    print(df["PHQ-9 Score"].value_counts())

    # 상대 빈도(비율)
    print(df["PHQ-9 Score"].value_counts(normalize=True))

    print(dataset.labels)
    train_labels = dataset.labels
    

    import numpy as np
    labels = np.array(train_labels)
    print("레이블 최소–최대:", labels.min(), "~", labels.max())
    print("레이블 표준편차:", labels.std())