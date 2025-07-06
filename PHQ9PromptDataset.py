import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
    


class PHQ9PromptDataset(Dataset):
    """PHQ-9 총점(0–27) 회귀용 Dataset
       - CSV를 직접 읽어 문항 프롬프트 삽입
       - 토크나이저로 즉시 인코딩
    """
    PROMPTS = [
        "Little interest or pleasure in doing things?",
        "Feeling down, depressed, or hopeless?",
        "Trouble falling or staying asleep, or sleeping too much?",
        "Feeling tired or having little energy?",
        "Poor appetite or overeating?",
        "Feeling bad about yourself?",
        "Trouble concentrating on things?",
        "Moving or speaking so slowly or being fidgety?",
        "Thoughts that you would be better off dead?",
    ]

    def __init__(self, csv_path:str, model_name:str, max_len:int=256):
        super().__init__()
        self.tk = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

        df = pd.read_csv(csv_path)

        self.labels = df["PHQ-9 Score"].astype(float).tolist()
        self.texts  = self._build_texts(df)

        enc = self.tk(
            self.texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_attention_mask=True
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def _build_texts(self, df: pd.DataFrame):
        sep = f" {self.tk.sep_token} "
        texts = []
        for _, row in df.iterrows():
            parts = []
            for i in range(9):
                ans = str(row.iloc[1 + i])
                parts.append(f"Q{i+1}: {self.PROMPTS[i]} {self.tk.sep_token} A: {ans}")
            texts.append(sep.join(parts))
        return texts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }

if __name__ == "__main__":
    csv_path   = "phq9_dataset.csv"
    model_name = "mental/mental-bert-base-uncased"


    full_ds = PHQ9PromptDataset(csv_path, model_name)