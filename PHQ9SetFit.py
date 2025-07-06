from torch.utils.data import Dataset
import pandas as pd
import torch


class PHQ9SetFitDataset(Dataset):
    def __init__(self, data:pd.DataFrame, tokenizer, label_norm=False):
        super(PHQ9SetFitDataset, self).__init__()

        if type(data) == str:
            data = pd.read_csv(data)
        if label_norm:
            label = [x / 27.0 for x in data["PHQ-9 Score"].astype(float).tolist()]
        else:
            label = data['PHQ-9 Score'].to_list()

        self.labels = label
        self.tokenizer = tokenizer
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

        
        return {
            "text": text,            # List[int]
            "labels": torch.tensor(label, dtype=torch.float)
        }