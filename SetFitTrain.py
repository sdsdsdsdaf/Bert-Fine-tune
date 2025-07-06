import numpy as np
import pandas as pd
from setfit import SetFitModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import torch
from setfit import Trainer, TrainingArguments
import wandb
from torch.utils.data import Subset
from PHQ9PromptDataset import PHQ9PromptDataset
from PHQ9SetFit import PHQ9SetFitDataset
from datasets import Dataset
from sklearn.linear_model import Ridge

def custom_rmse_metric(eval_dataset, pred_dataset):
    y_true = eval_dataset["label"]
    y_pred = pred_dataset["label"]  # SetFit 내부에서 predictions을 "label"로 넣어줌
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"rmse": rmse}

def subset_to_hf(ds, indices):
    # 샘플 하나를 확인해 보면 dict인지 튜플인지 알 수 있습니다
    sample = ds[indices[0]]
    if isinstance(sample, tuple):
        text_key, label_key = 0, 1
        get_text  = lambda item: item[text_key]
        get_label = lambda item: item[label_key]
    else:  # dict 또는 dataclass 라고 가정
        get_text  = lambda item: item["text"]
        get_label = lambda item: item["labels"]

    texts  = [get_text(ds[i])  for i in indices]
    labels = [get_label(ds[i]) for i in indices]

    return Dataset.from_dict({"text": texts, "label": labels})

wandb.init(project="PHQ9-Regression", name="phq9-regression")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SetFitModel.from_pretrained(
    "richie-ghost/setfit-mental-bert-base-uncased-MH-Topic-Check",
    device=device
)
SEED = 42
epochs = 1
scheduler = None
unfreeze_layer_start = 12

df = pd.read_csv("phq9_dataset.csv")
full_dataset = PHQ9SetFitDataset("phq9_dataset.csv", tokenizer=model.model_body.tokenizer)
all_indices = list(range(len(full_dataset)))
all_labels = full_dataset.labels

train_idx, test_idx = train_test_split(
    all_indices,
    test_size=0.2,
    stratify=all_labels,
    random_state=SEED
)

train_ds = subset_to_hf(full_dataset, train_idx)
test_ds  = subset_to_hf(full_dataset, test_idx)

args = TrainingArguments(
    num_epochs=(epochs, 20),          # (embeddings_epochs, classifier_epochs)
    batch_size=(32, 32),
    body_learning_rate=2e-5,
    head_learning_rate=1e-3,
    logging_strategy="steps",   # 손실도 100 step마다
    logging_steps=100,
    eval_strategy="steps",      # ★ 중간 평가 ON
    eval_steps=500, 
    max_steps=2000
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()


emb_train = model.model_body.encode(
    train_ds["text"], normalize_embeddings=True
)
emb_test  = model.model_body.encode(
    test_ds["text"],  normalize_embeddings=True
)

rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
)

rf.fit(emb_train, train_ds["label"])
model.head = rf 

from sklearn.metrics import mean_squared_error
pred = model.head.predict(emb_test)
rmse = np.sqrt(mean_squared_error(test_ds["label"], pred))
print(f"RMSE = {rmse:.3f}")