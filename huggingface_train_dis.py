from functools import partial
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, AutoModel,
        Trainer, TrainingArguments, DataCollatorWithPadding
    )
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from CustomTrainer import CustomTrainer
from loss import FocalLoss, combined_top10_loss
from PHQ9Dataset import PHQ9Dataset
from PHQ9PromptDataset import PHQ9PromptDataset
from model import PHQ9, PHQ9Distillation, PHQ9WithAttnPool, PHQ9DistillationWithAttnPool, PHQ9DistillationMeanPool
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup


"""
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    
    if preds.ndim == 2:
        preds = preds.squeeze(-1)

    preds  = np.asarray(preds).reshape(-1).astype(float)     # (N,)
    labels = np.asarray(labels).reshape(-1).astype(float)

    preds = np.clip(preds, 0, 27)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae  = mean_absolute_error(labels, preds)
    r2   = r2_score(labels, preds)
    return {
        "eval_rmse": rmse,
        "eval_mae": mae,
        "eval_r2": r2
    }
    """


class HuberTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # forward
        outputs = model(**inputs)

        if isinstance(outputs, dict):               # dict 형태
            logits = outputs["logits"]
        elif hasattr(outputs, "logits"):            # ModelOutput 형태
            logits = outputs.logits
        else:                                      # tuple 형태
            logits = outputs[0]

        logits = logits.squeeze(-1)                 # [B]로 변형

        loss_fct = torch.nn.SmoothL1Loss(beta=0.5)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


import wandb
model_str = "mental/mental-bert-base-uncased"
wandb.init(project="PHQ-Feat-Regression", name='6-feat-Attention-pool')

SEED = 42
epochs = 50
base_lr = 5e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
scheduler = None
unfreeze_layer_start = 11
stratrgy = 'epoch'
step = 1000
tokenizer = AutoTokenizer.from_pretrained(model_str)
loss_fn = combined_top10_loss

tokenizer = AutoTokenizer.from_pretrained(model_str)

model = PHQ9DistillationWithAttnPool(backbone=model_str, hidden_layer_list=[1024, 512, 256, 128, 64])

model.to(device)
print(model)

optimizer_grouped_parameters = []

for p in model.parameters():
    p.requires_grad = False
for param in model.regressor.parameters():
    param.requires_grad = True
for layer in model.classifier_model.bert.encoder.layer[-unfreeze_layer_start:]:
    for param in layer.parameters():
        param.requires_grad = True
for param in model.classifier_model.bert.pooler.parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params:,}")

loss_fn = torch.nn.MSELoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)

df = pd.read_csv("phq9_dataset.csv")
full_dataset = PHQ9Dataset("phq9_dataset.csv", model_name=model_str)

all_indices = list(range(len(full_dataset)))
all_labels = full_dataset.labels


train_idx, test_idx = train_test_split(
    all_indices,
    test_size=0.2,
    stratify=all_labels,
    random_state=SEED
)

train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

y_mean = np.array(all_labels)[train_idx].mean()
y_std  = np.array(all_labels)[train_idx].std()

train_dataset.dataset.labels = (train_dataset.dataset.labels - y_mean) / y_std
print(f"\n\nTrain_mean: {y_mean}, Train_std: {y_std}")

training_args = TrainingArguments(
    output_dir="./results",               # 모델 저장 경로
    per_device_train_batch_size=8,        # 학습 배치 사이즈
    per_device_eval_batch_size=8,         # 평가 배치 사이즈
    num_train_epochs=epochs,              # 학습 epoch 수
    weight_decay=0.01,                    # L2 정규화 계수
    logging_strategy=stratrgy,             # 로그 출력 주기 (ex: "steps", "epoch")
    eval_strategy=stratrgy,                # 평가 주기 (ex: "steps", "epoch")
    save_strategy=stratrgy,                # 모델 저장 주기
    load_best_model_at_end=True,          # 성능이 가장 좋은 모델 저장
    metric_for_best_model="eval_rmse",    # 가장 좋은 모델을 판단할 기준 (compute_metrics의 key)
    logging_dir="./logs",                 # 로그 저장 경로
    max_grad_norm=1.0,                    # 로그 출력 주기 (steps 단위)
    report_to=["wandb"],         
    run_name="phq9-regression",
)


def compute_metrics_fn(eval_pred, *, y_mean: float, y_std: float):
    preds, labels = eval_pred

    preds  = np.asarray(preds,  dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    preds_denorm  = preds  * y_std + y_mean
    labels_denorm = labels * y_std + y_mean

    rmse = np.sqrt(mean_squared_error(labels_denorm, preds_denorm))
    mae  = mean_absolute_error(labels_denorm, preds_denorm)
    r2   = r2_score(labels_denorm,  preds_denorm)
    return {"eval_rmse": rmse, "eval_mae": mae, "eval_r2": r2}

compute_metrics = partial(compute_metrics_fn, y_mean=y_mean, y_std=y_std)
collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

train_steps = (len(train_dataset) // training_args.per_device_train_batch_size) * training_args.num_train_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps = int(train_steps * 0.03),   # 전체의 3% 정도 워밍업
    num_training_steps = train_steps,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
    loss_fn=loss_fn,
    data_collator = lambda batch: collator(batch)
)


trainer.train()
metrics = trainer.evaluate()

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

loader = DataLoader(batch_size=8, dataset=test_dataset, collate_fn=collator)

all_label = []
all_pred = []
model.eval()
for batch in loader:

    input_ids, attn_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
    
    with torch.no_grad():
        output = model(input_ids, attn_mask)
        all_pred.append(output["logits"].cpu())
        all_label.append(labels.cpu())


y_true = torch.cat(all_label).flatten()
y_pred = torch.cat(all_pred).flatten()

print(model.log(input_ids, attn_mask))
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(y_true, y_pred, alpha=0.7)
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
ax.plot(lims, lims, "--")
ax.set_xlabel("Actual PHQ-9")
ax.set_ylabel("Predicted PHQ-9")
ax.set_title("PHQ-9: Predicted vs Actual")
plt.tight_layout()
plt.show()
        


    