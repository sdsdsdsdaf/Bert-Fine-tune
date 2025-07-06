from functools import partial
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from torch.amp import autocast, GradScaler
from PHQ9Dataset import sub_collate_fn
from loss import FocalLoss
from model import PHQ9



def train_one_epoch(model, loader, device, optimizer, loss_fn, use_amp=True):
    model.train()

    scaler = GradScaler(enabled=use_amp)
    total_loss = 0.0

    for (data, attn_mask), label in tqdm(loader, leave=False):
        data, attn_mask, label = data.to(device), attn_mask.to(device), label.to(device)
        optimizer.zero_grad()

        with autocast(enabled=use_amp, device_type=device):
            output = model(data, attn_mask)
            loss = loss_fn(output, label)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * data.size(0)

    return total_loss / len(loader.dataset)

def evaluate_cls(model, loader, device, loss_fn, use_amp=True):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():

        for (data, attn_mask), label in tqdm(loader, leave=False):
            data, attn_mask, label = data.to(device), attn_mask.to(device), label.to(device)

            with autocast(enabled=use_amp, device_type=device):
                output = model(data, attn_mask)
                loss   = loss_fn(output, label)

            total_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())

    avg_loss = total_loss / len(loader.dataset)

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    metric = {"loss": avg_loss, "macro_f1": macro_f1, "micro_f1": micro_f1, "weighted_f1": weighted_f1}

    return metric

def train(model, train_loader, epochs, device, optimizer, loss_fn, val_loader=None, use_amp=True, log=False, scheduler=None, task="cls"):

    assert task in ["cls", "reg"], "Task must be cls or reg"
    
    for E in range(epochs):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, loss_fn, use_amp=use_amp)
        print(f"[Epoch {E}]: Train Loss: {train_loss:.4f}")

        if scheduler is not None:
            scheduler.step()
            
        if val_loader is not None and log:
            if task == "cls":
                metric = evaluate_cls(model, val_loader, device, loss_fn, use_amp=use_amp)
                print(f"[Epoch {E}]: Val Loss: {metric['loss']:.4f}, Val Macro F1: {metric['macro_f1']:.4f}, Val Micro F1: {metric['micro_f1']:.4f}, Val Weighted F1: {metric['weighted_f1']:.4f}")
            if task == "reg":
                pass


if __name__ == "__main__":
    from transformers import (
        AutoTokenizer, AutoModel,
        Trainer, TrainingArguments, DataCollatorWithPadding
    )
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset, DataLoader

    from loss import FocalLoss
    from PHQ9Dataset import PHQ9Dataset
    from model import PHQ9
    import torch.optim as optim

    model_str = "mental/mental-bert-base-uncased"
    SEED = 42
    epochs = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = AutoModel.from_pretrained(model_str)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    model = PHQ9(backbone=backbone, task='regression')
    model.to(device)

    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True

    print(model)
    loss_fn = torch.nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-3,
        weight_decay=0.01)

    df = pd.read_csv("phq9_dataset.csv")
    full_dataset = PHQ9Dataset(df, model_name=model_str)

    all_indices = list(range(len(full_dataset)))
    all_labels = full_dataset.labels
    
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    collate_fn = partial(sub_collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

