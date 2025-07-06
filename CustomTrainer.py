from transformers import Trainer
import torch
import torch.nn.functional as F



class CustomTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_fn = loss_fn or torch.nn.SmoothL1Loss(beta=0.5)

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

        loss = self.loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
