import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup


class Head(nn.Module):
    def __init__(self, in_feat, out_feat, hidden_layer_list= None, act=nn.GELU, use_norm=False, dropout_ratio=0.0):
        super(Head, self).__init__()
        if hidden_layer_list is None:
            hidden_layer_list = [2048, 2048, 1024]
        dims = [in_feat] + hidden_layer_list + [out_feat]
        layer = nn.ModuleList()
        
        for idx in range(len(dims)-1):
            layer.append(nn.Linear(dims[idx], dims[idx+1]))
            if idx < len(dims)-2:
                if use_norm:
                    layer.append(nn.LayerNorm(dims[idx+1]))
                layer.append(act())
                if dropout_ratio > 0:
                    layer.append(nn.Dropout(dropout_ratio))
                
                
        self.model = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.model(x)


class PHQ9(nn.Module):
    def __init__(self, backbone=None, hidden_size=None, task='classification', use_norm=False, dropout_ratio=0.0, hidden_layer_list = [256]):

        task = task.lower()
        assert task in ['classification', 'regression'], "Task must be classification or regression"
        result_size = {'classification': 28, 'regression': 1}
        
        super(PHQ9, self).__init__()
        self.bert = backbone
        self.hidden_size = hidden_size
        self.task = task

        if type(self.bert) == str:
            self.bert = AutoModel.from_pretrained(self.bert)
        
        if self.bert is None:
            self.bert = AutoModel.from_pretrained("mental/mental-bert-base-uncased")
        
        if self.hidden_size is None:
            self.hidden_size = self.bert.config.hidden_size * 2

        self.classifier = Head(self.hidden_size, result_size[task], use_norm=use_norm, dropout_ratio=dropout_ratio, hidden_layer_list=hidden_layer_list)

    def _int_weight(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.pooler_output  # [CLS] 토큰 표현
        mean = outputs.last_hidden_state.mean(dim=1)
        feat = torch.cat([cls_repr, mean], dim=-1)
        logits = self.classifier(feat).squeeze(-1)  # shape: (B,)

        if labels is not None:
            loss = F.mse_loss(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    

class PHQ9WithAttnPool(nn.Module):
    def __init__(self, backbone=None, hidden_size=None, task='classification', use_norm=False, dropout_ratio=0.0, hidden_layer_list = [256], log=False):

        task = task.lower()
        assert task in ['classification', 'regression'], "Task must be classification or regression"
        result_size = {'classification': 28, 'regression': 1}
        
        super(PHQ9WithAttnPool, self).__init__()
        self.bert = backbone
        self.hidden_size = hidden_size
        self.task = task

        if type(self.bert) == str:
            self.bert = AutoModel.from_pretrained(self.bert)
        
        if self.bert is None:
            self.bert = AutoModel.from_pretrained("mental/mental-bert-base-uncased")
        
        if self.hidden_size is None:
            self.hidden_size = self.bert.config.hidden_size

        self.classifier = Head(self.hidden_size, result_size[task], use_norm=use_norm, dropout_ratio=dropout_ratio, hidden_layer_list=hidden_layer_list)
        self.token_attn = nn.Linear(self.hidden_size, 1)

    def _int_weight(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        
        outs = self.bert(input_ids=input_ids,
                         attention_mask=attention_mask)
        hs = outs.last_hidden_state               # (B, L, H)
        
        scores = self.token_attn(hs).squeeze(-1)
        scores = scores.masked_fill(attention_mask==0, -1e9)
        weights = torch.softmax(scores, dim=1)    # (B, L)
        weights = weights.unsqueeze(-1)           # (B, L, 1)

        pooled = (hs * weights).sum(dim=1)        # (B, H)

        logits = self.classifier(pooled).squeeze(-1)  # (B,)

        if labels is not None:
            loss = F.mse_loss(logits, labels.float())
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
    @torch.no_grad()
    def inference(self, input_ids, attention_mask=None):
        was_training = self.training
        self.eval() 
        out = self.forward(input_ids, attention_mask, labels=None)

        if was_training:
            self.train()
        return out['logits']
    
    
    def log(self, input_ids, attention_mask=None, labels=None):
        outs = self.bert(input_ids=input_ids,
                         attention_mask=attention_mask)
        hs = outs.last_hidden_state
        torch.save(hs, "hs.pt")
        print(f"HS: {hs}")

class PHQ9DistillationMeanPool(nn.Module):
    def __init__(
        self,
        backbone: str,
        hidden_layer_list=[256],
        use_norm=False,
        dropout_ratio=0.0,
    ):
        super().__init__()
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(backbone)

        for p in self.classifier_model.parameters():
            p.requires_grad = False

        hidden_size = self.classifier_model.config.hidden_size
        num_classes = self.classifier_model.config.num_labels  # 6
        reg_in_dim = hidden_size * 2 + num_classes

        layers = []
        in_dim = reg_in_dim

        self.regressor = Head(
            in_dim,
            1,
            hidden_layer_list=hidden_layer_list,
            use_norm=use_norm,
            dropout_ratio=dropout_ratio,
        )


    def forward(self, input_ids, attention_mask=None, labels=None):
        device = input_ids.device

        # 1. BERT 본체 (pooler_output + last_hidden_state)
        base_out = self.classifier_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = base_out.pooler_output.to(device)              # (B, hidden_size)
        hs = base_out.last_hidden_state.to(device)              # (B, L, hidden_size)

        # 2. mean pooling with mask
        mask = attention_mask.unsqueeze(-1).expand(hs.size())   # (B, L, H)
        hs_masked = hs * mask                                   # padding 위치 0
        mean = hs_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B, H)

        # 3. classification logits
        cls_out = self.classifier_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        cls_logits = cls_out.logits.to(device)                  # (B, 6)
        cls_probs  = torch.softmax(cls_logits, dim=-1)          # (B, 6)

        # 4. 회귀 입력 결합: pooled + mean + probs
        feat = torch.cat([pooled, mean, cls_probs], dim=-1)     # (B, 2H+6)

        # 5. 회귀 예측
        preds = self.regressor(feat).squeeze(-1)                # (B,)

        # 6. loss
        if labels is not None:
            labels = labels.to(device).float()
            loss = F.mse_loss(preds, labels)
            return {"loss": loss, "logits": preds}
        return {"logits": preds}
        
    @torch.no_grad()
    def inference(self, input_ids, attention_mask=None):
        was_training = self.training
        self.eval() 
        out = self.forward(input_ids, attention_mask, labels=None)

        if was_training:
            self.train()
        return out['logits']
    
    
    def log(self, input_ids, attention_mask=None, labels=None):
        outs = self.classifier_model.bert(input_ids=input_ids,
                         attention_mask=attention_mask)
        hs = outs.last_hidden_state
        torch.save(hs, "hs.pt")
        print(f"HS: {hs}")

class PHQ9Distillation(nn.Module):
    def __init__(
        self,
        backbone: str,
        hidden_layer_list=[256],
        use_norm=False,
        dropout_ratio=0.0,
    ):
        super().__init__()
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(backbone)

        for p in self.classifier_model.parameters():
            p.requires_grad = False

        hidden_size = self.classifier_model.config.hidden_size
        num_classes = self.classifier_model.config.num_labels  # 6
        reg_in_dim = hidden_size + num_classes

        layers = []
        in_dim = reg_in_dim

        self.regressor = Head(
            in_dim,
            1,
            hidden_layer_list=hidden_layer_list,
            use_norm=use_norm,
            dropout_ratio=dropout_ratio,
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        base_out = self.classifier_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = base_out.pooler_output               # (B, hidden_size)

        cls_out = self.classifier_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_logits = cls_out.logits                   # (B, 6)
        cls_probs  = torch.softmax(cls_logits, dim=-1)# (B, 6)

        feat = torch.cat([pooled, cls_probs], dim=-1) # (B, hidden+6)

        preds = self.regressor(feat).squeeze(-1)      # (B,)

        if labels is not None:
            loss = F.mse_loss(preds, labels.float())
            return {"loss": loss, "logits": preds}
        return {"logits": preds}
    
    @torch.no_grad()
    def inference(self, input_ids, attention_mask=None):
        was_training = self.training
        self.eval() 
        out = self.forward(input_ids, attention_mask, labels=None)

        if was_training:
            self.train()
        return out['logits']
    
    
    def log(self, input_ids, attention_mask=None, labels=None):
        outs = self.classifier_model.bert(input_ids=input_ids,
                         attention_mask=attention_mask)
        hs = outs.last_hidden_state
        torch.save(hs, "hs.pt")
        print(f"HS: {hs}")


class PHQ9DistillationWithAttnPool(nn.Module):
    def __init__(
        self,
        backbone: str,
        hidden_layer_list=[256],
        use_norm=False,
        dropout_ratio=0.0,
    ):
        super().__init__()
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(backbone)

        for p in self.classifier_model.parameters():
            p.requires_grad = False

        hidden_size = self.classifier_model.config.hidden_size
        num_classes = self.classifier_model.config.num_labels  # 6
        reg_in_dim = hidden_size + num_classes

        in_dim = reg_in_dim

        self.regressor = Head(
            in_dim,
            1,
            hidden_layer_list=hidden_layer_list,
            use_norm=use_norm,
            dropout_ratio=dropout_ratio,
        )
        self.token_attn = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        base_out = self.classifier_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hs = base_out.last_hidden_state
        scores = self.token_attn(hs).squeeze(-1)
        scores = scores.masked_fill(attention_mask==0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)

        pooled = (hs * weights).sum(dim=1)        # (B, H)


        cls_out = self.classifier_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_logits = cls_out.logits                   # (B, 6)
        cls_probs  = torch.softmax(cls_logits, dim=-1)# (B, 6)
        # print(cls_probs.shape)
        feat = torch.cat([pooled, cls_probs], dim=-1) # (B, hidden+6)

        preds = self.regressor(feat).squeeze(-1)      # (B,)

        if labels is not None:
            loss = F.mse_loss(preds, labels.float())
            return {"loss": loss, "logits": preds}
        return {"logits": preds}
    
    @torch.no_grad()
    def inference(self, input_ids, attention_mask=None):
        was_training = self.training
        self.eval() 
        out = self.forward(input_ids, attention_mask, labels=None)

        if was_training:
            self.train()
        return out['logits']
    
    
    def log(self, input_ids, attention_mask=None, labels=None):
        outs = self.classifier_model.bert(input_ids=input_ids,
                         attention_mask=attention_mask)
        hs = outs.last_hidden_state
        torch.save(hs, "hs.pt")
        print(f"HS: {hs}")   



if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("YeRyeongLee/mental-bert-base-uncased-finetuned-0505")
    model = PHQ9Distillation("YeRyeongLee/mental-bert-base-uncased-finetuned-0505", use_norm=True, dropout_ratio=0.1)
    # model = AutoModelForSequenceClassification.from_pretrained("YeRyeongLee/mental-bert-base-uncased-finetuned-0505")
    print(model)