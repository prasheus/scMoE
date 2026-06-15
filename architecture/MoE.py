import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import pandas as pd 

# sklearn preprocessing


# ENCODER
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)


# PROJECTION HEADS
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)


# Cancer Classifier
class CancerTypeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)   # raw logits


# EXPERTS
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)   # single logit
        )
    def forward(self, x):
        return self.net(x)   # raw logit


# MoE
class StaticMoE(nn.Module):
    def __init__(self, input_dim, num_experts, num_cancer_types, 
                 hidden_dim=128, proj_dim=32, expert_hidden=64):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.proj_ctc = ProjectionHead(hidden_dim, proj_dim)
        self.proj_exp = ProjectionHead(hidden_dim, proj_dim)
        self.classifier = CancerTypeClassifier(proj_dim, num_cancer_types)
        # list of experts
        self.experts = nn.ModuleList([
            Expert(proj_dim, expert_hidden) for _ in range(num_experts)
        ])
        self.num_experts = num_experts

    def forward(self, x, true_ct=None, p_true=1.0):
        # shared embed
        h = self.encoder(x)

        # heads
        h_ctc = self.proj_ctc(h)
        h_exp = self.proj_exp(h)

        # cacner type classifier
        ctc_logits = self.classifier(h_ctc)
        predicted_ct = ctc_logits.argmax(dim=1)   # (batch,)

        # routing and then scheduled mixing
        batch_size = x.size(0)
        routing_idx = torch.empty(batch_size, dtype=torch.long, device=x.device)
        if true_ct is not None and p_true > 0.0:
            # For each cell, decide randomly whether to use true label
            use_true = torch.rand(batch_size, device=x.device) < p_true
            routing_idx[use_true] = true_ct[use_true]
            routing_idx[~use_true] = predicted_ct[~use_true]
        else:
            routing_idx = predicted_ct

        # pred
        malig_logit = torch.zeros(batch_size, 1, device=x.device)
        for i in range(self.num_experts):
            mask = (routing_idx == i)
            if mask.sum() > 0:
                malig_logit[mask] = self.experts[i](h_exp[mask])

        return malig_logit, ctc_logits, routing_idx


# LOSS

# class weights for CTC loss, can help rare cancers
ct_class_counts = np.bincount(ct_train)   # ct_train from training split
ct_class_weights = 1.0 / (ct_class_counts + 1)
ct_class_weights = torch.tensor(ct_class_weights, dtype=torch.float32)

# losses
criterion_malig = nn.BCEWithLogitsLoss()   # sigmoid + bce
criterion_ctc = nn.CrossEntropyLoss(weight=ct_class_weights)