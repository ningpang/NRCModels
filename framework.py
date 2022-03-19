import torch
import random
import copy
import torch.nn as nn
import numpy as np
import torch.optim as optim
from transformers import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

def train_bert_model(config, model, train_data, num_epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': 0.00001},
                            {'params': model.classifier.parameters(), 'lr': 0.001}])
    for epoch in range(num_epochs):
        losses = []
        for step, (labels, lengthes, tokens, masks) in enumerate(train_data):
            model.zero_grad()
            labels = labels.to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            masks = torch.stack([x.to(config.device) for x in masks], dim=0)
            logits = model(tokens, masks)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f'Finetuning loss is {np.array(losses).mean()}')

def evaluate_bert_model(config, model, test_data):
    model.eval()
    n = len(test_data)
    correct = 0
    for step, (labels, lengthes, tokens, mask) in enumerate(test_data):
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        mask = torch.stack([x.to(config.device) for x in mask], dim=0)
        logits = model(tokens, mask)
        seen_sim = logits.cpu().data.numpy()
        max_smi = np.max(seen_sim, axis=1)
        label_sim = logits[:, labels].cpu().data.numpy()
        if label_sim >= max_smi:
            correct += 1
    return correct/n

def train_bertcnn_model(config, model, train_data, num_epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm', 'Layernorm.weight']
    param_optimizer = list(model.named_parameters())
    # print(param_optimizer)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) ], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) ], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = 5e-5,
                         warmup=0.5,
                         t_total=len(train_data)*10)
    for epoch in range(num_epochs):
        losses = []
        for step, (labels, lengthes, tokens, masks) in enumerate(train_data):
            model.zero_grad()
            labels = labels.to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            masks = torch.stack([x.to(config.device) for x in masks], dim=0)
            logits = model(tokens, masks)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f'Finetuning loss is {np.array(losses).mean()}')


def train_cnn_model(config, model, train_data, num_epochs):

    model.train()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.001}])
    optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000)
    for epoch in range(num_epochs):
        losses = []
        for step, (labels, length, tokens, pos1, pos2, mask) in enumerate(train_data):
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            pos1 = torch.stack([x.to(config.device) for x in pos1], dim=0)
            pos2 = torch.stack([x.to(config.device) for x in pos2], dim=0)
            mask = torch.stack([x.to(config.device) for x in mask], dim=0)
            labels = labels.to(config.device)

            model.zero_grad()
            logits = model(tokens, pos1, pos2, mask)

            loss = criterion(logits, labels)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), config.max_grad_norm)
        print(f'Finetuning loss is {np.array(losses).mean()}')

def evaluate_cnn_model(config, model, test_data):
    model.eval()
    n = len(test_data)
    correct = 0
    for step, (labels, length, tokens, pos1, pos2, mask) in enumerate(test_data):
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        pos1 = torch.stack([x.to(config.device) for x in pos1], dim=0)
        pos2 = torch.stack([x.to(config.device) for x in pos2], dim=0)
        mask = torch.stack([x.to(config.device) for x in mask], dim=0)
        labels = labels.to(config.device)
        logits = model(tokens, pos1, pos2, mask)
        seen_sim = logits.cpu().data.numpy()
        max_smi = np.max(seen_sim, axis=1)
        label_sim = logits[:, labels].cpu().data.numpy()
        if label_sim >= max_smi:
            correct += 1
    return correct/n

def train_lstm_model(config, model, train_data, num_epochs):

    model.train()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.001}])
    optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000)

    for epoch in range(num_epochs):
        losses = []
        for step, (labels, length, tokens, pos1, pos2, mask) in enumerate(train_data):
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            pos1 = torch.stack([x.to(config.device) for x in pos1], dim=0)
            pos2 = torch.stack([x.to(config.device) for x in pos2], dim=0)
            mask = torch.stack([x.to(config.device) for x in mask], dim=0)
            labels = labels.to(config.device)

            model.zero_grad()
            logits = model(tokens, pos1, pos2, mask)

            loss = criterion(logits, labels)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), config.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
        print(f'Finetuning loss is {np.array(losses).mean()}')

def evaluate_lstm_model(config, model, test_data):
    model.eval()
    n = len(test_data)
    correct = 0
    for step, (labels, length, tokens, pos1, pos2, mask) in enumerate(test_data):
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        pos1 = torch.stack([x.to(config.device) for x in pos1], dim=0)
        pos2 = torch.stack([x.to(config.device) for x in pos2], dim=0)
        mask = torch.stack([x.to(config.device) for x in mask], dim=0)
        labels = labels.to(config.device)
        logits = model(tokens, pos1, pos2, mask)
        seen_sim = logits.cpu().data.numpy()
        max_smi = np.max(seen_sim, axis=1)
        label_sim = logits[:, labels].cpu().data.numpy()
        if label_sim >= max_smi:
            correct += 1
    return correct/n