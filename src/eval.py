import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_dir = '/Home/siv33/vbo084/DotaPicker/data/processed/'
model_dir = '/Home/siv33/vbo084/DotaPicker/saved_models/'

fnames = ['dota_train.pt', 'dota_val.pt', 'dota_test.pt']

# Idea: compute hits@1, 3, 5, 10, MRR

# We verify two different metrics:
# a) does the medal correctly pick heroes,
# b) given a pool of picks/bans, does the model correctly predict who won?

k_values = [1, 3, 5, 10]

def eval(model, dloader, k_values, device):

    model.eval()
    with torch.no_grad:
        for batch in dloader:
            inputs = batch
            inputs = inputs.to(device)
            targets = inputs[:, 1:]
            scores = model(inputs)[:, :-1, :] # Gets rid of the last prediction
            pickban_scores = scores[:, :-1, :].reshape(pickban_scores.size(0) * pickban_scores.size(1), pickban_scores.size(2)) # Gets rid of the prediction for RADWIN or DIREWIN
            teamwin_scores = scores[:, -1, :].reshape(pickban_scores.size(0), pickban_scores.size(2)) # Only considers the win prediction
            
            for k in k_values:
                logits, idcs = torch.topk(pickban_scores, k, dim=0)


            