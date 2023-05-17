from re import M
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import math
import numpy as np
import os

whereIam = os.uname()[1]
print(whereIam)
HOME = whereIam



class OT_Loss_quantile(nn.Module):
    def __init__(self, all_models, target_classe, onedim = True):
        super(OT_Loss_quantile, self).__init__()
        self.target = target_classe
        self.targets = {}
        self.target_quantile = {}
        self.weights = {}
        self.onedim = onedim

        for model in all_models:
            self.target_quantile[model] = torch.load(f"data/{model}.pt", map_location="cuda")[self.target]

        self.n_quantile = self.target_quantile[all_models[0]].size(0)
        self.q = torch.linspace(0, 1, self.n_quantile, device="cuda")



    def forward(self, features, model):
        

        if model in ["deit_s", "deit_s_adv", "deit_t", "deit_b"]:
            features = features
        else: 
            features = features.mean((2,3))


        features_quantiles = torch.quantile(features, self.q, dim=0)


        l2 = ((features_quantiles - self.target_quantile[model])**2).mean(0).sum()

        l1 = ((features_quantiles - self.target_quantile[model]).abs()).mean(0).sum()

        loss = l2 #+ l1


        return loss, l2, l1
    


#---------------------------------------------------------------------
# Smoothness loss function
#---------------------------------------------------------------------
def Smoothness_loss(patch):
    device = patch.device.type
    p_h, p_w = patch.shape[-2:]

    if torch.max(patch) > 1:
        patch = patch / 255
    diff_w = torch.square(patch[:, :, :-1, :] - patch[:, :, 1:, :])
    zeros_w = torch.zeros((1, 3, 1, p_w), device=device)
    diff_h = torch.square(patch[:, :, :, :-1] - patch[:, :, :, 1:])
    zeros_h = torch.zeros((1, 3, p_h, 1), device=device)
    return torch.sum(torch.cat((diff_w, zeros_w), dim=2) + torch.cat((diff_h, zeros_h), dim=3))

