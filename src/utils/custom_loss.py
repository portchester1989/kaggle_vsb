import torch.nn as nn
import torch
def mixup_loss(output,y_a,y_b,lam):
    y_a_loss = y_a * torch.log(output + 1e-8) + (1 - y_a) * torch.log(1 - output + 1e-8)
    y_b_loss = y_b * torch.log(output + 1e-8) + (1 - y_b) * torch.log(1 - output + 1e-8)
    return - torch.mean(lam * y_a_loss + (1 - lam) * y_b_loss)
