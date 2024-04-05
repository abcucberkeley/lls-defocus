import torch
import torch.nn as nn

class Custom_MAE(nn.Module):
    def __init__(self, threshold):
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        # if difference is greater than the threshold, penalize more
        abs_diff = torch.abs(y_pred - y_true)
        small_dev_mask = abs_diff >= self.threshold

        # penalize for all abs_diff that are greater than the threshold 
        # to ensure penalization of small (yet still significant) deviations from true value
        if abs_diff < 1:
            loss = torch.where(small_dev_mask, 1 / abs_diff, abs_diff)
        else:
            loss = abs_diff ** 2
        return torch.mean(loss)