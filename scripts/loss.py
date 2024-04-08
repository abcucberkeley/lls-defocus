import torch
import torch.nn as nn

class Custom_MAE(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        # if difference is greater than the threshold, penalize more
        abs_diff = torch.abs(y_pred - y_true)
        small_dev_mask = (abs_diff >= self.threshold) & (abs_diff < 1) & (abs_diff > 0)

        # penalize for all abs_diff that are greater than the threshold 
        # to ensure penalization of small (yet still significant) deviations from true value
        loss = torch.where(small_dev_mask, 1 / torch.add(abs_diff, 1e-6), torch.pow(abs_diff,2)) # avoid dividing by zero
        print(loss)
        return torch.mean(loss)