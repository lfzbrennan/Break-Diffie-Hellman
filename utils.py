import torch
import numpy as np
import os


def accuracy(outputs, labels):
    return np.sum(np.all(outputs == labels, axis=1)) / outputs.shape[0]


def save_albert(output_dir, model):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model.state_dict(), f"{output_dir}/model.pt")


def save_mlm_albert(output_dir, model):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(model.state_dict(), f"{output_dir}/model.pt")


def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group["lr"]
