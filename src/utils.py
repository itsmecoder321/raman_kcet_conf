import os
import json
import shutil
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def get_class_counts(root):
    counts = {}
    for split in os.listdir(root):
        split_path = os.path.join(root, split)
        if not os.path.isdir(split_path):
            continue
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path): continue
            counts.setdefault(split, {})[cls] = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    return counts

def save_checkpoint(state, filename="model/checkpoint.pth"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(model, path, device):
    import torch
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def plot_confusion_matrix(y_true, y_pred, classes, out_file=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    if out_file:
        plt.savefig(out_file, bbox_inches='tight')
    else:
        plt.show()

def print_classification_report(y_true, y_pred, classes):
    print(classification_report(y_true, y_pred, target_names=classes))
