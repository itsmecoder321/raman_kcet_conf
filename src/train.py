# src/train.py
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils import get_class_counts, print_classification_report, plot_confusion_matrix, save_checkpoint

# Config
DATA_DIR = r"Fetal Echocardiography First Trimester\data"    # path to folder containing train/valid/test
MODEL_DIR = r"Fetal Echocardiography First Trimester\model"
BATCH_SIZE = 16
NUM_EPOCHS = 2
LR = 1e-4
NUM_WORKERS = 4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1

def get_dataloaders(data_dir, img_size=224, batch_size=16):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=val_transforms)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    class_names = train_ds.classes
    return train_loader, val_loader, test_loader, class_names

def build_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def train():
    train_loader, val_loader, test_loader, class_names = get_dataloaders(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    model = build_model(len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        running_loss = 0.0
        preds = []
        trues = []
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())

        epoch_loss = running_loss / (len(train_loader.dataset))
        epoch_acc = accuracy_score(trues, preds)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_preds = []
        val_trues = []
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy().tolist())
                val_trues.extend(labels.cpu().numpy().tolist())

        val_loss = val_loss / (len(val_loader.dataset))
        val_acc = accuracy_score(val_trues, val_preds)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        scheduler.step(val_acc)

        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'class_names': class_names
            }, os.path.join(MODEL_DIR, 'best_model.pth'))
            print("Saved new best model")

    # Load best weights
    model.load_state_dict(best_model_wts)
    print(f"Training complete. Best val acc: {best_acc:.4f}")

    # Final evaluation on test set
    model.eval()
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy().tolist())
            test_trues.extend(labels.cpu().numpy().tolist())
    print("Test set results:")
    print_classification_report(test_trues, test_preds, class_names)
    plot_confusion_matrix(test_trues, test_preds, class_names, out_file=os.path.join(MODEL_DIR, 'confusion_matrix.png'))

if __name__ == "__main__":
    train()
