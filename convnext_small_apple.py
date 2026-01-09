# ================================================================
#                     SYSTEM AND LIBRARIES
# ================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import copy
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

# ================================================================
#                     HYPERPARAMETERS AND SETTINGS
# ================================================================
# User Control
transfer_learning = True

# Data Directory
DATA_DIR = "/content/drive/MyDrive/apple_dataset"

NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ================================================================
#              DATASET PREPARATION
# ================================================================
class TransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_data_loaders():
    # 1. Data Loading
    full_dataset = datasets.ImageFolder(DATA_DIR)
    classes = full_dataset.classes

    # 2. Split Ratios
    train_len = int(len(full_dataset) * 0.70)
    val_len   = int(len(full_dataset) * 0.15)
    test_len  = len(full_dataset) - train_len - val_len

    train_subset, val_subset, test_subset = random_split(full_dataset, [train_len, val_len, test_len])

    # 3. Statistics Table and Weight Calculation
    # Calculate distribution after random split
    # Accessing indices in the main dataset via subset.indices
    all_targets = np.array(full_dataset.targets)

    stats = {cls: {"Train": 0, "Val": 0, "Test": 0} for cls in classes}

    for idx in train_subset.indices: stats[classes[all_targets[idx]]]["Train"] += 1
    for idx in val_subset.indices:   stats[classes[all_targets[idx]]]["Val"] += 1
    for idx in test_subset.indices:  stats[classes[all_targets[idx]]]["Test"] += 1

    df = pd.DataFrame(stats).T
    df["Total"] = df.sum(axis=1)

    # Grand Total Row
    total_row = df.sum(axis=0)
    total_row.name = "GRAND TOTAL"
    df = pd.concat([df, pd.DataFrame(total_row).T])

    print("\n--- Dataset Distribution Table ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df)
    print("-" * 50)
    # 4. Transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_data = TransformDataset(train_subset, transform=train_tf)
    val_data   = TransformDataset(val_subset,   transform=test_tf)
    test_data  = TransformDataset(test_subset,  transform=test_tf)

    loaders = {
        "train": DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True),
        "val":   DataLoader(val_data,   batch_size=BATCH_SIZE),
        "test":  DataLoader(test_data,  batch_size=BATCH_SIZE)
    }

    return loaders, classes, test_data

try:
    loaders, classes, test_dataset_for_vis = get_data_loaders()
except Exception as e:
    print(f"ERROR: Dataset could not be loaded. Please check DATA_DIR path.\n{e}")
    loaders, classes, test_dataset_for_vis = None, [], None

# ================================================================
#                 TRAINING FUNCTION
# ================================================================
if loaders:
    print(f"\n{'='*40}")
    print(f"STARTING: TRANSFER LEARNING = {transfer_learning}")
    print(f"{'='*40}")

    # 1. Model Setup
    # weights='DEFAULT' automatically downloads best ImageNet weights
    model = convnext_small(weights='DEFAULT')

    # Freezing Process
    if transfer_learning:
        # In ConvNeXt, the main feature extractor is 'features' block
        for param in model.features.parameters():
            param.requires_grad = False
        print("Feature Extractor frozen.")
    else:
        print("All layers will be trained (Full Fine-Tuning).")

    # Change the last layer
    # ConvNeXt classifier is a Sequential block:
    # (0) LayerNorm2d, (1) Flatten, (2) Linear
    # We modify the final Linear layer
    num_f = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_f, len(classes))
    model = model.to(DEVICE)

    # --- MODEL ARCHITECTURE AND PARAMETER COUNTS ---
    print("\n--- DETAILED MODEL ARCHITECTURE ---")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}\n")
    print("-" * 40)

    # Optimizer
    if transfer_learning:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    criterion = nn.CrossEntropyLoss()

    # 2. Training Loop
    history = {"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}

    start_t = time.time()
    for e in range(NUM_EPOCHS):
        model.train()
        t_loss, correct, total = 0, 0, 0

        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()*x.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = t_loss/total
        train_acc = correct/total*100

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for x, y in loaders["val"]:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                v_loss += criterion(out, y).item()*x.size(0)
                _, pred = torch.max(out, 1)
                v_correct += (pred==y).sum().item()
                v_total += y.size(0)

        val_loss = v_loss/v_total
        val_acc = v_correct/v_total*100

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {e+1}/{NUM_EPOCHS} | T.Loss: {train_loss:.4f} T.Acc: {train_acc:.2f}% | "
              f"V.Loss: {val_loss:.4f} V.Acc: {val_acc:.2f}%")

    duration = (time.time() - start_t) / 60
    print(f"Training Completed. Duration: {duration:.2f} min")

    # ================================================================
    #                 PLOTS AND RESULTS
    # ================================================================

    # 1. Loss / Accuracy Plots
    epochs_range = range(1, NUM_EPOCHS+1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], marker='o', label='Train Loss')
    plt.plot(epochs_range, history["val_loss"], marker='o', label='Val Loss')
    plt.title(f'Loss ({("TL=True" if transfer_learning else "TL=False")})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], marker='o', label='Train Acc')
    plt.plot(epochs_range, history["val_acc"], marker='o', label='Val Acc')
    plt.title(f'Accuracy ({("TL=True" if transfer_learning else "TL=False")})')
    plt.legend()
    plt.show()

    # 2. Confusion Matrix and Report
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loaders["test"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Draw Confusion Matrix First
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Print Test Report
    print("\n--- Test Report ---")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 3. Sample Predictions (2 from each class)
    print("\n--- Sample Predictions (2 per Class) ---")

    def unnormalize(tensor):
        return (tensor * 0.5 + 0.5).clamp(0, 1)

    examples_found = {cls: [] for cls in classes}

    # Scan test set and find 2 examples per class
    for i in range(len(test_dataset_for_vis)):
        img, lbl = test_dataset_for_vis[i]
        label_name = classes[lbl]

        if len(examples_found[label_name]) < 2:
            examples_found[label_name].append((img, lbl))

        if all(len(v) == 2 for v in examples_found.values()):
            break

    # Visualization
    # Dynamic grid: as many rows as classes, 2 columns per class
    num_rows = len(classes)
    plt.figure(figsize=(12, 4 * num_rows)) # ~4 units height per row
    plot_idx = 1

    for cls_name in classes:
        for img, lbl in examples_found[cls_name]:
            model.eval()
            img_tensor = img.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(img_tensor)
                _, pred = torch.max(out, 1)

            pred_name = classes[pred.item()]

            # Image
            img_np = unnormalize(img).permute(1, 2, 0).cpu().numpy()

            plt.subplot(num_rows, 2, plot_idx)
            plt.imshow(img_np)
            color = 'green' if pred_name == cls_name else 'red'
            plt.title(f"Actual: {cls_name}\nPred: {pred_name}", color=color)
            plt.axis('off')
            plot_idx += 1

    plt.tight_layout()
    plt.show()