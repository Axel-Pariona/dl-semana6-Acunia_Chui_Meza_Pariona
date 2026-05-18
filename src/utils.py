import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def set_seed(seed=42):
    """
    Fija semillas para reproducibilidad.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Cuenta parámetros entrenables.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_data_loaders(data_dir, batch_size=32, img_size=64, val_split=0.2, seed=42):
    """
    Carga TRAIN y TEST usando ImageFolder.
    Divide TRAIN en train/validation.
    """

    train_dir = os.path.join(data_dir, "TRAIN")
    test_dir = os.path.join(data_dir, "TEST")

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_train_aug = datasets.ImageFolder(train_dir, transform=train_transform)
    full_train_eval = datasets.ImageFolder(train_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    val_size = int(len(full_train_aug) * val_split)
    train_size = len(full_train_aug) - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_train_aug), generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train_aug, train_indices)
    val_dataset = Subset(full_train_eval, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, num_workers=2,
        pin_memory=True, generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, num_workers=2,
        pin_memory=True
    )

    class_names = full_train_aug.classes

    return train_loader, val_loader, test_loader, class_names


def plot_training_curves(history, model_name, save_dir="results/figures"):
    """
    Guarda curvas de pérdida y accuracy.
    """

    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title(f"Curva de pérdida - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_loss.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title(f"Curva de accuracy - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_accuracy.png"), bbox_inches="tight")
    plt.close()


def get_transfer_data_loaders(data_dir, batch_size=32, img_size=224, val_split=0.2, seed=42):
    """
    Carga TRAIN y TEST para Transfer Learning.
    Usa imágenes 224x224 y separa train/validation.
    """

    train_dir = os.path.join(data_dir, "TRAIN")
    test_dir = os.path.join(data_dir, "TEST")

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_train_aug = datasets.ImageFolder(train_dir, transform=train_transform)
    full_train_eval = datasets.ImageFolder(train_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    val_size = int(len(full_train_aug) * val_split)
    train_size = len(full_train_aug) - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_train_aug), generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train_aug, train_indices)
    val_dataset = Subset(full_train_eval, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    class_names = full_train_aug.classes

    return train_loader, val_loader, test_loader, class_names