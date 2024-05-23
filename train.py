import os
import random

import numpy as np
import matplotlib.pyplot as plt

from dataset import CelebretiesDataset
from model import SiameseNetwork

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torchvision.transforms as v2

from torchmetrics.accuracy import BinaryAccuracy

def main():
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ds_path = './datasets/celebreties'
    names = os.listdir(ds_path)
    random.shuffle(names)

    num_epochs = 30
    train_split = 0.8
    batch_size = 5
    lr = 1e-4

    train_transform = v2.Compose([
        v2.Resize((112, 112)),
        #v2.Lambda(lambda image: v2.functional.erase(image, 60, 0, 52, 112, 0)),
        v2.RandomHorizontalFlip(),
        #v2.RandomVerticalFlip(),
        #v2.RandomRotation([0, 359]),
        v2.ColorJitter(),
        v2.ToDtype(torch.float, scale=True)
    ])

    val_transform = v2.Compose([
        v2.Resize((112, 112)),
        v2.ToDtype(torch.float, scale=True)
    ])

    train_ds = CelebretiesDataset(ds_path, names[:int(train_split * len(names))], transform=train_transform)
    val_ds = CelebretiesDataset(ds_path, names[int(train_split * len(names)):], transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, shuffle=False)

    model = SiameseNetwork().to(device)
    #criterion = ContrastiveLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = BinaryAccuracy()
        for batch_idx, (images_1, images_2, labels) in enumerate(train_loader):
            images_1, images_2, labels = images_1.to(device), images_2.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images_1, images_2)
            loss = criterion(outputs, labels)
            train_loss += loss.sum().item()
            train_acc.update(outputs, labels)

            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * batch_size}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {train_loss / (batch_idx * batch_size + 1e-6):.6f}\tAcc: {train_correct / (batch_idx * batch_size + 1e-6):.6f}')

        model.eval()
        val_loss = 0
        val_acc = BinaryAccuracy()
        

        with torch.no_grad():
            for (images_1, images_2, labels) in val_loader:
                images_1, images_2, labels = images_1.to(device), images_2.to(device), labels.to(device)

                outputs = model(images_1, images_2)

                loss = criterion(outputs, labels)
                val_loss += loss.sum().item()
                val_acc.update(outputs, labels)

        val_loss /= len(val_loader.dataset)
        print(f'Validation: val_loss:{val_loss:.4f}, val_acc:{val_acc.compute()}\n')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                            'epoch': epoch
                        }, 'best.pth')
            
        torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        'epoch': epoch
                    }, 'last.pth')

if __name__ == '__main__':
    main()