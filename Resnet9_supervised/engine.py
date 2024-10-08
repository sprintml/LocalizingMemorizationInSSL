import torch
import numpy as np

def train_fn(dataloader, model, device, optimizer, scheduler):
    model.train()
    epoch_loss = []
    for batch in dataloader:
        loss = model.train_step(batch, model, device)
        epoch_loss.append(loss.cpu().detach())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    return np.mean(epoch_loss)

def eval_fn(dataloader, model, device):
    model.eval()

    accuracy = []
    with torch.no_grad():
        for batch in dataloader:
            acc = model.validation_step(batch, model, device)
            accuracy.append(acc.cpu().detach())
            break

    return np.mean(accuracy)