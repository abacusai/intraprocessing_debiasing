import logging
import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):

    def __init__(self, input_size, num_deep=10, hid=32, dropout_p=0.2):
        super().__init__()
        self.fc0 = nn.Linear(input_size, hid)
        self.bn0 = nn.BatchNorm1d(hid)
        self.fcs = nn.ModuleList([nn.Linear(hid, hid) for _ in range(num_deep)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hid) for _ in range(num_deep)])
        self.out = nn.Linear(hid, 2)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, t):
        t = self.bn0(self.dropout(F.relu(self.fc0(t))))
        for bn, fc in zip(self.bns, self.fcs):
            t = bn(self.dropout(F.relu(fc(t))))
        return torch.sigmoid(self.out(t))

    def trunc_forward(self, t):
        t = self.bn0(self.dropout(F.relu(self.fc0(t))))
        for bn, fc in zip(self.bns, self.fcs):
            t = bn(self.dropout(F.relu(fc(t))))
        return t


def load_model(input_size, hyperparameters):
    return Model(input_size, **hyperparameters)


def train_model(model, data, epochs=1001):
    loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    Patience = namedtuple('Patience', ('loss', 'model', 'step'))
    patience = Patience(math.inf, None, 0)
    patience_limit = 10
    for epoch in range(epochs):
        model.train()
        batch_idxs = torch.split(torch.randperm(data.X_train.size(0)), 64)
        train_loss = 0
        for batch in batch_idxs:
            X = data.X_train[batch, :]
            y = data.y_train[batch]

            optimizer.zero_grad()
            loss = loss_fn(model(X)[:, 0], y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = loss_fn(model(data.X_valid)[:, 0], data.y_valid)
        scheduler.step(valid_loss)
        if epoch % 10 == 0:
            if valid_loss > patience.loss:
                patience = Patience(patience.loss, patience.model, patience.step+1)
            else:
                patience = Patience(valid_loss, model.state_dict(), 0)
            if patience.step > patience_limit:
                logging.info('Ending early, patience limit has been crossed without an increase in validation loss!')
                model.load_state_dict(patience.model)
                break
            logging.info(f'=======> Epoch: {epoch} Train loss: {train_loss / len(batch_idxs)} '
                         f'Valid loss: {valid_loss} Patience valid loss: {patience.loss}')


class Critic(nn.Module):

    def __init__(self, sizein, num_deep=3, hid=32):
        super().__init__()
        self.fc0 = nn.Linear(sizein, hid)
        self.fcs = nn.ModuleList([nn.Linear(hid, hid) for _ in range(num_deep)])
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hid, 1)

    def forward(self, t):
        t = t.reshape(1, -1)
        t = self.fc0(t)
        for fc in self.fcs:
            t = F.relu(fc(t))
            t = self.dropout(t)
        return self.out(t)
