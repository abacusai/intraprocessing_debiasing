"""
Mitigating Intraprocessing Algorithm.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import load_model
from utils import get_best_thresh, get_test_objective, get_valid_objective

logger = logging.getLogger("Debiasing")


def mitigating_debiasing(model_state_dict, data, config, device):
    logger.info('Training Mitigating model.')
    actor = load_model(data.num_features, config.get('hyperparameters', {}))
    actor.load_state_dict(model_state_dict)
    actor.to(device)
    critic = nn.Sequential(
        nn.Linear(32, 32),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.Softmax()
    )
    critic.to(device)
    critic_optimizer = optim.Adam(critic.parameters())
    critic_loss_fn = torch.nn.BCELoss()

    actor_optimizer = optim.Adam(actor.parameters(), lr=config['mitigating']['lr'])
    actor_loss_fn = torch.nn.BCELoss()

    for epoch in range(config['mitigating']['epochs']):
        for param in critic.parameters():
            param.requires_grad = True
        for param in actor.parameters():
            param.requires_grad = False
        actor.eval()
        critic.train()
        for step in range(config['mitigating']['critic_steps']):
            critic_optimizer.zero_grad()
            indices = torch.randint(0, data.X_valid.size(0), (config['mitigating']['batch_size'],))
            cy_valid = data.y_valid_gpu[indices]
            cX_valid = data.X_valid_gpu[indices]
            cp_valid = data.p_valid_gpu[indices]
            with torch.no_grad():
                scores = actor(cX_valid)[:, 0].reshape(-1).cpu().numpy()

            res = critic(actor.trunc_forward(cX_valid))
            loss = critic_loss_fn(res[:, 0], cp_valid.type(torch.float32))
            loss.backward()
            train_loss = loss.item()
            critic_optimizer.step()
            if (epoch % 5 == 0) and (step % 100 == 0):
                logger.info(f'=======> Critic Epoch: {(epoch, step)} loss: {train_loss}')

        for param in critic.parameters():
            param.requires_grad = False
        for param in actor.parameters():
            param.requires_grad = True
        actor.train()
        critic.eval()
        for step in range(config['mitigating']['actor_steps']):
            actor_optimizer.zero_grad()
            indices = torch.randint(0, data.X_valid.size(0), (config['mitigating']['batch_size'],))
            cy_valid = data.y_valid_gpu[indices]
            cX_valid = data.X_valid_gpu[indices]
            cp_valid = data.p_valid_gpu[indices]

            cx_predict = actor(cX_valid)
            loss_pred = actor_loss_fn(cx_predict[:, 0], cy_valid)

            cp_predict = critic(actor.trunc_forward(cX_valid))
            loss_adv = critic_loss_fn(cp_predict[:, 0], cp_valid.type(torch.float32))

            for param in actor.parameters():
                try:
                    lp = torch.autograd.grad(loss_pred, param, retain_graph=True)[0]
                    la = torch.autograd.grad(loss_adv, param, retain_graph=True)[0]
                except RuntimeError:
                    continue
                shape = la.shape
                lp = lp.flatten()
                la = la.flatten()
                lp_proj = (lp.T @ la) * la
                grad = lp - lp_proj - config['mitigating']['alpha']*la
                grad = grad.reshape(shape)
                param.backward(grad)

            actor_optimizer.step()
            if (epoch % 5 == 0) and (step % 100 == 0):
                logger.info(f'=======> Actor Epoch: {(epoch, step)}')

        if epoch % 5 == 0:
            with torch.no_grad():
                scores = actor(data.X_valid_gpu)[:, 0].reshape(-1, 1).cpu().numpy()
                _, best_mit_obj = get_best_thresh(scores, np.linspace(0, 1, 1001), data, config, valid=False, margin=config['mitigating']['margin'])
                logger.info(f'Objective: {best_mit_obj}')

    logger.info('Finding optimal threshold for Mitigating model.')
    with torch.no_grad():
        scores = actor(data.X_valid_gpu)[:, 0].reshape(-1, 1).cpu().numpy()

    best_mit_thresh, _ = get_best_thresh(scores, np.linspace(0, 1, 1001), data, config, valid=False, margin=config['mitigating']['margin'])

    logger.info('Evaluating Mitigating model on best threshold.')
    with torch.no_grad():
        labels = (actor(data.X_valid_gpu)[:, 0] > best_mit_thresh).reshape(-1, 1).cpu().numpy()
    results_valid = get_valid_objective(labels, data, config)
    logger.info(f'Results: {results_valid}')

    with torch.no_grad():
        labels = (actor(data.X_test_gpu)[:, 0] > best_mit_thresh).reshape(-1, 1).cpu().numpy()
    results_test = get_test_objective(labels, data, config)

    return results_valid, results_test
