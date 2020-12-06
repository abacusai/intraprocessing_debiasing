"""
Adversarial Intraprocessing Algorithm.
"""
import logging

import numpy as np
import torch
import torch.optim as optim

from models import load_model, Critic
from utils import get_best_thresh, get_test_objective, get_valid_objective, compute_bias

logger = logging.getLogger("Debiasing")


def adversarial_debiasing(model_state_dict, data, config, device):
    logger.info('Training Adversarial model.')
    actor = load_model(data.num_features, config.get('hyperparameters', {}))
    actor.load_state_dict(model_state_dict)
    actor.to(device)
    hid = config['hyperparameters']['hid'] if 'hyperparameters' in config else 32
    critic = Critic(hid * config['adversarial']['batch_size'], num_deep=config['adversarial']['num_deep'], hid=hid)
    critic.to(device)
    critic_optimizer = optim.Adam(critic.parameters())
    critic_loss_fn = torch.nn.MSELoss()

    actor_optimizer = optim.Adam(actor.parameters(), lr=config['adversarial']['lr'])
    actor_loss_fn = torch.nn.BCELoss()

    for epoch in range(config['adversarial']['epochs']):
        for param in critic.parameters():
            param.requires_grad = True
        for param in actor.parameters():
            param.requires_grad = False
        actor.eval()
        critic.train()
        for step in range(config['adversarial']['critic_steps']):
            critic_optimizer.zero_grad()
            indices = torch.randint(0, data.X_valid.size(0), (config['adversarial']['batch_size'],))
            cX_valid = data.X_valid_gpu[indices]
            cy_valid = data.y_valid[indices]
            cp_valid = data.p_valid[indices]
            with torch.no_grad():
                scores = actor(cX_valid)[:, 0].reshape(-1).cpu().numpy()

            bias = compute_bias(scores, cy_valid.numpy(), cp_valid, config['metric'])

            res = critic(actor.trunc_forward(cX_valid))
            loss = critic_loss_fn(torch.tensor([bias], device=device), res[0])
            loss.backward()
            train_loss = loss.item()
            critic_optimizer.step()
            if (epoch % 10 == 0) and (step % 100 == 0):
                logger.info(f'=======> Critic Epoch: {(epoch, step)} loss: {train_loss}')

        for param in critic.parameters():
            param.requires_grad = False
        for param in actor.parameters():
            param.requires_grad = True
        actor.train()
        critic.eval()
        for step in range(config['adversarial']['actor_steps']):
            actor_optimizer.zero_grad()
            indices = torch.randint(0, data.X_valid.size(0), (config['adversarial']['batch_size'],))
            cy_valid = data.y_valid_gpu[indices]
            cX_valid = data.X_valid_gpu[indices]

            pred_bias = critic(actor.trunc_forward(cX_valid))
            bceloss = actor_loss_fn(actor(cX_valid)[:, 0], cy_valid)

            # loss = lam*abs(pred_bias) + (1-lam)*loss
            objloss = max(1, config['adversarial']['lambda']*(abs(pred_bias[0][0])-config['objective']['epsilon']+config['adversarial']['margin'])+1) * bceloss

            objloss.backward()
            train_loss = objloss.item()
            actor_optimizer.step()
            if (epoch % 10 == 0) and (step % 100 == 0):
                logger.info(f'=======> Actor Epoch: {(epoch, step)} loss: {train_loss}')

        if epoch % 10 == 0:
            with torch.no_grad():
                scores = actor(data.X_valid_gpu)[:, 0].reshape(-1, 1).cpu().numpy()
                _, best_adv_obj = get_best_thresh(scores, np.linspace(0, 1, 1001), data, config, valid=False, margin=config['adversarial']['margin'])
                logger.info(f'Objective: {best_adv_obj}')

    logger.info('Finding optimal threshold for Adversarial model.')
    with torch.no_grad():
        scores = actor(data.X_valid_gpu)[:, 0].reshape(-1, 1).cpu().numpy()

    best_adv_thresh, _ = get_best_thresh(scores, np.linspace(0, 1, 1001), data, config, valid=False, margin=config['adversarial']['margin'])

    logger.info('Evaluating Adversarial model on best threshold.')
    with torch.no_grad():
        labels = (actor(data.X_valid_gpu)[:, 0] > best_adv_thresh).reshape(-1, 1).cpu().numpy()
    results_valid = get_valid_objective(labels, data, config)
    logger.info(f'Results: {results_valid}')

    with torch.no_grad():
        labels = (actor(data.X_test_gpu)[:, 0] > best_adv_thresh).reshape(-1, 1).cpu().numpy()
    results_test = get_test_objective(labels, data, config)

    return results_valid, results_test
