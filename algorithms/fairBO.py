"""
FairBO Intraprocessing Algorithm.
"""
import math
import logging
import copy

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from botorch.acquisition import ConstrainedExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import StandardScaler

from models import load_model
from utils import get_best_thresh, get_test_objective, get_valid_objective

logger = logging.getLogger("Debiasing")


def fairBO_debiasing(model_state_dict, data, config, device):
    def evaluate(lr, beta1, beta2, alpha, T0, verbose=False):
        model = load_model(data.num_features, config.get('hyperparameters', {}))
        model.load_state_dict(model_state_dict)
        model.to(device)

        loss_fn = torch.nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=alpha
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(T0))

        for epoch in range(201):
            model.train()
            batch_idxs = torch.split(torch.randperm(data.X_valid.size(0)), 64)
            train_loss = 0
            for batch in batch_idxs:
                X = data.X_valid_gpu[batch, :]
                y = data.y_valid_gpu[batch]

                optimizer.zero_grad()
                loss = loss_fn(model(X)[:, 0], y)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                scheduler.step(X.size(0))
            if epoch % 10 == 0 and verbose:
                model.eval()
                with torch.no_grad():
                    valid_loss = loss_fn(model(data.X_valid_valid.to(device))[:, 0], data.y_valid_valid.to(device))
                print(f'=======> Epoch: {epoch} Train loss: {train_loss / len(batch_idxs)} '
                      f'Valid loss: {valid_loss}')

        model.eval()
        with torch.no_grad():
            scores = model(data.X_valid_gpu)[:, 0].reshape(-1).cpu().numpy()

        best_thresh, _ = get_best_thresh(scores, np.linspace(0, 1, 1001), data, config, valid=False, margin=config['fairBO']['margin'])
        return get_valid_objective(scores > best_thresh, data, config, valid=False), model, best_thresh

    space = config['fairBO']['hyperparameters']
    search_space = {}
    bounds_dict = {}
    for var in space:
        search_space[var] = np.arange(space[var]['start'], space[var]['end'], space[var]['step'])
        bounds_dict[var] = torch.tensor([space[var]['start'], space[var]['end']])
        if space[var]['log_scale']:
            search_space[var] = np.exp(np.log(10) * search_space[var])
            bounds_dict[var] = torch.exp(float(np.log(10)) * bounds_dict[var])

    def sample_space(): return {var: np.random.choice(rng) for var, rng in search_space.items()}

    X_hyp = []
    y_hyp = []
    best_model = [None, -math.inf, -1]
    for it in range(config['fairBO']['initial_budget']):
        X_hyp.append(sample_space())
        logger.info(f'(Iteration {it}) Evaluating fairBO with sample {X_hyp[-1]}')
        y_eval, model_candidate, thresh = evaluate(**X_hyp[-1])
        logger.info(f'Result: {y_eval}')
        if y_eval['objective'] > best_model[1]:
            best_model[0] = copy.deepcopy(model_candidate)
            best_model[1] = y_eval['objective']
            best_model[2] = thresh
        y_hyp.append(y_eval)

    X_df = pd.DataFrame(X_hyp)
    X = torch.tensor(X_df.to_numpy())
    y = torch.tensor(pd.DataFrame(y_hyp)[['performance', 'bias']].to_numpy())

    for it in range(config['fairBO']['total_budget'] - config['fairBO']['initial_budget']):
        xscaler = StandardScaler()
        gp = SingleTaskGP(torch.tensor(xscaler.fit_transform(X)), y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        cEI = ConstrainedExpectedImprovement(gp, y[:, 0].max().item(), 0, {1: (-0.05, 0.05)})
        bounds = torch.stack([bounds_dict[x] for x in X_df.columns])
        candidate, _ = optimize_acqf(cEI, bounds.T, 1, 100, 1024)
        inv_candidate = xscaler.inverse_transform(candidate)

        hyp = {k: v.item() for k, v in zip(X_df.columns, inv_candidate[0])}
        logger.info(f'(Iteration {it+config["fairBO"]["initial_budget"]}) Evaluating fairBO with sample {hyp}')

        X = torch.cat((X, candidate))

        y_eval, model_candidate, thresh = evaluate(**hyp)
        logger.info(f'Result: {y_eval}')
        if y_eval['objective'] > best_model[1]:
            best_model[0] = copy.deepcopy(model_candidate)
            best_model[1] = y_eval['objective']
            best_model[2] = thresh
        y = torch.cat((y, torch.tensor([[y_eval['performance'], y_eval['bias']]])))

    logger.info('Evaluating best fairBO debiased model.')
    best_model[0].eval()
    with torch.no_grad():
        y_pred = (best_model[0](data.X_valid_gpu)[:, 0] > best_model[2]).reshape(-1).cpu().numpy()
    results_valid = get_valid_objective(y_pred, data, config)
    logger.info(f'Results: {results_valid}')

    best_model[0].eval()
    with torch.no_grad():
        y_pred = (best_model[0](data.X_test_gpu)[:, 0] > best_model[2]).reshape(-1).cpu().numpy()
    results_test = get_test_objective(y_pred, data, config)
    return results_valid, results_test
