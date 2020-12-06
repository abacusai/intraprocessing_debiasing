"""
Layerwise Optimizer Intraprocessing Algorithm.
"""
import copy
import logging
import math

import numpy as np
import torch
from models import load_model
from skopt import gbrt_minimize
from skopt.space import Real
from utils import get_best_thresh, get_test_objective, get_valid_objective

logger = logging.getLogger("Debiasing")


def layerwiseOpt_debiasing(model_state_dict, data, config, device):
    logger.info('Training layerwiseOpt model.')
    base_model = load_model(data.num_features, config.get('hyperparameters', {}))
    base_model.load_state_dict(model_state_dict)
    base_model.to(device)
    best_state_dict, best_obj, best_thresh = None, math.inf, -1

    total_params = len(list(base_model.parameters()))
    for index, param in enumerate(base_model.parameters()):
        if index < total_params - config['layerwiseOpt']['num_layers']:
            continue
        logger.info(f'Evaluating param number {index} of {total_params}')
        param_copy = copy.deepcopy(param)

        def objective(new_param, return_thresh=False):
            param.data[indices] = torch.tensor(new_param)
            base_model.eval()
            with torch.no_grad():
                scores = base_model(data.X_valid_gpu)[:, 0].reshape(-1).numpy()
            best_thresh, best_obj = get_best_thresh(scores, np.linspace(0, 1, 501), data, config, valid=False, margin=config['layerwiseOpt']['margin'])
            print(f'Evaluating param number {index} of {total_params}')
            if return_thresh:
                return -float(best_obj), float(best_thresh)
            return -float(best_obj)

        mean = param.flatten().cpu().detach().numpy().mean()
        std = param.flatten().cpu().detach().numpy().std()
        num_elems = param.size().numel()
        ratio = min(1., config['layerwiseOpt']['max_sparsity'] / num_elems)
        indices = torch.rand(param.size()) < ratio
        space = [Real(float(x.cpu().detach()) - 2.2*std, float(x.cpu().detach()) + 2.2*std) for x in param[indices]]

        # std = param.flatten().cpu().detach().numpy().std()
        # num_elems = param.size().numel()
        # ratio = min(1., config['layerwiseOpt']['max_sparsity'] / num_elems)
        # indices = torch.rand(param.size()) < ratio
        logger.info(f'Number of sparse indices: {indices.sum().item()}')
        res_gbrt = gbrt_minimize(
            objective,
            space,
            n_calls=config['layerwiseOpt']['n_calls'],
            verbose=True
        )

        if res_gbrt.fun < best_obj:
            param.data[indices] = torch.tensor(res_gbrt.x)
            best_state_dict = base_model.state_dict()
            best_obj, best_thresh = objective(res_gbrt.x, return_thresh=True)
            best_obj = -best_obj
        param.data = param_copy.data

    best_model = load_model(data.num_features, config.get('hyperparameters', {}))
    best_model.to(device)
    best_model.load_state_dict(best_state_dict)
    best_model.eval()
    with torch.no_grad():
        y_pred = (best_model(data.X_valid_gpu)[:, 0] > best_thresh).reshape(-1).numpy()
    results_valid = get_valid_objective(y_pred, data, config)

    best_model.eval()
    with torch.no_grad():
        y_pred = (best_model(data.X_test_gpu)[:, 0] > best_thresh).reshape(-1).numpy()
    results_test = get_test_objective(y_pred, data, config)

    return results_valid, results_test
