"""
Random Intraprocessing Algorithm.
"""
import copy
import logging
import math

import numpy as np
import torch

from models import load_model
from utils import get_best_thresh, get_test_objective, get_valid_objective

logger = logging.getLogger("Debiasing")


def random_debiasing(model_state_dict, data, config, device, verbose=True):
    logger.info('Generating Random Debiased models.')
    rand_model = load_model(data.num_features, config.get('hyperparameters', {}))
    rand_model.to(device)
    rand_result = {'objective': -math.inf, 'model': rand_model.state_dict(), 'thresh': -1}
    for iteration in range(config['random']['num_trials']):
        rand_model.load_state_dict(model_state_dict)
        for param in rand_model.parameters():
            param.data = param.data * (torch.randn_like(param) * config['random']['stddev'] + 1)

        rand_model.eval()
        with torch.no_grad():
            scores = rand_model(data.X_valid_gpu)[:, 0].reshape(-1).cpu().numpy()

        threshs = np.linspace(0, 1, 501)
        best_rand_thresh, best_obj = get_best_thresh(scores, threshs, data, config, valid=False, margin=config['random']['margin'])
        if best_obj > rand_result['objective']:
            rand_result = {'objective': best_obj, 'model': copy.deepcopy(rand_model.state_dict()), 'thresh': best_rand_thresh}
            rand_model.eval()
            with torch.no_grad():
                y_pred = (rand_model(data.X_test_gpu)[:, 0] > best_rand_thresh).reshape(-1).cpu().numpy()
            best_test_result = get_test_objective(y_pred, data, config)['objective']

        if iteration % 10 == 0 and verbose:
            logger.info(f'{iteration} / {config["random"]["num_trials"]} trials have been sampled.')
            logger.info(f'Best result so far = {rand_result["objective"]}')
            logger.info(f'Best test result so = {best_test_result}')

    logger.info('Evaluating best random debiased model.')
    rand_model.load_state_dict(rand_result['model'])
    rand_model.eval()
    with torch.no_grad():
        y_pred = (rand_model(data.X_valid_gpu)[:, 0] > rand_result['thresh']).reshape(-1).cpu().numpy()
    results_valid = get_valid_objective(y_pred, data, config)
    logger.info(f'Results: {results_valid}')

    rand_model.eval()
    with torch.no_grad():
        y_pred = (rand_model(data.X_test_gpu)[:, 0] > rand_result['thresh']).reshape(-1).cpu().numpy()
    results_test = get_test_objective(y_pred, data, config)

    return results_valid, results_test
