"""
Main file to run for each experiment with the correct config.yml file as the argument.
"""
import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from aif360.algorithms.postprocessing import (CalibratedEqOddsPostprocessing,
                                              EqOddsPostprocessing,
                                              RejectOptionClassification)
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from utils import get_data, get_valid_objective, get_test_objective
from tabular_models import load_model, train_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

logger = logging.getLogger("Debiasing")
log_handler = logging.StreamHandler()
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.propagate = False


class Data(object):

    def __init__(self, config, seed):
        self.train, self.valid, self.test, self.priv, self.unpriv = get_data(config['dataset'], config['protected'], seed=seed)
        # priv_index is the index of the priviledged column.
        priv_index = self.train.protected_attribute_names.index(list(self.priv[0].keys())[0])

        scale_orig = StandardScaler()
        self.X_train = torch.tensor(scale_orig.fit_transform(self.train.features), dtype=torch.float32)
        self.y_train = torch.tensor(self.train.labels.ravel(), dtype=torch.float32)
        self.p_train = self.train.protected_attributes[:, priv_index]

        self.X_valid = torch.tensor(scale_orig.transform(self.valid.features), dtype=torch.float32)
        self.X_valid_gpu = self.X_valid.to(device)
        self.y_valid = torch.tensor(self.valid.labels.ravel(), dtype=torch.float32)
        self.y_valid_gpu = self.y_valid.to(device)
        self.p_valid = self.valid.protected_attributes[:, priv_index]
        self.p_valid_gpu = torch.tensor(self.p_valid).to(device)

        valid_train_indices, valid_valid_indices = torch.split(torch.randperm(self.X_valid.size(0)), int(0.7*self.X_valid.size(0)))
        self.X_valid_train, self.X_valid_valid = self.X_valid[valid_train_indices, :], self.X_valid[valid_valid_indices, :]
        self.y_valid_train, self.y_valid_valid = self.y_valid[valid_train_indices], self.y_valid[valid_valid_indices]
        self.p_valid_train, self.p_valid_valid = self.p_valid[valid_train_indices], self.p_valid[valid_valid_indices]

        self.X_test = torch.tensor(scale_orig.transform(self.test.features), dtype=torch.float32)
        self.X_test_gpu = self.X_test.to(device)
        self.y_test = torch.tensor(self.test.labels.ravel(), dtype=torch.float32)
        self.y_test_gpu = self.y_test.to(device)
        self.p_test = self.test.protected_attributes[:, priv_index]
        self.p_test_gpu = torch.tensor(self.p_test).to(device)

        self.num_features = self.X_train.size(1)


def main(config):

    seed = np.random.randint(0, high=10000)
    if 'seed' in config:
        seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup directories to save models and results
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    # Get Data
    logger.info(f'Loading Data from dataset: {config["dataset"]}.')
    data = Data(config, seed)

    # Get trained model
    model = load_model(data.num_features, config.get('hyperparameters', {}))
    model_path = (Path('models') / Path(config['modelpath']))
    if model_path.is_file():
        logger.info(f'Loading Model from {model_path}.')
        model.load_state_dict(torch.load(model_path))
    else:
        logger.info(f'{model_path} does not exist. Retraining model from scratch.')
        train_model(model, data, epochs=config.get('epochs', 1001))
        torch.save(model.state_dict(), model_path)
    model_state_dict = copy.deepcopy(model.state_dict())

    # Preliminaries
    logger.info('Setting up preliminaries.')
    model.eval()
    with torch.no_grad():

        valid_pred = data.valid.copy(deepcopy=True)
        valid_pred.scores = model(data.X_valid)[:, 0].reshape(-1, 1).numpy()
        valid_pred.labels = np.array(valid_pred.scores > 0.5)

        test_pred = data.test.copy(deepcopy=True)
        test_pred.scores = model(data.X_test)[:, 0].reshape(-1, 1).numpy()
        test_pred.labels = np.array(test_pred.scores > 0.5)

    results_valid = {}
    results_test = {}

    # Evaluate default model
    if 'default' in config['models']:
        logger.info('Finding best threshold for default model to minimize objective function')
        threshs = np.linspace(0, 1, 1001)
        performances = []
        for thresh in threshs:
            perf = balanced_accuracy_score(data.y_valid, valid_pred.scores > thresh)
            performances.append(perf)
        best_thresh = threshs[np.argmax(performances)]

        logger.info('Evaluating default model with best threshold.')
        results_valid['default'] = get_valid_objective(valid_pred.scores > best_thresh, data, config)
        logger.info(f'Results: {results_valid["default"]}')

        results_test['default'] = get_test_objective(test_pred.scores > best_thresh, data, config)

    # Evaluate ROC
    if 'ROC' in config['models']:
        metric_map = {
            'spd': 'Statistical parity difference',
            'aod': 'Average odds difference',
            'eod': 'Equal opportunity difference'
        }
        ROC = RejectOptionClassification(unprivileged_groups=data.unpriv,
                                         privileged_groups=data.priv,
                                         low_class_thresh=0.01, high_class_thresh=0.99,
                                         num_class_thresh=100, num_ROC_margin=50,
                                         metric_name=metric_map[config['metric']],
                                         metric_ub=0.05, metric_lb=-0.05)

        logger.info('Training ROC model with validation dataset.')
        ROC = ROC.fit(data.valid, valid_pred)

        logger.info('Evaluating ROC model.')
        y_pred = ROC.predict(valid_pred).labels.reshape(-1)
        results_valid['ROC'] = get_valid_objective(y_pred, data, config)
        logger.info(f'Results: {results_valid["ROC"]}')

        y_pred = ROC.predict(test_pred).labels.reshape(-1)
        results_test['ROC'] = get_test_objective(y_pred, data, config)
        ROC = None

    # Evaluate Equality of Odds
    if 'EqOdds' in config['models']:
        eqodds = EqOddsPostprocessing(privileged_groups=data.priv,
                                      unprivileged_groups=data.unpriv)

        logger.info('Training Equality of Odds model with validation dataset.')
        eqodds = eqodds.fit(data.valid, valid_pred)

        logger.info('Evaluating Equality of Odds model.')
        y_pred = eqodds.predict(valid_pred).labels.reshape(-1)
        results_valid['EqOdds'] = get_valid_objective(y_pred, data, config)
        logger.info(f'Results: {results_valid["EqOdds"]}')

        y_pred = eqodds.predict(test_pred).labels.reshape(-1)
        results_test['EqOdds'] = get_test_objective(y_pred, data, config)
        eqodds = None

    # Evaluate Calibrated Equality of Odds
    if 'CalibEqOdds' in config['models']:
        cost_constraint = config['CalibEqOdds']['cost_constraint']

        cpp = CalibratedEqOddsPostprocessing(privileged_groups=data.priv,
                                             unprivileged_groups=data.unpriv,
                                             cost_constraint=cost_constraint)

        logger.info('Training Calibrated Equality of Odds model with validation dataset.')
        cpp = cpp.fit(data.valid, valid_pred)

        logger.info('Evaluating Calibrated Equality of Odds model.')
        y_pred = cpp.predict(valid_pred).labels.reshape(-1)
        results_valid['CalibEqOdds'] = get_valid_objective(y_pred, data, config)
        logger.info(f'Results: {results_valid["CalibEqOdds"]}')

        y_pred = cpp.predict(test_pred).labels.reshape(-1)
        results_test['CalibEqOdds'] = get_test_objective(y_pred, data, config)

        cpp = None

    # Evaluate Random Debiasing
    if 'random' in config['models']:
        from algorithms.random import random_debiasing
        results_valid['random'], results_test['random'] = random_debiasing(model_state_dict, data, config, device)

    # Evaluate fairBO
    if 'fairBO' in config['models']:
        from algorithms.fairBO import fairBO_debiasing
        results_valid['fairBO'], results_test['fairBO'] = fairBO_debiasing(model_state_dict, data, config, device)

    # Evaluate Layerwise Optimizer
    if 'layerwiseOpt' in config['models']:
        from algorithms.layerwiseOpt import layerwiseOpt_debiasing
        results_valid['layerwiseOpt'], results_test['layerwiseOpt'] = layerwiseOpt_debiasing(model_state_dict, data, config, device)

    # Evaluate Adversarial
    if 'adversarial' in config['models']:
        from algorithms.adversarial import adversarial_debiasing
        results_valid['adversarial'], results_test['adversarial'] = adversarial_debiasing(model_state_dict, data, config, device)

    # Mitigating Unwanted Biases with Adversarial Learning
    if 'mitigating' in config['models']:
        from algorithms.mitigating import mitigating_debiasing
        results_valid['mitigating'], results_test['mitigating'] = mitigating_debiasing(model_state_dict, data, config, device)

    # Save Results
    results_valid['config'] = config
    logger.info(f'Validation Results: {results_valid}')
    logger.info(f'Saving validation results to {config["experiment_name"]}_valid_output.json')
    with open(Path('results') / f'{config["experiment_name"]}_valid_output.json', 'w') as fh:
        json.dump(results_valid, fh)

    results_test['config'] = config
    logger.info(f'Test Results: {results_test}')
    logger.info(f'Saving validation results to {config["experiment_name"]}_test_output.json')
    with open(Path('results') / f'{config["experiment_name"]}_test_output.json', 'w') as fh:
        json.dump(results_test, fh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to configuration yaml file.')
    args = parser.parse_args()
    with open(args.config, 'r') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
