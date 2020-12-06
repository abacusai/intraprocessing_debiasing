import numpy as np
from sklearn.metrics import balanced_accuracy_score


def get_data(dataset, protected_attribute, seed=101):
    def protected_attribute_error():
        raise ValueError(f'protected attribute {protected_attribute} is not available for dataset {dataset}')

    if dataset == 'adult':
        from aif360.datasets import AdultDataset
        dataset_orig = AdultDataset()
        if protected_attribute == 'sex':
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        elif protected_attribute == 'sex_or_race':
            dataset_orig.feature_names += ['sex_or_race']
            dataset_orig.features = np.hstack([dataset_orig.features, np.expand_dims(np.logical_or(*dataset_orig.features[:, [2, 3]].T).astype(np.float64), -1)])
            dataset_orig.protected_attributes = np.hstack([dataset_orig.protected_attributes, dataset_orig.features[:, [-1]]])
            dataset_orig.protected_attribute_names += ['sex_or_race']
            dataset_orig.privileged_protected_attributes += [np.array([1.])]
            dataset_orig.unprivileged_protected_attributes += [np.array([0.])]
            privileged_groups = [{'sex_or_race': 1}]
            unprivileged_groups = [{'sex_or_race': 0}]
        elif protected_attribute == 'race':
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        else:
            protected_attribute_error()

    elif dataset == 'german':
        from aif360.datasets import GermanDataset
        dataset_orig = GermanDataset()
        if protected_attribute == 'sex':
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        elif protected_attribute == 'age':
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
        else:
            protected_attribute_error()

    elif dataset == 'compas':
        from aif360.datasets import CompasDataset
        dataset_orig = CompasDataset()
        if protected_attribute == 'sex':
            privileged_groups = [{'sex': 0}]
            unprivileged_groups = [{'sex': 1}]
        elif protected_attribute == 'sex_or_race':
            dataset_orig.feature_names += ['sex_or_race']
            dataset_orig.features = np.hstack([dataset_orig.features, np.expand_dims(np.logical_or(*dataset_orig.features[:, [0, 2]].T).astype(np.float64), -1)])
            dataset_orig.protected_attributes = np.hstack([dataset_orig.protected_attributes, dataset_orig.features[:, [-1]]])
            dataset_orig.protected_attribute_names += ['sex_or_race']
            dataset_orig.privileged_protected_attributes += [np.array([1.])]
            dataset_orig.unprivileged_protected_attributes += [np.array([0.])]
            privileged_groups = [{'sex_or_race': 1}]
            unprivileged_groups = [{'sex_or_race': 0}]
        elif protected_attribute == 'race':
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        else:
            protected_attribute_error()

    elif dataset == 'bank':
        from aif360.datasets import BankDataset
        dataset_orig = BankDataset()
        if protected_attribute == 'age':
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
        else:
            protected_attribute_error()

    else:
        raise ValueError(f'{dataset} is not an available dataset.')

    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True, seed=seed)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True, seed=seed)

    return dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups


def compute_bias(y_pred, y_true, priv, metric):
    def zero_if_nan(x):
        return 0. if np.isnan(x) else x

    gtpr_priv = zero_if_nan(y_pred[priv * y_true == 1].mean())
    gfpr_priv = zero_if_nan(y_pred[priv * (1-y_true) == 1].mean())
    mean_priv = zero_if_nan(y_pred[priv == 1].mean())

    gtpr_unpriv = zero_if_nan(y_pred[(1-priv) * y_true == 1].mean())
    gfpr_unpriv = zero_if_nan(y_pred[(1-priv) * (1-y_true) == 1].mean())
    mean_unpriv = zero_if_nan(y_pred[(1-priv) == 1].mean())

    if metric == 'spd':
        return mean_unpriv - mean_priv
    elif metric == 'aod':
        return 0.5 * ((gfpr_unpriv - gfpr_priv) + (gtpr_unpriv - gtpr_priv))
    elif metric == 'eod':
        return gtpr_unpriv - gtpr_priv


def objective_function(bias, performance, lam=0.75):
    return - lam*abs(bias) - (1-lam)*(1-performance)


def sharp_objective_function(bias, performance, sharpness=500., epsilon=0.05):
    def sigmoid(value, sharpness, epsilon):
        return 1. / (1. + np.exp(sharpness*(np.abs(value)-epsilon)))
    return sigmoid(bias, sharpness, epsilon) * performance


def threshold_objective_function(bias, performance, epsilon=0.05):
    if abs(bias) < epsilon:
        return performance
    else:
        return 0.0


def get_objective(y_pred, y_true, priv, metric, sharpness=500., epsilon=0.05, kind='threshold'):
    bias = compute_bias(y_pred, y_true, priv, metric)
    performance = balanced_accuracy_score(y_true, y_pred)
    if kind == 'default':
        objective = objective_function(bias, performance, epsilon)
    elif kind == 'sharp':
        objective = sharp_objective_function(bias, performance, sharpness, epsilon)
    elif kind == 'threshold':
        objective = threshold_objective_function(bias, performance, epsilon)
    else:
        raise ValueError(f'objective function of kind {kind} is not available.')
    return {'objective': objective, 'bias': bias, 'performance': performance}


def get_valid_objective(y_pred, data, config, valid=False, margin=0.00, num_samples=5):
    y_val = data.y_valid_valid if valid else data.y_valid
    p_val = data.p_valid_valid if valid else data.p_valid
    indices = np.random.choice(np.arange(y_pred.size), num_samples*y_pred.size, replace=True).reshape(num_samples, y_pred.size)
    results = {'objective': [], 'bias': [], 'performance': []}
    for index in indices:
        result = get_objective(y_pred[index], y_val.numpy()[index], p_val[index],
                               config['metric'], config['objective']['sharpness'], config['objective']['epsilon'] - margin)
        results = {k: v+[result[k]] for k, v in results.items()}
    return {k: np.mean(v) for k, v in results.items()}


def get_test_objective(y_pred, data, config):
    return get_objective(y_pred, data.y_test.numpy(), data.p_test,
                         config['metric'], config['objective']['sharpness'], config['objective']['epsilon'])


def get_best_thresh(scores, threshs, data, config, valid=False, margin=0.00):
    objectives = []
    for thresh in threshs:
        valid_objective = get_valid_objective(scores > thresh, data, config, valid=valid, margin=margin)
        objectives.append(valid_objective['objective'])
    return threshs[np.argmax(objectives)], np.max(objectives)
