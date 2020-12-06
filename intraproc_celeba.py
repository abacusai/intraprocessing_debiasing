"""
post_hoc_celeba.py

Debias image models trained on celeba
"""
import argparse
import copy
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from aif360.algorithms.postprocessing import (CalibratedEqOddsPostprocessing,
                                              EqOddsPostprocessing,
                                              RejectOptionClassification)
from aif360.datasets import StandardDataset
from sklearn.metrics import roc_auc_score
from skopt import gbrt_minimize
from skopt.space import Real
from torchvision import models, transforms

from celeba_race import CelebRace, unambiguous

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


logger = logging.getLogger("Debiasing CelebA")
log_stream_handler = logging.StreamHandler()
log_file_handler = logging.FileHandler('posthoc_celeba.log')
logger.addHandler(log_stream_handler)
logger.addHandler(log_file_handler)
logger.setLevel(logging.INFO)
log_stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.propagate = False


descriptions = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
                'Young', 'White', 'Black', 'Asian', 'Index', 'Female']


def load_celeba(input_size=224, num_workers=2, trainsize=100, testsize=100, batch_size=4, transform_type='normalize'):
    """Load CelebA dataset"""

    if transform_type == 'normalize':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif transform_type == 'augmentation':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.ToTensor()

    trainset = CelebRace(root='./data', download=True, split='train', transform=transform)
    testset = CelebRace(root='./data', download=True, split='test', transform=transform)

    # return only the images which were predicted white, black, or asian by >70%.
    trainset = unambiguous(trainset, split='train')
    testset = unambiguous(testset, split='test')

    if trainsize >= 0:
        # cut down the training set
        trainset, _ = torch.utils.data.random_split(trainset, [trainsize, len(trainset) - trainsize])
    trainset, valset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.6), int(len(trainset)*0.4)])
    if testsize >= 0:
        testset, _ = torch.utils.data.random_split(testset, [testsize, len(testset) - testsize])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainset, valset, testset, trainloader, valloader, testloader


def get_resnet_model():
    """Get Pretrained resnet model"""
    resnet18 = models.resnet18(pretrained=True)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 2)
    resnet18.to(device)
    return resnet18


def train_model(model, trainloader, valloader, criterion, optimizer, checkpoint, protected_index, prediction_index, epochs=2, start_epoch=0):
    """Fine-tune resnet model on dataset"""
    best_acc, best_model, patience = 0., None, 10
    for epoch in range(start_epoch, epochs):
        logger.info('Epoch {}/{}'.format(epoch+1, epochs))
        logger.info('-' * 10)

        model.train()

        running_loss = 0.
        running_corrects = 0

        for index, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), (labels[:, prediction_index]).float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs[:, 0], labels)

            preds = torch.sigmoid(outputs[:, 0]) > 0.5

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if (index-1) % 101 == 0:
                num_examples = index * inputs.size(0)
                print(f"({index}/{len(trainloader)}) Loss: {running_loss / num_examples:.4f} Acc: {running_corrects.float() / num_examples:.4f}")

        acc, _ = val_model(model, valloader, get_best_balanced_accuracy, protected_index, prediction_index)
        if acc < best_acc:
            patience -= 1
            if patience <= 0:
                model.load_state_dict(best_model)
        else:
            best_acc = acc
            best_model = model.state_dict()
            patience = 10
        logger.info(f"Best Accuracy on Validation set: {best_acc}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint)
        if patience <= 0:
            break


def val_model(model, loader, criterion, protected_index, prediction_index):
    """Validate model on loader with criterion function"""
    y_true, y_pred, y_prot = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, full_labels in loader:
            inputs, labels, protected = inputs.to(device), full_labels[:, prediction_index].float().to(device), full_labels[:, protected_index].float().to(device)
            y_true.append(labels)
            y_prot.append(protected)
            y_pred.append(torch.sigmoid(model(inputs)[:, 0]))
    y_true, y_pred, y_prot = torch.cat(y_true), torch.cat(y_pred), torch.cat(y_prot)
    return criterion(y_true, y_pred, y_prot)


def compute_priors(data, protected_index, prediction_index):
    """Compute priors on the data"""
    counts = np.zeros((2, 2))
    for batch in list(data):
        _, labels = batch[0], batch[1]

        for label in labels:
            prot_value = label[protected_index]
            pred_value = label[prediction_index]
            counts[prot_value][pred_value] += 1
    total = sum(sum(counts))

    prot_rate = np.round(counts[1][1]/sum(counts[1]), 4)
    unprot_rate = np.round(counts[0][1]/sum(counts[0]), 4)

    print('Prob. protected class:', np.round(sum(counts[1])/total, 4))
    print('Prob. positive outcome:', np.round(sum(counts[:, 1])/total, 4))
    print('Prob. positive outcome given protected class', prot_rate)
    print('Prob. positive outcome given unprotected class', unprot_rate)


def compute_bias(y_pred, y_true, prot, metric):
    """Compute bias on the dataset"""
    def zero_if_nan(data):
        """Zero if there is a nan"""
        return 0. if torch.isnan(data) else data

    gtpr_prot = zero_if_nan(y_pred[prot * y_true == 1].mean())
    gfpr_prot = zero_if_nan(y_pred[prot * (1-y_true) == 1].mean())
    mean_prot = zero_if_nan(y_pred[prot == 1].mean())

    gtpr_unprot = zero_if_nan(y_pred[(1-prot) * y_true == 1].mean())
    gfpr_unprot = zero_if_nan(y_pred[(1-prot) * (1-y_true) == 1].mean())
    mean_unprot = zero_if_nan(y_pred[(1-prot) == 1].mean())

    if metric == "spd":
        return mean_prot - mean_unprot
    elif metric == "aod":
        return 0.5 * ((gfpr_prot - gfpr_unprot) + (gtpr_prot - gtpr_unprot))
    elif metric == "eod":
        return gtpr_prot - gtpr_unprot


def get_best_accuracy(y_true, y_pred, _):
    """Select threshold that maximizes accuracy"""
    threshs = torch.linspace(0, 1, 1001)
    best_perf, best_thresh = 0., 0.
    for thresh in threshs:
        perf = (torch.mean((y_pred > thresh)[y_true.type(torch.bool)].type(torch.float32)))
        if perf > best_perf:
            best_perf, best_thresh = perf, thresh
    return best_perf, best_thresh


def get_best_balanced_accuracy(y_true, y_pred, _):
    """Select threshold that maximizes accuracy"""
    threshs = torch.linspace(0, 1, 1001)
    best_perf, best_thresh = 0., 0.
    for thresh in threshs:
        perf = (torch.mean((y_pred > thresh)[y_true.type(torch.bool)].type(torch.float32)) + torch.mean((y_pred <= thresh)[~y_true.type(torch.bool)].type(torch.float32))) / 2
        if perf > best_perf:
            best_perf, best_thresh = perf, thresh
    return best_perf, best_thresh


def compute_objective(performance, bias, epsilon=0.05, margin=0.01):
    if abs(bias) <= (epsilon-margin):
        return performance
    else:
        return 0.0


def get_objective_with_best_accuracy(y_true, y_pred, y_prot):
    """Get objective for best accuracy threshold"""
    global yaml_config
    rocauc_score = roc_auc_score(y_true.cpu(), y_pred.cpu())
    perf, best_thresh = get_best_balanced_accuracy(y_true, y_pred, y_prot)
    bias = compute_bias((y_pred > best_thresh).float().cpu(), y_true.float().cpu(), y_prot.float().cpu(), yaml_config['metric'])
    obj = compute_objective(perf, bias)
    return rocauc_score, perf, bias, obj


def get_best_objective(y_true, y_pred, y_prot):
    """Find the threshold for the best objective"""
    global yaml_config
    num_samples = 5
    threshs = torch.linspace(0, 1, 501)
    best_obj, best_thresh = -math.inf, 0.
    for thresh in threshs:
        indices = np.random.choice(np.arange(y_pred.size()[0]), num_samples*y_pred.size()[0], replace=True).reshape(num_samples, y_pred.size()[0])
        objs = []
        for index in indices:
            y_pred_tmp = y_pred[index]
            y_true_tmp = y_true[index]
            y_prot_tmp = y_prot[index]
            perf = (torch.mean((y_pred_tmp > thresh)[y_true_tmp.type(torch.bool)].type(torch.float32)) + torch.mean((y_pred_tmp <= thresh)[~y_true_tmp.type(torch.bool)].type(torch.float32))) / 2
            bias = compute_bias((y_pred_tmp > thresh).float().cpu(), y_true_tmp.float().cpu(), y_prot_tmp.float().cpu(), yaml_config['metric'])
            objs.append(compute_objective(perf, bias))
        obj = float(torch.tensor(objs).mean())
        if obj > best_obj:
            best_obj, best_thresh = obj, thresh

    return best_obj, best_thresh


def get_objective_results(best_thresh):
    """Get the objective results with the best_threshold"""
    def _get_results(y_true, y_pred, y_prot):
        """Inner function to be returned"""
        global yaml_config
        rocauc_score = roc_auc_score(y_true.cpu(), y_pred.cpu())
        perf = (torch.mean((y_pred > best_thresh)[y_true.type(torch.bool)].type(torch.float32)) + torch.mean((y_pred <= best_thresh)[~y_true.type(torch.bool)].type(torch.float32))) / 2
        bias = compute_bias((y_pred > best_thresh).float().cpu(), y_true.float().cpu(), y_prot.float().cpu(), yaml_config['metric'])
        obj = compute_objective(perf, bias)

        return rocauc_score, perf, bias, obj
    return _get_results


def print_objective_results(dataloader, model, thresh, protected_index, prediction_index):
    global yaml_config
    rocauc_score, acc, bias, obj = val_model(model, dataloader, get_objective_results(thresh), protected_index, prediction_index)

    print('roc auc', rocauc_score)
    print('accuracy with best thresh', acc)
    print(yaml_config['metric'], float(bias))
    print('objective', float(obj))

    result_dict = {
        'roc_auc': float(rocauc_score),
        'accuracy': float(acc),
        'bias': float(bias),
        'objective': float(obj)
    }

    return result_dict


class Critic(nn.Module):
    """Critic class for adversarial debiasing method"""

    def __init__(self, sizein, num_deep=3, hid=32):
        super().__init__()
        self.fc0 = nn.Linear(sizein, hid)
        self.fcs = nn.ModuleList([nn.Linear(hid, hid) for _ in range(num_deep)])
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hid, 1)

    def forward(self, t):
        t = t.reshape(1, -1)
        t = self.fc0(t)
        for fully_connected in self.fcs:
            t = F.relu(fully_connected(t))
            t = self.dropout(t)
        return self.out(t)


def main(config):
    """Main Function"""

    seed = np.random.randint(0, high=10000)
    if 'seed' in config:
        seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    protected_index = descriptions.index(config['protected_attr'])
    prediction_index = descriptions.index(config['prediction_attr'])
    valid_results, test_results = {}, {}

    _, _, _, trainloader, valloader, testloader = load_celeba(
        trainsize=config['trainsize'],
        testsize=config['testsize'],
        num_workers=config['num_workers'],
        batch_size=config['batch_size']
    )
    if config['print_priors']:
        logger.info('train priors')
        compute_priors(trainloader, protected_index, prediction_index)
        logger.info('val priors')
        compute_priors(valloader, protected_index, prediction_index)
        logger.info('test priors')
        compute_priors(testloader, protected_index, prediction_index)

    net = get_resnet_model()
    criterion = nn.BCEWithLogitsLoss()
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=config['lr'])
    else:
        optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    checkpoint_file = Path('seed'+str(seed)+config['optimizer']+str(config['lr'])+'_pro_'+config['protected_attr']+'_pre_'+config['prediction_attr']+config['checkpoint'])

    start_epoch = 0
    if checkpoint_file.is_file() and (not config['retrain']):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        logger.info('loaded from %s', checkpoint_file)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        train_model(
            net,
            trainloader,
            valloader,
            criterion,
            optimizer,
            'seed'+str(seed)+config['metric']+config['optimizer']+str(config['lr'])+'_pro_'+config['protected_attr']+'_pre_'+config['prediction_attr']+config['checkpoint'],
            protected_index,
            prediction_index,
            epochs=config['epochs'],
            start_epoch=start_epoch
        )

    _, best_thresh = val_model(net, valloader, get_best_balanced_accuracy, protected_index, prediction_index)

    print('val_results, thresh', best_thresh.item())
    valid_results['base_model'] = print_objective_results(valloader, net, best_thresh, protected_index, prediction_index)
    print()
    print('test_results')
    result_dict = print_objective_results(testloader, net, best_thresh, protected_index, prediction_index)
    print()
    test_results['base_model'] = result_dict

    def to_dataframe(y_true, y_pred, y_prot):
        y_true, y_pred, y_prot = y_true.float().cpu().numpy(), y_pred.float().cpu().numpy(), y_prot.float().cpu().numpy()
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'y_prot': y_prot})
        dataset = StandardDataset(df, 'y_true', [1.], ['y_prot'], [[1.]])
        dataset.scores = y_pred.reshape(-1, 1)
        return dataset

    val_dataset = val_model(net, valloader, to_dataframe, protected_index, prediction_index)
    test_dataset = val_model(net, testloader, to_dataframe, protected_index, prediction_index)

    def eval_aif360_algorithm(y_pred, dataset, verbose=True):
        global yaml_config
        acc = float(np.mean(y_pred == dataset.labels.reshape(-1)))
        bias = compute_bias(
            torch.tensor(y_pred),
            torch.tensor(dataset.labels.reshape(-1)),
            torch.tensor(dataset.protected_attributes.reshape(-1)),
            yaml_config['metric']
        ).item()
        obj = compute_objective(acc, bias)

        if verbose:
            print('accuracy ', acc)
            print(yaml_config['metric'], bias)
            print('objective', obj)

        return {
            'roc_auc': None,
            'accuracy': float(acc),
            'bias': float(bias),
            'objective': float(obj)
        }

    # Evaluate ROC
    if "ROC" in config['models']:
        ROC = RejectOptionClassification(unprivileged_groups=[{'y_prot': 1.}],
                                         privileged_groups=[{'y_prot': 0.}],
                                         low_class_thresh=0.01, high_class_thresh=0.99,
                                         num_class_thresh=100, num_ROC_margin=50,
                                         metric_name="Average odds difference",
                                         metric_ub=0.05, metric_lb=-0.05)

        print("Training ROC model with validation dataset.")
        ROC = ROC.fit(val_dataset, val_dataset)

        print("Evaluating ROC model.")

        print('ROC val results')
        val_y_pred = ROC.predict(val_dataset).labels.reshape(-1)
        valid_results['ROC'] = eval_aif360_algorithm(val_y_pred, val_dataset)
        print()

        print('ROC test results')
        test_y_pred = ROC.predict(test_dataset).labels.reshape(-1)
        test_results['ROC'] = eval_aif360_algorithm(test_y_pred, test_dataset)
        print()
        ROC = None

    if 'EqOdds' in config['models']:
        eo = EqOddsPostprocessing(privileged_groups=[{'y_prot': 0.}],
                                  unprivileged_groups=[{'y_prot': 1.}])

        print("Training Equality of Odds model with validation dataset.")
        eo = eo.fit(val_dataset, val_dataset)

        print("Evaluating Equality of Odds model.")

        print('Equality of Odds val results')
        val_y_pred = eo.predict(val_dataset).labels.reshape(-1)
        valid_results['EqOdds'] = eval_aif360_algorithm(val_y_pred, val_dataset)
        print()

        print('Equality of Odds test results')
        test_y_pred = eo.predict(test_dataset).labels.reshape(-1)
        test_results['EqOdds'] = eval_aif360_algorithm(test_y_pred, test_dataset)
        print()

        eo = None

    if 'CalibEqOdds' in config['models']:
        cost_constraint = config['CalibEqOdds']['cost_constraint']

        cpp = CalibratedEqOddsPostprocessing(privileged_groups=[{'y_prot': 0.}],
                                             unprivileged_groups=[{'y_prot': 1.}],
                                             cost_constraint=cost_constraint)

        print("Training Calibrated Equality of Odds model with validation dataset.")
        cpp = cpp.fit(val_dataset, val_dataset)

        print("Evaluating Calibrated Equality of Odds model.")

        print('Calibrated Equality of Odds val results')
        valid_y_pred = cpp.predict(val_dataset).labels.reshape(-1)
        valid_results['CalibEqOdds'] = eval_aif360_algorithm(valid_y_pred, val_dataset)
        print()

        print('Equality of Odds test results')
        test_y_pred = cpp.predict(test_dataset).labels.reshape(-1)
        test_results['CalibEqOdds'] = eval_aif360_algorithm(valid_y_pred, val_dataset)
        print()
        cpp = None

    if 'random' in config['models']:
        rand_result = [-np.inf, None, -1]

        for iteration in range(101):
            rand_model = copy.deepcopy(net)
            rand_model.to(device)
            for param in rand_model.parameters():
                param.data = param.data * (torch.randn_like(param) * 0.1 + 1)

            rand_model.eval()
            best_obj, best_thresh = val_model(rand_model, valloader, get_best_objective, protected_index, prediction_index)
            print('iteration', iteration, 'obj', float(best_obj))

            if best_obj > rand_result[0]:
                print('found new best')
                del rand_result[1]
                rand_result = [best_obj, copy.deepcopy(rand_model.state_dict()), best_thresh]

            if iteration % 10 == 0:
                print(f"{iteration} / 101 trials have been sampled.")
                print('current best obj', float(rand_result[0]))

        # evaluate best random model
        best_model = copy.deepcopy(net)
        best_model.load_state_dict(rand_result[1])
        best_model.to(device)
        best_thresh = rand_result[2]

        print('val_results')
        valid_results['random'] = print_objective_results(valloader, best_model, best_thresh, protected_index, prediction_index)
        print()
        print('test_results')
        test_results['random'] = print_objective_results(testloader, best_model, best_thresh, protected_index, prediction_index)
        print()

        torch.save(best_model.state_dict(), 'seed'+str(seed)+config['metric']+config['optimizer']+str(config['lr']) +
                   '_pro_'+config['protected_attr']+'_pre_'+config['prediction_attr']+config['random']['checkpoint'])

    if 'layerwiseOpt' in config['models']:
        base_model = copy.deepcopy(net)
        best_state_dict, best_thresh, best_obj = None, None, np.inf

        total_params = len(list(base_model.parameters()))
        for index, param in enumerate(base_model.parameters()):
            if index < total_params-config['layerwiseOpt']['num_layers']:
                continue
            print(f'Evaluating param number {index} of {total_params}')
            param_copy = copy.deepcopy(param)

            def objective(new_param):
                param.data[indices] = torch.tensor(new_param).to(device)
                base_model.eval()
                best_obj, thresh = val_model(base_model, valloader, get_best_objective, protected_index, prediction_index)
                print(f'Evaluating param number {index} of {total_params}')
                # print('VALIDATION')
                # print_objective_results(valloader, base_model, thresh, protected_index, prediction_index)
                # print('TEST')
                # print_objective_results(testloader, base_model, thresh, protected_index, prediction_index)
                # print()
                return -float(best_obj)

            # mean = param.flatten().cpu().detach().numpy().mean()
            std = param.flatten().cpu().detach().numpy().std()
            num_elems = param.size().numel()
            ratio = min(1., config['layerwiseOpt']['max_sparsity'] / num_elems)
            indices = torch.rand(param.size()) < ratio
            space = [Real(float(x.cpu().detach()) - 2.2*std, float(x.cpu().detach()) + 2.2*std) for x in param[indices]]
            print(f'Number of sparse indices: {indices.sum().item()}')
            res_gbrt = gbrt_minimize(
                objective,
                space,
                n_calls=20,
                verbose=True
            )

            if res_gbrt.fun < best_obj:
                param.data[indices] = torch.tensor(res_gbrt.x).to(device)
                best_state_dict = copy.deepcopy(base_model.state_dict())
                best_obj, best_thresh = val_model(base_model, valloader, get_best_objective, protected_index, prediction_index)
            param.data = param_copy.data

        best_model = copy.deepcopy(net)
        best_model.load_state_dict(best_state_dict)
        best_model.to(device)

        print('val_results')
        valid_results['layerwiseOpt'] = print_objective_results(valloader, best_model, best_thresh, protected_index, prediction_index)
        print()
        print('test_results')
        test_results['layerwiseOpt'] = print_objective_results(testloader, best_model, best_thresh, protected_index, prediction_index)
        print()

    if 'adversarial' in config['models']:
        unrefined_net = get_resnet_model()
        base_model = copy.deepcopy(unrefined_net)
        base_model.fc = nn.Linear(base_model.fc.in_features, base_model.fc.in_features)

        actor = nn.Sequential(base_model, nn.Linear(base_model.fc.in_features, 2))
        actor.to(device)
        actor_optimizer = optim.Adam(actor.parameters())
        actor_loss_fn = nn.BCEWithLogitsLoss()
        actor_loss = 0.
        actor_steps = config['adversarial']['actor_steps']

        critic = Critic(config['batch_size']*unrefined_net.fc.in_features)
        critic.to(device)
        critic_optimizer = optim.Adam(critic.parameters())
        critic_loss_fn = nn.MSELoss()
        critic_loss = 0.
        critic_steps = config['adversarial']['critic_steps']

        for epoch in range(config['adversarial']['epochs']):
            for param in critic.parameters():
                param.requires_grad = True
            for param in actor.parameters():
                param.requires_grad = False
            actor.eval()
            critic.train()
            for step, (inputs, labels) in enumerate(valloader):
                if step > critic_steps:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.size(0) != config['batch_size']:
                    continue
                critic_optimizer.zero_grad()

                with torch.no_grad():
                    y_pred = actor(inputs)

                y_true = labels[:, prediction_index].float().to(device)
                y_prot = labels[:, protected_index].float().to(device)

                bias = compute_bias(y_pred, y_true, y_prot, config['metric'])
                res = critic(base_model(inputs))
                loss = critic_loss_fn(bias.unsqueeze(0), res[0])
                loss.backward()
                critic_loss += loss.item()
                critic_optimizer.step()
                if step % 100 == 0:
                    print_loss = critic_loss if (epoch*critic_steps + step) == 0 else critic_loss / (epoch*critic_steps + step)
                    print(f'=======> Epoch: {(epoch, step)} Critic loss: {print_loss:.3f}')

            for param in critic.parameters():
                param.requires_grad = False
            for param in actor.parameters():
                param.requires_grad = True
            actor.train()
            critic.eval()
            for step, (inputs, labels) in enumerate(valloader):
                if step > actor_steps:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.size(0) != config['batch_size']:
                    continue
                actor_optimizer.zero_grad()

                y_true = labels[:, prediction_index].float().to(device)
                y_prot = labels[:, protected_index].float().to(device)

                est_bias = critic(base_model(inputs))
                loss = actor_loss_fn(actor(inputs)[:, 0], y_true)

                # todo change this to the sharpness loss function
                loss = max(1, 10*(abs(est_bias)-config['objective']['epsilon']+config['adversarial']['margin'])+1) * loss
                # loss = lam*abs(est_bias) + (1-lam)*loss

                loss.backward()
                actor_loss += loss.item()
                actor_optimizer.step()
                if step % 100 == 0:
                    print_loss = critic_loss if (epoch*actor_steps + step) == 0 else critic_loss / (epoch*actor_steps + step)
                    print(f'=======> Epoch: {(epoch, step)} Actor loss: {print_loss:.3f}')

        _, best_thresh = val_model(actor, valloader, get_best_objective, protected_index, prediction_index)

        print('val_results')
        valid_results['adversarial'] = print_objective_results(valloader, actor, best_thresh, protected_index, prediction_index)
        print()
        print('test_results')
        test_results['adversarial'] = print_objective_results(testloader, actor, best_thresh, protected_index, prediction_index)
        print()

        torch.save(actor.state_dict(), config['adversarial']['checkpoint'])

    with open('valid_seed'+str(seed)+config['metric']+config['optimizer']+str(config['lr'])+'_pro_'+config['protected_attr']+'_pre_'+config['prediction_attr']+config['output'], 'w') as filehandler:
        json.dump(valid_results, filehandler)
    with open('test_seed'+str(seed)+config['metric']+config['optimizer']+str(config['lr'])+'_pro_'+config['protected_attr']+'_pre_'+config['prediction_attr']+config['output'], 'w') as filehandler:
        json.dump(test_results, filehandler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for CelebA experiments')
    parser.add_argument("config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    global yaml_config
    with open(args.config, 'r') as fh:
        yaml_config = yaml.load(fh, Loader=yaml.FullLoader)

    main(yaml_config)
