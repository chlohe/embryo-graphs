import torch
import torch.nn.functional as F
import numpy as np
import optuna
import wandb
import utils

from multiprocessing import Manager
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SubsetRandomSampler
from torchvision import transforms
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from models import BaselineCNN
from datasets import EmbryoDatasetWithAdjacency, BottleneckDataset

LOGGING_ENABLED = False

def train_one_epoch(model, loader, optimizer, device, pos_classes=['live birth'], transforms=None):
    model.train()
    losses = []
    for batch in loader:
        stack, target = batch['stack'], batch['label']
        stack = stack.to(device).float()
        target = torch.tensor([0 if x in pos_classes else 1 for x in target])
        target = target.to(device)
        
        if transforms is not None:
            stack = transforms(stack)

        # Forward pass
        pred = model(stack)
        # Get loss
        loss = F.cross_entropy(pred, target)
        losses.append(loss.cpu().detach().item())

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if LOGGING_ENABLED:
        wandb.log({'loss': np.mean(losses)})
    return model

def eval_model(model, loader, device, pos_classes=['live birth']):
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        stack, target = batch['stack'], batch['label']
        stack = stack.to(device).float()
        target = [0 if x in pos_classes else 1 for x in target]
        targets.extend(target)

        # Forward pass
        pred = model(stack)
        preds.extend(pred.detach().argmax(dim=1).cpu().tolist())
    # Compute metrics
    return {
        'acc': metrics.accuracy_score(targets, preds),
        'prec': metrics.precision_score(targets, preds, pos_label=0),
        'sens': metrics.recall_score(targets, preds, pos_label=0),
        'spec': metrics.recall_score(targets, preds, pos_label=1),
        'f1': metrics.f1_score(targets, preds, pos_label=0),
        'auc': metrics.roc_auc_score(targets, preds)
    }

def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def train_cv(epochs, device, transforms=None, lr=1e-2, weight_decay=1e-4, num_folds=5, num_repeats=1, keep_one_fold_for_testing=True, 
             model_selection_metric='auc', pos_classes=['live birth'], show_progress=True):
    train_results = []
    val_results = []
    models = []
    for i in range(num_repeats):
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
        dataset = BottleneckDataset()
        targets = [0 if x['label'] in pos_classes else 1 for x in dataset]
        for train_idxs, val_idxs in kfold.split(dataset, targets):
            train_sampler = SubsetRandomSampler(train_idxs)
            val_sampler = SubsetRandomSampler(val_idxs)
            # Load data
            loader_train = DataLoader(dataset, batch_size=16, sampler=train_sampler)
            loader_val = DataLoader(dataset, batch_size=16, sampler=val_sampler)
            
            # Set up model
            model = BaselineCNN().to(device)
            model.preprocess = torch.nn.Identity()
            model.net.conv1 = torch.nn.Identity()
            model.net.bn1 = torch.nn.Identity()
            model.net.relu = torch.nn.Identity()
            model.net.maxpool = torch.nn.Identity()
            model.net.layer1 = torch.nn.Identity()
            model.net.layer2 = torch.nn.Identity()
            model.net.layer3 = torch.nn.Identity()
            model.net.layer4 = torch.nn.Identity()
            model.net.avgpool = torch.nn.Identity()
            model.net.fc.apply(init_weights)

            optimizer = torch.optim.Adam(model.net.fc.parameters(), lr=lr, weight_decay=weight_decay)
            for epoch in tqdm(range(epochs)) if show_progress else range(epochs):
                model = train_one_epoch(model, loader_train, optimizer, device)      
            train_results.append(eval_model(model, loader_train, device))
            val_results.append(eval_model(model, loader_val, device))
            models.append(model)
    best_model_idx = val_results.index(min(val_results, key=lambda x: x[model_selection_metric]))
    best_model = models[best_model_idx]
    return best_model, utils.mean_and_std_over_dict(train_results), utils.mean_and_std_over_dict(val_results)

def train_optimism_adjusted_bootstrap(epochs, devices, transforms=None, num_iterations=50, lr=1e-2, weight_decay=1e-4, show_progress=True):
    def fit_and_eval(loader_train, loader_val, device):
        # Set up model
        model = BaselineCNN().to(device)
        model.net.fc.apply(init_weights)
        # Fit model
        optimizer = torch.optim.Adam(list(model.net.fc.parameters()) + list(model.net.layer4.parameters()), lr=lr, weight_decay=weight_decay)
        for epoch in tqdm(range(epochs)) if show_progress else range(epochs):
            model = train_one_epoch(model, loader_train, optimizer, device, transforms=transforms)
        # Eval model 
        return model, eval_model(model, loader_val, device)
    def compute_optimism(device_queue, dataset_train, loader_train, sampler):
        try:
            # Get some compute
            device = device_queue.get()
            # Take bootstrap sample (WITH replacement)
            loader_sample_train = DataLoader(dataset_train, batch_size=8, sampler=sampler)
            # Fit new model and calculate apparent performance on bootstrap sample and compute apparent performance on sample dataset
            model_sample, apparent_performance_sample = fit_and_eval(loader_sample_train, loader_sample_train, device)
            print(apparent_performance_sample)
            # Compute performance on full dataset
            test_performance_sample = eval_model(model_sample, loader_train, device)
            # Compute optimism
            optimism = utils.subtract_dicts(apparent_performance_sample, test_performance_sample)
            return optimism
        except Exception as e:
            print(f'ERROR - {e}')
        finally:
            device_queue.put(device)
    assert len(devices) > 0
    # Fit model with original training set and calculate apparent performance
    dataset_train = EmbryoDatasetWithAdjacency(clinical_endpoint='simplified_outcome')
    loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
    _, apparent_performance_full = fit_and_eval(loader_train, loader_train, devices[0])
    print(apparent_performance_full)
    with Manager() as manager:
        # Init resource pool
        device_queue = manager.Queue()
        for device in devices:
            device_queue.put(device)
        # Compute optimism adjustment
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=len(dataset_train))
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [
                pool.submit(compute_optimism, device_queue, dataset_train, loader_train, sampler) 
                for i in range(num_iterations)
            ]
            wait(futures, return_when=ALL_COMPLETED)
            optimisms = [future.result() for future in futures]
    # Compute optimism adjusted performance
    mean_optimism = utils.mean_over_dict(optimisms)
    ci_optimism = utils.CI_over_dict(optimisms, alpha=0.95)
    optimism_adjusted_performance = utils.subtract_dicts(apparent_performance_full, mean_optimism)
    print(optimisms)
    return optimism_adjusted_performance, ci_optimism

def optimize_hyperparams(trials, epochs, device, repeats=2, no_gpus=2, no_concurrent_processes_per_gpu=4):
    # Define objective
    def objective(trial, gpu_queue):
        # Init search space
        lr_exp = trial.suggest_float('lr_exp', -5, 0)
        wd_exp = trial.suggest_float('wd_exp', -5, 0)
        # Get a GPU
        gpu_id = gpu_queue.get()
        # Do search
        aucs = []
        try:
            for _ in range(repeats):
                _, train_results, val_results = train_cv(epochs, f'cuda:{gpu_id}' if device == 'cuda' else device, 
                                                      lr=10**lr_exp, weight_decay=10**wd_exp, show_progress=False, num_folds=3)
                aucs.append(float(val_results['auc'].split('(')[0])) # TODO: Kinda gross - refactor later
            val_auc = np.mean(aucs)
            return 1-val_auc
        finally:
            gpu_queue.put(gpu_id)
    # Do optimisation
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3"
    )
    # Distribute across multiple GPUs with no_concurrent_processes_per_gpu on each GPU.
    with Manager() as manager:
        gpu_queue = manager.Queue()
        for i in range(no_gpus):
            for j in range(no_concurrent_processes_per_gpu):
                gpu_queue.put(i)
        study.optimize(lambda trial: objective(trial, gpu_queue), n_trials=trials, n_jobs=no_concurrent_processes_per_gpu*no_gpus)
    print(study.best_params)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # optimize_hyperparams(100, 20, device, repeats=3)
    # Do training
    best_model, train_results, val_results = train_cv(
        epochs=20, 
        device=device,
        transforms=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(360),
            transforms.RandomResizedCrop(size=(512,512), scale=(0.5, 1))
        ]),
        lr=10**-4.364408081368803,
        weight_decay=10**-3.2682086406925426,
        num_folds=5,
        num_repeats=10,
        keep_one_fold_for_testing=True,
        model_selection_metric='auc',
        show_progress=True
    )
    print('Training', train_results)
    print('Validation', val_results)
    torch.save(best_model.state_dict(), 'model_baseline.ckpt')