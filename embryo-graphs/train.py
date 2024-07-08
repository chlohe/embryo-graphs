import optuna
import torch
import torch.nn.functional as F
import torch_geometric
import utils

from torch_geometric.loader import DataLoader
from torch.utils.data import RandomSampler, SubsetRandomSampler
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Manager
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from itertools import product

from datasets import *
from models import GNN

def train_one_epoch(model, loader, optimizer, device, pos_classes=['live birth']):
    model.train()
    # epoch_losses = []
    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        target = torch.tensor(
            [[1, 0] if label in pos_classes else [0, 1] for label in batch.y]
        ).to(device).float()

        # Forward pass
        pred = model(x, edge_index, batch.batch)

        # Get loss
        loss = F.cross_entropy(pred, target)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # print(np.mean(epoch_losses))
    return model

def compute_metrics(targets, preds):
    return {
        'acc': metrics.accuracy_score(targets, preds),
        'prec': metrics.precision_score(targets, preds, pos_label=0),
        'sens': metrics.recall_score(targets, preds, pos_label=0),
        'spec': metrics.recall_score(targets, preds, pos_label=1),
        'f1': metrics.f1_score(targets, preds, pos_label=0),
        'auc': metrics.roc_auc_score(targets, preds)
    }

def predict(model, loader, device, pos_classes):
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        target = [0 if label in pos_classes else 1 for label in batch.y]
        targets.extend(target)

        # Forward pass
        pred = model(x, edge_index, batch.batch)
        preds.extend(pred.detach().argmax(dim=1).cpu().tolist())
    return preds, targets


def eval_model(model, loader, device, pos_classes=['live birth']):
    preds, targets = predict(model, loader, device, pos_classes)
    # Compute metrics
    return compute_metrics(targets, preds)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_cv(epochs, device, lr=1e-2, weight_decay=1e-4, hidden_size=32, num_convs=0, num_folds=10, num_repeats=1, keep_one_fold_for_testing=True, 
             model_selection_metric='auc', pos_classes=['live birth'], show_progress=True):
    # Do cross validation
    train_results = []
    val_results = []
    models = []

    for i in range(num_repeats):
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
        dataset = EmbryoGraphDatasetMetadataOnly(clinical_endpoint='simplified_outcome', 
                                                 dataset_path='data/adjacency_and_bbox_dataset.csv')
        targets = [0 if x.y in pos_classes else 1 for x in dataset]
        for train_idxs, val_idxs in kfold.split(dataset, targets):
            train_sampler = SubsetRandomSampler(train_idxs)
            val_sampler = SubsetRandomSampler(val_idxs)
            # Load data
            loader_train = DataLoader(dataset, batch_size=16, sampler=train_sampler)
            loader_val = DataLoader(dataset, batch_size=16, sampler=val_sampler)
            
            # Set up model
            # model = ImageGNN().to(device)#
            model = GNN(3, hidden_size, num_convs).to(device)
            model.apply(init_weights)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            for epoch in tqdm(range(epochs)) if show_progress else range(epochs):
                model = train_one_epoch(model, loader_train, optimizer, device, pos_classes=pos_classes)
                # print(eval_model(model, loader_train, device, pos_classes=pos_classes))
                # print(eval_model(model, loader_val, device, pos_classes=pos_classes))
            train_result = eval_model(model, loader_train, device, pos_classes=pos_classes)
            val_result = eval_model(model, loader_val, device, pos_classes=pos_classes)
            train_results.append(train_result)
            val_results.append(val_result) 

            models.append(model)
    best_model_idx = val_results.index(min(val_results, key=lambda x: x[model_selection_metric]))
    best_model = models[best_model_idx]
    return best_model, utils.mean_and_std_over_dict(train_results), utils.mean_and_std_over_dict(val_results)

def optimize_hyperparams(trials, epochs, device, repeats=5, pos_classes=['live birth']):
    # Define objective
    def objective(trial):
        # Init search space
        lr_exp = trial.suggest_float('lr_exp', -5, 0)
        wd_exp = trial.suggest_float('wd_exp', -5, 0)
        hidden_exp = trial.suggest_int('hidden_exp', 0, 11)
        num_convs = trial.suggest_int('num_convs', 0, 2)
        # Do search
        aucs = []
        for _ in range(repeats):
            _, train_results, val_results = train_cv(epochs, device, lr=10**lr_exp, weight_decay=10**wd_exp, hidden_size=2**hidden_exp, num_convs=num_convs,
                                                     pos_classes=pos_classes, show_progress=False)
            aucs.append(float(val_results['auc'].split('(')[0])) # TODO: Kinda gross - refactor later
        val_auc = np.mean(aucs)
        return 1-val_auc
    # Do optimisation
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3"
    )
    study.optimize(objective, n_trials=trials, n_jobs=2)
    return study.best_params


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    # best_params = optimize_hyperparams(trials=100, epochs=5, repeats=3, device=device, pos_classes=['live birth'])
    # print(best_params)

    best_model, train_results, val_results = train_cv(
        epochs=5, 
        device=device,
        lr=10**-2.0505939035645597,#8.31e-4,#1.58e-5,#
        weight_decay=10**-1.8384008595658732,#1.63e-4,#3.80e-5,#10**-5.64,
        hidden_size=32,#2**9,
        num_convs=2,
        keep_one_fold_for_testing=True,
        model_selection_metric='auc',
        show_progress=True,
        num_folds=5,
        num_repeats=10
    )
    print('Training', train_results)
    print('Validation', val_results)
    torch.save(best_model.state_dict(), 'model.ckpt')