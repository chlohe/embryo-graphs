import numpy as np 

def mean_and_std_over_dict(dicts):
    assert len(dicts) > 1
    result = {k: f'{np.mean([x[k] for x in dicts]):.3f} ({np.std([x[k] for x in dicts]):.3f})' for k in dicts[0].keys()}
    return result

def mean_over_dict(dicts):
    assert len(dicts) > 1
    result = {k: np.mean([x[k] for x in dicts]) for k in dicts[0].keys()}
    return result

def std_over_dict(dicts):
    assert len(dicts) > 1
    result = {k: np.std([x[k] for x in dicts]) for k in dicts[0].keys()}
    return result

def se_over_dict(dicts, N):
    # standard error
    assert len(dicts) > 1
    result = {k: np.std([x[k] for x in dicts]) / np.sqrt(N) for k in dicts[0].keys()}
    return result

def CI_over_dict(dicts, alpha=0.95):
    # standard error
    N = len(dicts)
    assert N > 1
    lower = int(np.floor((1 - alpha) / 2 * N)) + 1
    upper = N - int(np.floor((1 - alpha) / 2 * N)) - 1   
    result = {k: (
        sorted([x[k] for x in dicts])[lower],
        sorted([x[k] for x in dicts])[upper]
    ) for k in dicts[0].keys()}
    return result

def subtract_dicts(a, b):
    result = {k : a[k] - b[k] for k in a.keys()}
    return result

def add_dicts(a, b):
    result = {k : a[k] + b[k] for k in a.keys()}
    return result

def multiply_dicts(a, b):
    result = {k : a[k] * b[k] for k in a.keys()}
    return result

def divide_dicts(a, b):
    result = {k : a[k] / b[k] for k in a.keys()}
    return result

def apply_to_dict(d, fn):
    result = {k : fn(d[k]) for k in d.keys()}
    return result