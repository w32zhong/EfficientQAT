import torch
import pickle
from functools import partial
from collections import defaultdict

def hook_fn(collect, model, path, module, inputs, output):
    x = inputs[0]
    y = output
    for i in range(x.shape[1]):
        x_row = x[0,i,:].cpu().numpy()
        y_row = y[0,i,:].cpu().numpy()
        collect[path].append((x_row, y_row))


def add(model, limited_paths=None):
    collect = defaultdict(list)
    for path, module in model.named_modules():
        if limited_paths is None or path in limited_paths:
            print('[hook]', path)
            hook = module.register_forward_hook(
                partial(hook_fn, collect, model, path)
            )
    return collect

def save(collect):
    with open('collect.plk', 'wb') as fh:
        pickle.dump(collect, fh)
