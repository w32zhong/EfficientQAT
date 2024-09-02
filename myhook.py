import fire
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
    with open('collect.pkl', 'wb') as fh:
        pickle.dump(collect, fh)


def analyze(filename='collect.pkl',
    channels=range(1170, 1180), timesteps=slice(-1), io=1, do_act=False, act_base=0.2):

    with open('collect.pkl', 'rb') as fh:
        collect = pickle.load(fh)

    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    act_fn = lambda x: max(x, act_base) - act_base if do_act else x
    for key, values in collect.items():
        values = values[timesteps]
        for chann in channels:
            ys = [act_fn(time_values[io][chann]) for time_values in values]
            xs = np.arange(len(ys))
            ax.bar(xs, ys, zs=chann, zdir='y', alpha=0.8)
        ax.set_title(f'{key} ({"input" if io == 0 else "output"})')
    ax.set_xlabel('time')
    ax.set_ylabel('channel')
    ax.set_zlabel('activation')
    ax.set_yticks(list(channels))
    plt.show()

if __name__ == '__main__':
    fire.Fire(analyze)
