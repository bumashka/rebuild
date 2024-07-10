from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def save_time_to_file(time, config):
    with open("execution_time.txt", "a") as file:
        file.write(
        f"""^^^^^^^^^^
{datetime.now()}
Execution time for {config["dataset_name"]} with {config["max_epochs"]} number of epoc,
engine {config["engine"]}, accelerator {config["accelerator"]}.
Execution time is {time} seconds.
^^^^^^^^^^""")


def plot_2d(latent, labels, alpha=0.7, title="", fontsize=25, s=2.0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.scatter(latent[:, 0], latent[:, 1], alpha=alpha, c=labels, s=s, label=labels)
    if len(title):
        ax.set_title(title, fontsize=fontsize)
    return ax


def visualize(config):
    dataset = [config['dataset_name']]
    version = config['version']
    add_original = False
    models = {'RTD AutoEncoder H1': 'RTD'}
    fig, axes = plt.subplots(1, len(models) + int(add_original),
                             figsize=((len(models) + int(add_original)) * 6, 6), squeeze=False)
    for i, dataset in enumerate(dataset):
        print(f"dataset: {dataset}, version: {version}")
        labels = None  # refactor
        try:
            labels = np.load(f"data/{dataset}/prepared/train_labels.npy")
        except FileNotFoundError:
            pass
        try:
            labels = np.load(f"data/{dataset}/prepared/labels.npy")
        except FileNotFoundError:
            pass
        if add_original:
            original_data = np.load(f"data/{dataset}/prepared/train_data.npy")
            axes[i][0].scatter(original_data[:, 0], original_data[:, 1], original_data[:,2], c=labels, s=1.0, alpha=0.7,
                               cmap=plt.cm.get_cmap('nipy_spectral', 11))
            if i == 0:
                axes[0][0].set_title('Original data', fontsize=40)
            axes[i][0].tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False,
                right=False,
                left=False,
                labelleft=False
            )
            d = dataset
            axes[i][0].set_ylabel(d, fontsize=40)
        for j, name in enumerate(models):
            if add_original:
                j += 1

            latent = None
            print(name)

            potential_filenames = [
                f'data/{dataset}/{name}_output_{version}.npy',
                f'data/{dataset}/{name}_output_d2.npy',
                f'data/{dataset}/{name}_output_.npy',
                f'data/{dataset}/{name}_output.npy'
            ]
            for n in potential_filenames:
                try:
                    print(n)
                    latent = np.load(n)
                    break
                except FileNotFoundError:
                    print("FileNotFoundError")
            if latent is None:
                raise FileNotFoundError(f'No file for model: {name}, dataset: {dataset}')
            axes[i][j].scatter(latent[:, 1], latent[:, 0], c=labels, s=1.0, alpha=0.7,
                               cmap=plt.cm.get_cmap('nipy_spectral', 11))
            if i == 0:
                axes[i][j].set_title(f'{models[name]}', fontsize=30)
            if j == 0 and not add_original:
                d = dataset
                axes[i][j].set_ylabel(d, fontsize=30)
            axes[i][j].tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False,
                right=False,
                left=False,
                labelleft=False
            )
    plt.savefig(f'data/{dataset}/{name}_{config["engine"]}_{config["max_epochs"]}_real.png')
