import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from RTD_AE.src.autoencoder import AutoEncoder
from RTD_AE.src.utils import *
from RTD_AE.src.rtd import RTDLoss, MinMaxRTDLoss

from tqdm.notebook import tqdm


def get_model(input_dim, latent_dim=2, n_hidden_layers=2, m_type='encoder'):
    n = int(np.log2(input_dim))-1
    layers = []
    if m_type == 'encoder':
        in_dim = input_dim
        if input_dim  // 2 >= latent_dim:
            out_dim = input_dim // 2
        else:
            out_dim = input_dim
        for i in range(min(n, n_hidden_layers)):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            in_dim = out_dim
            if in_dim  // 2 >= latent_dim:
                out_dim = in_dim // 2
            else:
                out_dim = in_dim
        layers.extend([nn.Linear(in_dim, latent_dim)])
    elif m_type == 'decoder':
        in_dim = latent_dim
        out_dim = latent_dim * 2
        for i in range(min(n, n_hidden_layers)):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            in_dim = out_dim
            out_dim *= 2
        layers.extend([nn.Linear(in_dim, input_dim)])
    return nn.Sequential(*layers)


def get_list_of_models(**config):
    # define a list of models
    encoder = get_linear_model(
        m_type='encoder',
        **config
    )
    decoder = get_linear_model(
        m_type='decoder',
        **config
    )
    models = {
         'RTD AutoEncoder H1': AutoEncoder(
             encoder = encoder,
             decoder = decoder,
             RTDLoss = RTDLoss(dim=1, lp=1.0,  **config), # only H1
             MSELoss = nn.MSELoss(),
             **config
         )
    }
    return models, encoder, decoder


def collate_with_matrix(samples):
    indicies, data, labels = zip(*samples)
    data, labels = torch.tensor(np.asarray(data)), torch.tensor(np.asarray(labels))
    if len(data.shape) > 2:
        dist_data = torch.flatten(data, start_dim=1)
    else:
        dist_data = data
    x_dist = torch.cdist(dist_data, dist_data, p=2) / np.sqrt(dist_data.shape[1])
#     x_dist = (x_dist + x_dist.T) / 2.0 # make symmetrical (cdist is prone to computational errors)
    return data, x_dist, labels


def collate_with_matrix_geodesic(samples):
    indicies, data, labels, dist_data = zip(*samples)
    data, labels = torch.tensor(np.asarray(data)), torch.tensor(np.asarray(labels))
    x_dist = torch.tensor(np.asarray(dist_data)[:, indicies])
    return data, x_dist, labels


def train_autoencoder(model, train_loader, val_loader=None, model_name='default',
                      dataset_name='', devices=[0], accelerator='gpu', max_epochs=100, run=0, version=""):
    version = f"{dataset_name}_{model_name}_{version}_{run}"
    print(version)
    logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), name='lightning_logs', version=version)
    trainer = pl.Trainer(
        logger=logger,
        devices=devices,
        accelerator=accelerator,
        max_epochs=max_epochs,
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )
    print(model)
    print(train_loader)
    trainer.fit(model, train_loader, val_loader)
    return model


def dump_figures(figures, dataset_name, version):
    for model_name in figures:
        figures[model_name].savefig(f'results/{dataset_name}/{model_name}_{version}.png')


def train_models(train_loader, val_loader, dataset_name="", max_epochs=1, devices=[0], accelerator='gpu',
                 version='', **kwargs):
    models, encoder, decoder = get_list_of_models(**kwargs)
    for model_name in tqdm(models, desc=f"Training models"):
        if 'AutoEncoder' in model_name: # train an autoencoder
          models[model_name] = train_autoencoder(
                models[model_name],
                train_loader,
                val_loader,
                model_name,
                dataset_name,
                devices,
                accelerator,
                max_epochs,
                0,
                version
            )
        else: # umap / pca / t-sne (sklearn interface)
          train_latent = models[model_name].fit_transform(train_loader.dataset.data)
    return encoder, decoder, models


class Training:
    def __init__(self, config):
        self.config = config
        self.dataset_name = self.config['dataset_name']
        self.train_data = np.load(f'data/{self.dataset_name}/prepared/train_data.npy').astype(np.float32)
        self.test_data = None

        try:
            self.test_data = np.load(f'data/{self.dataset_name}/prepared/test_data.npy').astype(np.float32)
        except FileNotFoundError:
            ids = np.random.choice(np.arange(len(self.train_data)), size=int(0.2 * len(self.train_data)), replace=False)
            self.test_data = self.train_data[ids]

        try:
            self.train_labels = np.load(f'data/{self.dataset_name}/prepared/train_labels.npy')
        except FileNotFoundError:
            self.train_labels = None

        try:
            self.test_labels = np.load(f'data/{self.dataset_name}/prepared/test_labels.npy')
        except FileNotFoundError:
            if self.train_labels is None:
                self.test_labels = None
            else:
                self.test_labels = self.train_labels[ids]

    def train(self):

        scaler = FurthestScaler()
        flatten = True
        geodesic = False

        train = FromNumpyDataset(
            self.train_data,
            self.train_labels,
            geodesic=geodesic,
            scaler=scaler,
            flatten=flatten,
            n_neighbors=2
        )
        print("Train done")
        test = FromNumpyDataset(
            self.test_data,
            self.test_labels,
            geodesic=geodesic,
            scaler=train.scaler,
            flatten=flatten,
            n_neighbors=2
        )

        train_loader = DataLoader(
            train,
            batch_size=self.config["batch_size"],
            num_workers=2,
            collate_fn=collate_with_matrix_geodesic if geodesic else collate_with_matrix,
            shuffle=True
        )

        val_loader = DataLoader(
            test,
            batch_size=self.config["batch_size"],
            num_workers=2,
            collate_fn=collate_with_matrix_geodesic if geodesic else collate_with_matrix,
        )

        encoder, decoder, trained_models = train_models(train_loader, val_loader, **self.config)
        print(trained_models)
        version = self.config['version']
        train_loader = DataLoader(
            train,
            batch_size=self.config["batch_size"],
            num_workers=2,
            collate_fn=collate_with_matrix_geodesic if geodesic else collate_with_matrix,
            shuffle=False
        )

        for model_name in trained_models:
            latent, labels = get_latent_representations(trained_models[model_name], train_loader)
            np.save(f'data/{self.dataset_name}/{model_name}_output_{version}.npy', latent)
            np.save(f'data/{self.dataset_name}/{model_name}_labels_{version}.npy', labels)

