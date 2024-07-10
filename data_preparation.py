import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Папка '{folder_name}' успешно создана.")
    else:
        print(f"Папка '{folder_name}' уже существует.")

"""
Если используем свои данные, то в config указываем dataset_name: MyData

data: numpy массивы
labels: numpy массивы
"""

def my_data(data, labels=None):
    create_folder("data/MyData/prepared")
    np.save('data/MyData/prepared/train_data.npy', data)
    if labels is not None:
        np.save('data/MyData/prepared/train_labels.npy', labels)


def dinosaur():
    data = pd.read_csv("data/Dinosaur/Dinosaur_Notexture_10k.csv")
    X = np.array(data["//X"].to_numpy())
    Y = np.array(data["Y"].to_numpy())
    Z = np.array(data["Z"].to_numpy())
    data = np.vstack((X, Y, Z))
    data = np.transpose(data)
    np.save('data/Dinosaur/prepared/train_data.npy', data)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    plt.title('Dinosaur 10k', fontsize=25)
    ax1.scatter(X, Z, Y, s=1)
    plt.savefig('data/Dinosaur/OriginalData.png')


def random_cube():
    data = np.random.rand(500, 2)
    x, y = zip(*data)
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, s=10.0)
    plt.title('Random points', fontsize=25)
    create_folder("data/RandomCube/prepared")
    np.save('data/RandomCube/prepared/train_data.npy', data)


def mammoth():
    data = json.load(open('data/Mammoth/understanding-umap/raw_data/mammoth_3d.json'))
    labels = json.load(open('data/Mammoth/understanding-umap/public/mammoth_10k_encoded.json'))
    data = np.array(data)
    label_data = []
    for i, offset in enumerate(labels['labelOffsets']):
        label_data.extend([i] * offset)
    label_data = np.array(label_data)
    create_folder("data/Mammoth/prepared")
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    plt.title('Mammoth', fontsize=25)
    ax1.scatter(data[:, 0], data[:, 2], data[:, 1], s=1)
    plt.savefig('data/Mammoth/OriginalData.png')
    np.save('data/Mammoth/prepared/train_data.npy', data)
    np.save('data/Mammoth/prepared/train_labels.npy', label_data)


