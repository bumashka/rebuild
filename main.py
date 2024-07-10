import time

import data_preparation as dp
import ae_training
import visualization

if __name__ == '__main__':

    dp.dinosaur()
    dp.random_cube()
    dp.mammoth()

    config = {
        "dataset_name": "Mammoth", # Название дата-сета, указываем для точного пути data/{dataset_name}/...
        "version": "d2",              # Версия, можно и не указывать
        "model_name": "default",
        "max_epochs": 50,          # Максимальное кол-во эпох. Влияет на время и точность работы
        "devices": 2,              # Используемые устройства, для GPU - [0]
        "accelerator": "cpu",      # Вид вычислителя, в нашем случае CPU
        "rtd_every_n_batches": 1,
        "rtd_start_epoch": 0,
        "rtd_l": 1.0,              # РТД-потеря
        "n_runs": 1,
        "card": 50,                # Количество точек в диаграмме персистентности
        "n_threads": 50,           # Количество трэдов для параллельного вычисления
        "latent_dim": 2,           # Размерность выходных данных - обычно 2
        "input_dim": 3,            # Размерность входных данных
        "n_hidden_layers": 3,      # Количество скрытых слоев модели
        "hidden_dim": 512,         # Размерность скрытых слоев
        "batch_size": 64,          # Размер пакета обучения
        #     "width":80,
        #     "heigth":80,
        "engine": "giotto",        # Используемый движок; для CPU - giotto, GPU - ripser
        "is_sym": True,
        "lr": 5e-4,
        # 'mode':'minimum',
        # 'lp':1.0
    }
    start_time = time.time()  # время начала выполнения
    T = ae_training.Training(config)
    T.train()
    end_time = time.time()  # время окончания выполнения
    execution_time = end_time - start_time  # вычисляем время выпол
    visualization.save_time_to_file(execution_time, config)
    visualization.visualize(config)



