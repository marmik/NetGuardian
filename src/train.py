import os
import optuna
from ultralytics import YOLO
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.metrics import fitness

def train_yolov8n(data_yaml, epochs=100, batch_size=16, img_size=640, device=''):
    """
    Train the YOLOv8n model using the Ultralytics library.
    Args:
        data_yaml (str): Path to the data.yaml file.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        img_size (int): Image size for training.
        device (str): Device to use for training (e.g., 'cpu', 'cuda').
    """
    device = select_device(device)
    model = YOLO('yolov8n.yaml').to(device)
    model.train(data=data_yaml, epochs=epochs, batch_size=batch_size, imgsz=img_size)

def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.
    Args:
        trial (optuna.trial.Trial): Optuna trial object.
    Returns:
        float: Fitness score of the model.
    """
    data_yaml = 'path/to/data.yaml'
    epochs = trial.suggest_int('epochs', 50, 200)
    batch_size = trial.suggest_int('batch_size', 8, 32)
    img_size = trial.suggest_int('img_size', 320, 640)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = YOLO('yolov8n.yaml').to(device)
    model.train(data=data_yaml, epochs=epochs, batch_size=batch_size, imgsz=img_size)

    dataloader = build_dataloader(data_yaml, batch_size, img_size, augment=False, cache=False, rect=True, rank=-1, workers=8)
    fitness_score = fitness(model, dataloader)

    return fitness_score

def hyperparameter_tuning(n_trials=50):
    """
    Perform hyperparameter tuning using Optuna.
    Args:
        n_trials (int): Number of Optuna trials.
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    LOGGER.info(f'Best trial: {study.best_trial.params}')
    return study.best_trial.params

if __name__ == '__main__':
    data_yaml = 'path/to/data.yaml'
    best_params = hyperparameter_tuning()
    train_yolov8n(data_yaml, epochs=best_params['epochs'], batch_size=best_params['batch_size'], img_size=best_params['img_size'])
