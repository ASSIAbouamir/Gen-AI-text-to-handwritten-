from __future__ import print_function
from collections import deque
from datetime import datetime
import logging
import os
import pprint as pp
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# No import from tf_utils needed - PyTorch uses .shape directly
# from tf_utils import shape  # Removed: not needed in PyTorch

class PyTorchBase(nn.Module):
    """
    PyTorch port of TFBaseModel: Boilerplate for training PyTorch models.
    
    Subclassing models must implement self.calculate_loss(batch) -> loss tensor.
    Handles training loop, parameter updates, checkpointing, and inference.
    
    Args similar to TF, adapted for PyTorch (e.g., no session/graph).
    """

    def __init__(
        self,
        reader=None,
        batch_sizes=[128],
        num_training_steps=20000,
        learning_rates=[0.01],
        beta1_decays=[0.99],
        optimizer='adam',
        grad_clip=5,
        regularization_constant=0.0,
        keep_prob=1.0,
        patiences=[3000],
        warm_start_init_step=0,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=100,
        log_interval=20,
        logging_level=logging.INFO,
        loss_averaging_window=100,
        validation_batch_size=64,
        log_dir='logs',
        checkpoint_dir='checkpoints',
        prediction_dir='predictions',
        device='cpu',
        num_workers=0,  # For DataLoader
    ):
        super(PyTorchBase, self).__init__()
        assert len(batch_sizes) == len(learning_rates) == len(patiences)
        self.batch_sizes = batch_sizes
        self.learning_rates = learning_rates
        self.beta1_decays = beta1_decays
        self.patiences = patiences
        self.num_restarts = len(batch_sizes) - 1
        self.restart_idx = 0
        self.update_train_params()

        self.reader = reader
        self.num_training_steps = num_training_steps
        self.optimizer_name = optimizer
        self.grad_clip = grad_clip
        self.regularization_constant = regularization_constant
        self.keep_prob_scalar = keep_prob
        self.enable_parameter_averaging = enable_parameter_averaging  # Simplified: use EMA if True
        self.min_steps_to_checkpoint = min_steps_to_checkpoint
        self.log_interval = log_interval
        self.loss_averaging_window = loss_averaging_window
        self.validation_batch_size = validation_batch_size

        self.log_dir = log_dir
        self.logging_level = logging_level
        self.prediction_dir = prediction_dir
        self.checkpoint_dir = checkpoint_dir
        if self.enable_parameter_averaging:
            self.checkpoint_dir_averaged = checkpoint_dir + '_avg'

        self.device = torch.device(device)
        self.num_workers = num_workers

        self.init_logging(self.log_dir)
        logging.info('\nnew run with parameters:\n{}'.format(pp.pformat(self.__dict__)))

        logging.info('built model')

    def update_train_params(self):
        self.batch_size = self.batch_sizes[self.restart_idx]
        self.learning_rate = self.learning_rates[self.restart_idx]
        self.beta1_decay = self.beta1_decays[self.restart_idx]
        self.early_stopping_steps = self.patiences[self.restart_idx]

    def calculate_loss(self, batch):
        raise NotImplementedError('subclass must implement this')

    def fit(self):
        # Set optimizer here
        self.optimizer = self.get_optimizer()
        if self.enable_parameter_averaging:
            # Use torch's EMA if available, or simple moving avg
            from torch.optim.swa_utils import AveragedModel
            self.ema_model = AveragedModel(self)  # Simplified EMA

        if self.warm_start_init_step:
            self.restore(self.warm_start_init_step)
            step = self.warm_start_init_step
        else:
            step = 0

        train_loader = self.reader.train_dataloader(self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = self.reader.val_dataloader(self.validation_batch_size, shuffle=False, num_workers=self.num_workers)

        train_loss_history = deque(maxlen=self.loss_averaging_window)
        val_loss_history = deque(maxlen=self.loss_averaging_window)
        train_time_history = deque(maxlen=self.loss_averaging_window)
        val_time_history = deque(maxlen=self.loss_averaging_window)
        if not hasattr(self, 'metrics'):
            self.metrics = {}

        metric_histories = {
            metric_name: deque(maxlen=self.loss_averaging_window) for metric_name in self.metrics
        }
        best_validation_loss, best_validation_tstep = float('inf'), 0

        self.train()  # PyTorch mode

        while step < self.num_training_steps:
            # Validation
            val_start = time.time()
            val_loss = 0
            num_val_batches = 0
            val_metrics = {k: 0 for k in self.metrics}
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in val_batch.items()}
                    loss = self.calculate_loss(val_batch)
                    val_loss += loss.item()
                    num_val_batches += 1
                    
                    # Metrics (assume self.metrics is dict of callables)
                    for name, metric_fn in self.metrics.items():
                        if callable(metric_fn):
                            val_metrics[name] += metric_fn(val_batch, loss).item()

            val_loss /= num_val_batches if num_val_batches > 0 else 1
            for k in val_metrics:
                val_metrics[k] /= num_val_batches if num_val_batches > 0 else 1
            val_loss_history.append(val_loss)
            val_time_history.append(time.time() - val_start)
            for key in val_metrics:
                metric_histories[key].append(val_metrics[key])

            # Train step (one batch per step for simplicity; adapt for full epoch if needed)
            train_start = time.time()
            try:
                train_batch = next(iter(train_loader))  # One batch
            except StopIteration:
                train_loader = self.reader.train_dataloader(self.batch_size, shuffle=True, num_workers=self.num_workers)
                train_batch = next(iter(train_loader))
                
            train_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in train_batch.items()}
            train_loss = self.calculate_loss(train_batch)
            
            self.optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.optimizer.step()
            
            if self.enable_parameter_averaging:
                self.ema_model.update_parameters(self.parameters())
            
            train_loss_history.append(train_loss.item())
            train_time_history.append(time.time() - train_start)

            if step % self.log_interval == 0:
                avg_train_loss = sum(train_loss_history) / len(train_loss_history)
                avg_val_loss = sum(val_loss_history) / len(val_loss_history)
                avg_train_time = sum(train_time_history) / len(train_time_history)
                avg_val_time = sum(val_time_history) / len(val_time_history)
                metric_log = (
                    "[[step {:>8}]]     "
                    "[[train {:>4}s]]     loss: {:<12}     "
                    "[[val {:>4}s]]     loss: {:<12}     "
                ).format(
                    step,
                    round(avg_train_time, 4),
                    round(avg_train_loss, 8),
                    round(avg_val_time, 4),
                    round(avg_val_loss, 8),
                )
                early_stopping_metric = avg_val_loss
                for metric_name, metric_history in metric_histories.items():
                    metric_val = sum(metric_history) / len(metric_history)
                    metric_log += '{}: {:<4}     '.format(metric_name, round(metric_val, 4))
                    if hasattr(self, 'early_stopping_metric') and metric_name == self.early_stopping_metric:
                        early_stopping_metric = metric_val

                logging.info(metric_log)

                if early_stopping_metric < best_validation_loss:
                    best_validation_loss = early_stopping_metric
                    best_validation_tstep = step
                    if step > self.min_steps_to_checkpoint:
                        self.save(step)
                        if self.enable_parameter_averaging:
                            self.save(step, averaged=True)

                if step - best_validation_tstep > self.early_stopping_steps:
                    if self.num_restarts is None or self.restart_idx >= self.num_restarts:
                        logging.info('best validation loss of {} at training step {}'.format(
                            best_validation_loss, best_validation_tstep))
                        logging.info('early stopping - ending training.')
                        return

                    if self.restart_idx < self.num_restarts:
                        self.restore(best_validation_tstep)
                        step = best_validation_tstep
                        self.restart_idx += 1
                        self.update_train_params()
                        # Reset loader
                        train_loader = self.reader.train_dataloader(self.batch_size, shuffle=True, num_workers=self.num_workers)

            step += 1

        if step <= self.min_steps_to_checkpoint:
            best_validation_tstep = step
            self.save(step)
            if self.enable_parameter_averaging:
                self.save(step, averaged=True)

        logging.info('num_training_steps reached - ending training')

    def predict(self, chunk_size=256):
        if not os.path.isdir(self.prediction_dir):
            os.makedirs(self.prediction_dir)

        if hasattr(self, 'prediction_tensors'):  # Assume dict of model outputs
            prediction_dict = {tensor_name: [] for tensor_name in self.prediction_tensors}
            test_loader = self.reader.test_dataloader(chunk_size)

            self.eval()
            with torch.no_grad():
                for i, test_batch in enumerate(test_loader):
                    if i % 10 == 0:
                        print(i * chunk_size)
                    test_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in test_batch.items()}
                    
                    # Run prediction (adapt to model.forward)
                    outputs = self(test_batch)  # Or self.predict_step(test_batch)
                    for name, out in zip(self.prediction_tensors.keys(), outputs):
                        prediction_dict[name].append(out.cpu().numpy())

            for tensor_name, tensor_list in prediction_dict.items():
                np_tensor = np.concatenate(tensor_list, axis=0)
                save_file = os.path.join(self.prediction_dir, '{}.npy'.format(tensor_name))
                logging.info('saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                np.save(save_file, np_tensor)

        self.train()

        if hasattr(self, 'parameter_tensors'):  # Save model params
            state = {name: param.cpu().numpy() for name, param in self.named_parameters()}
            np.savez(os.path.join(self.prediction_dir, 'parameters.npz'), **state)
            logging.info('saved parameters to predictions/parameters.npz')

    def save(self, step, averaged=False):
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f'model-step-{step}.pt')
        logging.info('saving model to {}'.format(model_path))
        state_dict = self.ema_model.state_dict() if averaged and hasattr(self, 'ema_model') else self.state_dict()
        torch.save({'model': state_dict, 'step': step}, model_path)

    def restore(self, step=None, averaged=False):
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if not step:
            # Find latest
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                latest = max(checkpoints, key=lambda x: int(x.split('-')[-1].split('.')[0]))
                model_path = os.path.join(checkpoint_dir, latest)
            else:
                raise ValueError("No checkpoint found")
        else:
            model_path = os.path.join(checkpoint_dir, f'model-step-{step}.pt')
        
        logging.info('restoring model from {}'.format(model_path))
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model']
        if averaged and hasattr(self, 'ema_model'):
            self.ema_model.load_state_dict(state_dict)
        else:
            self.load_state_dict(state_dict)

    def init_logging(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = os.path.join(log_dir, f'log_{date_str}.txt')

        # Clear handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=self.logging_level,
            format='[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )

    def update_parameters(self, loss):
        # L2 reg
        if self.regularization_constant != 0:
            l2_norm = sum(p.norm(2) for p in self.parameters())
            loss = loss + self.regularization_constant * l2_norm

        # Optimizer (set in fit)
        if self.enable_parameter_averaging:
            from torch.optim.swa_utils import AveragedModel
            self.ema_model = AveragedModel(self, decay=0.99)  # EMA setup

        # Step is in fit loop
        logging.info('trainable parameter count: {}'.format(
            sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def get_optimizer(self):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.beta1_decay, 0.999))
        elif self.optimizer_name == 'gd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'rms':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, alpha=self.beta1_decay, momentum=0.9)
        else:
            raise ValueError(f'optimizer must be adam, gd, or rms: got {self.optimizer_name}')

    # In fit, set self.optimizer = self.get_optimizer()
    # self.scheduler = ReduceLROnPlateau(self.optimizer, patience=self.early_stopping_steps // self.log_interval)