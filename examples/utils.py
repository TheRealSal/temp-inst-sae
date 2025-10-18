import os
import io
import sys
import random
import numpy as np
import torch
import psutil
import GPUtil
import logging
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset, IterableDataset


def set_seed(seed):
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNGs
    torch.manual_seed(seed)              # CPU
    torch.cuda.manual_seed(seed)         # Current GPU
    torch.cuda.manual_seed_all(seed)     # All GPUs (if using DataParallel or multi-GPU)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Turn off autotune to keep it deterministic

def setup_logging(args, config_name_keys):
    # Prepare model saving directory
    log_model_save_prefix = os.path.join(args.results_dir, '{:s}/{:s}/{:s}'.format(args.model_name.replace('/', '-'), 
                                                                                    args.context.replace('/', '-'), 
                                                                                    args.text.replace('/', '-')))
    config_name = "_".join(f"{key}_{getattr(args, key)}" for key in config_name_keys)
    
    log_model_save_dir = os.path.join(log_model_save_prefix, config_name)
    os.makedirs(log_model_save_dir, exist_ok=True)
    assert os.path.exists(log_model_save_dir) 

    model_save_dir = os.path.join(log_model_save_dir, 'ckps')
    os.makedirs(model_save_dir, exist_ok=True)
    assert os.path.exists(model_save_dir) 

    log_path = os.path.join(log_model_save_dir, os.path.basename(log_model_save_dir)+'_log.output')
    loss_path = os.path.join(log_model_save_dir, os.path.basename(log_model_save_dir)+'_loss.json')

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    # Console handler (optional, for real-time monitoring)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    # Clear existing handlers (if rerunning in notebook or script)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Redirect stdout and stderr to logging
    sys.stdout = open(log_path, 'a')
    sys.stderr = sys.stdout

    # First log
    logger.info("Logging to: {:s}".format(log_path))

    return logger, log_path, loss_path, model_save_dir

def logging_mem_usage(logger):
    # CPU usage
    process = psutil.Process()
    cpu_mem_mb = process.memory_info().rss / 1024**2  # in MB
    cpu_percent = process.cpu_percent(interval=0.1)
    logger.info(f"CPU Mem Used: {cpu_mem_mb} MB ({cpu_percent} %)")

    # GPU usage
    gpu = GPUtil.getGPUs()[0]
    logger.info(f"GPU Mem Used: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB")

# Tqdm to Logger class
class TqdmToLogger(io.StringIO):
    def __init__(self, logger, level=logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, buf):
        self.logger.log(self.level, buf.strip())

    def flush(self):
        pass  # tqdm doesn't need flush here

def gen_window_slicing_batch(batch, window_size, stride=1):
    # Slide over the sequence length dimension
    seq_len = batch.shape[0]
    window_slicing_batch = []
    for i in range(0, seq_len - window_size + 1, stride):
        window_slicing_batch.append(batch[i:i + window_size, :].T)
   
    return torch.stack(window_slicing_batch, dim=0)

def draw_loss(loss_dict, loss_path, total_tokens):
    f, axs = plt.subplots(nrows=2, ncols=3, figsize=(3*3, 2*2))

    for idx, (key, vals) in enumerate(loss_dict.items()):
        n_points = 5
        token_per_points = total_tokens / (n_points-1)

        xticks = [i * token_per_points for i in range(n_points)]
        xtick_labels = [f"{x :.1f}" for x in xticks]

        ax = axs[idx//3, idx%3]
        ax.plot(vals)
        ax.set_title(key)
        ax.set_xlabel('Training tokens (M)')
        # ax.set_xticks(range(len(vals)))
        # ax.set_xticklabels(xtick_labels, rotation=45)

    fig_path = loss_path.replace('.json', '.png')
    f.tight_layout()
    f.savefig(fig_path)

def hugging_face_login(token):
    from huggingface_hub import login
    login(token)
