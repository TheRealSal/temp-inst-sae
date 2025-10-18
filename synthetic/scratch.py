# %%
import numpy as np
from scipy.stats import special_ortho_group
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import set_seed, compute_mcc, create_matrix_figure
from tqdm import tqdm
import wandb
import json
import matplotlib.pyplot as plt
from datetime import datetime
# %%
def generate_synthetic_data(
    B: np.ndarray,
    A: np.ndarray,
    num_samples: int = 1024,
    noise_type: str = "laplace",
    noise_scale: float = 1.0,
    length: int = 1,
    w_inst: float = 0.2,
):
    """
    Generate synthetic temporal data with ground truth parameters.
    
    Parameters:
    -----------
    num_samples: int
        Number of samples to generate
    B: ndarray
        Transition matrix for hidden state dynamics
    A: ndarray
        Observation matrix
    noise_type: str
        Type of noise to use ('normal' or 'laplace')
    x_dim: int
        Dimension of observed data
    z_dim: int
        Dimension of hidden state
    noise_scale: float
        Scale of noise
    length: int
        Length of time series
    w_inst: float
        Weight for instantaneous dependencies
    w_hist: float
        Weight for historical dependencies
    
    Returns:
    --------
    dict with keys:
        'Z': Hidden states (shape: num_samples, length+1, z_dim)
        'X': Observed data (shape: num_samples, length+1, x_dim)
    """
    z_dim = B.shape[1]
    # Initialize first hidden state
    # z_0 = np.random.normal(0, noise_scale, (num_samples, z_dim))
    z_0 = np.random.uniform(0, 1, (num_samples, z_dim))
    Z = [z_0]

    z_l = z_0
    for t in range(length):
        z_hist = np.dot(z_l, B.T)

        # Generate noise with proper shape directly
        if noise_type == "normal":
            noise = np.random.normal(0, noise_scale, (num_samples, z_dim))
        elif noise_type == "laplace":
            noise = np.random.laplace(0, noise_scale, (num_samples, z_dim))
        else:
            raise ValueError("Unsupported noise type. Choose 'normal' or 'laplace'.")
        
        # The full model uses both historical and instantaneous dependencies
        z_t = np.zeros((num_samples, z_dim))
        
        # # First dimension with historical dependency only
        z_t[:, 0] = z_hist[:, 0] + noise[:, 0]
        
        # Remaining dimensions with both historical and instantaneous dependencies
        for i in range(1, z_dim):
            z_t[:, i] = (
                z_hist[:, i]
                + z_t[:, i - 1] * w_inst
                + noise[:, i]
            )
        # z_t = z_hist + noise
        
        Z.append(z_t)
        z_l = z_t

    # Convert list to array - shape (num_samples, length+1, z_dim)
    Z = np.array(Z).transpose((1, 0, 2))
    
    # Generate observations
    X = np.matmul(Z, A.T)

    return {
        "Z": Z,
        "X": X
    }
def sample_sparse_B(z_dim, sparsity=0.5):
    """
    Sample a sparse matrix B of size (z_dim, z_dim) with specified sparsity.
    Sparsity applies to the strictly upper triangular part; the diagonal is always retained.
    """
    B = special_ortho_group.rvs(z_dim)

    # Generate a mask for the upper triangle excluding the diagonal
    upper_mask = np.triu(np.ones((z_dim, z_dim)), k=1)
    
    # Apply sparsity mask to the upper triangle
    sparse_mask = np.random.binomial(1, sparsity, size=(z_dim, z_dim)) * upper_mask

    # Combine with identity to keep the diagonal
    final_mask = np.eye(z_dim) + sparse_mask

    return B * final_mask

def get_chain_M(z_dim, w_inst=0.5):
    """
    Generate a chain matrix M of size (z_dim, z_dim) with specified sparsity.
    The chain matrix has 1s on the diagonal and the first superdiagonal.
    """
    M = np.zeros((z_dim, z_dim))
    for i in range(z_dim):
        if i == 0:
            continue
        M[i, i-1] = w_inst
    return M
class LinearTempInstICA(nn.Module):
    def __init__(self, x_dim, z_dim, M_mask="tril"):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.A = nn.Parameter(torch.randn(x_dim, z_dim))
        self.B = nn.Parameter(torch.randn(z_dim, z_dim))
        self.M = nn.Parameter(torch.randn(z_dim, z_dim))
        self.M_mask = M_mask

    def encode(self, x):
        A_pinv = torch.pinverse(self.A)
        return torch.matmul(x, A_pinv.T)
    def decode(self, z):
        return torch.matmul(z, self.A.T)
    def get_M(self):
        if self.M_mask == "tril":
            # Mask the upper triangular part of M
            return torch.tril(self.M, diagonal=-1)
        elif self.M_mask == "off_diag":
            return self.M * (1 - torch.eye(self.z_dim, device=self.M.device))
        raise ValueError("Invalid M_mask value. Choose 'tril' or 'off_diag'.")
        
    def estimate_prior(self, z):
        z_t0 = z[:, 0, :]
        z_t1 = z[:, 1, :]
        # first calculate I - M
        I_M = torch.eye(self.z_dim, device=self.M.device) - self.get_M()
        z_t1_I_M_T = torch.matmul(z_t1, I_M.T)
        eps = z_t1_I_M_T - torch.matmul(z_t0, self.B.T)
        return eps
    def forward(self, x):
        z_hat = self.encode(x)
        eps_hat = self.estimate_prior(z_hat)

        x_hat = self.decode(z_hat)
        return {
            "x_hat": x_hat,
            "z_hat": z_hat,
            "eps_hat": eps_hat,
        }
# %%
import argparse
parser = argparse.ArgumentParser(description="Train a linear temporal ICA model.")
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--seed", type=int, default=44)
args = parser.parse_args()

x_dim = args.dim
z_dim = x_dim
length = 1
w_inst = 0.5
lr = 1e-3 
wd = 6e-4 
total_steps = 50_000
batch_size = 1024
noise_type = "laplace"
noise_scale = 1.0
log_interval = 100
kl_div_coeff = 1.0
M_sparsity_loss_coeff = 1e-3
B_sparsity_loss_coeff = 1e-5
seed = args.seed 
M_mask = "tril" 

set_seed(seed)
A = special_ortho_group.rvs(z_dim)
B = sample_sparse_B(z_dim, sparsity=0.1)
M = get_chain_M(z_dim, w_inst=0.5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%

with wandb.init(
    project="complexity-linear-temporal-instantaneous",
    name=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}-{seed}-{x_dim}",
    config={
        "x_dim": x_dim,
        "z_dim": z_dim,
        "lr": lr,
        "wd": wd,
        "length": length,
        "w_inst": w_inst,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "noise_type": noise_type,
        "noise_scale": noise_scale,
        "log_interval": log_interval,
        "kl_div_coeff": kl_div_coeff,
        "M_sparsity_loss_coeff": M_sparsity_loss_coeff,
        "B_sparsity_loss_coeff": B_sparsity_loss_coeff,
        "M_mask": M_mask,
        "seed": seed,
    },
) as wandb_run, tqdm(total=total_steps) as pbar:
    
    # add time complexity and memory usage
    start_time = datetime.now()
    model = LinearTempInstICA(x_dim=x_dim, z_dim=z_dim, M_mask=M_mask).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    mcc = 0
    for step in range(total_steps):
        batch = generate_synthetic_data(
            B=B, A=A, num_samples=batch_size, noise_type=noise_type, noise_scale=noise_scale, length=length, w_inst=w_inst
        )
        X_batch, Z_batch =  batch["X"], batch["Z"]
        optimizer.zero_grad()
        X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
        outputs = model(X_batch)
        Z_hat_batch = outputs["z_hat"]
        X_hat_batch = outputs["x_hat"]
        eps_hat_batch = outputs["eps_hat"]
        kl_div = torch.abs(eps_hat_batch).mean()
        recon_loss = F.mse_loss(X_hat_batch, X_batch)
        B_sparsity_loss = torch.abs(model.B).mean()
        M_sparsity_loss = torch.abs(model.get_M()).mean()
        loss = recon_loss + \
            kl_div * kl_div_coeff + \
            M_sparsity_loss * M_sparsity_loss_coeff + \
            B_sparsity_loss * B_sparsity_loss_coeff
        loss.backward()
        optimizer.step()
        
        
        if step % log_interval == 0:
            z_flat = Z_batch.reshape(-1, z_dim).T
            z_hat_flat = Z_hat_batch.detach().cpu().numpy().reshape(-1, z_dim).T
            mcc_dict = compute_mcc(z_flat, z_hat_flat, dict_size=z_dim, return_dict=True)
            mcc = mcc_dict["mcc"]
            cc = mcc_dict["cc"]
            wandb_run.log({
                "loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "kl_div": kl_div.item(),
                "B_sparsity_loss": B_sparsity_loss.item(),
                "M_sparsity_loss": M_sparsity_loss.item(),
                "mcc": mcc,
            }, step=step)
            B_est = model.B.detach().cpu().numpy()
            B_est_permutated = B_est[:, mcc_dict["col_ind"]][mcc_dict["col_ind"],:]
            M_est = model.get_M().detach().cpu().numpy()
            # M_est_permutated = M_est[:, mcc_dict["col_ind"]][mcc_dict["col_ind"],:]

            B_err = np.abs(np.abs(B_est_permutated) - B).sum()
            M_err = np.abs(np.abs(M_est) - M).sum()
            wandb_run.log({
                "B_err": B_err,
                "M_err": M_err,
                "err": B_err + M_err,
            }, step=step)
            # log memory usage
            wandb_run.log({
                "memory": torch.cuda.memory_allocated() / 1024**2,
            }, step=step)
            # log time complexity
            wandb_run.log({
                "time": (datetime.now() - start_time).total_seconds(),
            }, step=step)

        pbar.set_postfix({"loss": loss.item(), "mcc": mcc})
        pbar.update(1)

# %%


