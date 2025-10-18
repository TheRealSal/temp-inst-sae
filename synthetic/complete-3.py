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
    # w_hist: float = 0.8
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
# %%
# Move data to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
class LinearTempInstICA(nn.Module):
    def __init__(self, x_dim, z_dim, M_mask="tril"):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.A = nn.Parameter(torch.randn(x_dim, z_dim))
        self.B = nn.Parameter(torch.randn(z_dim, z_dim))
        self.M = nn.Parameter(torch.randn(z_dim, z_dim))
        self.M_mask = M_mask
        # self.A.data = torch.tensor(special_ortho_group.rvs(z_dim), dtype=torch.float32).to(self.A.device)
        # Initialize A with ground truth
        # self.A.data = torch.tensor(data["A"], dtype=torch.float32).to(self.A.device)
        # Initialize A with orthogonal matrix
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
        elif self.M_mask == "off_diag_perm":
            no_self_loop_M = self.M * (1 - torch.eye(self.z_dim, device=self.M.device))
            _, permutation = torch.sort(no_self_loop_M.abs().sum(dim=1))
            perm_tril = torch.tril(no_self_loop_M[permutation][:, permutation])
            inverse_permutation= torch.zeros_like(permutation)
            inverse_permutation[permutation] = torch.arange(self.z_dim, device=self.M.device)
            inv_perm_tril = perm_tril[inverse_permutation][:, inverse_permutation]
            return inv_perm_tril
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

x_dim = 3
z_dim = 3
length = 1
w_inst = 0.2
lr = 8e-3 
wd = 6e-4 
total_steps = 50_000
batch_size = 1024
noise_type = "laplace"
noise_scale = 1.0
log_interval = 100
M_sparsity_loss_coeff = 1e-3 
B_sparsity_loss_coeff = 1e-8 
kl_div_coeff = 1.0
seed = 44 
M_mask = "tril" 

set_seed(seed) # Fix a random seed for reproducibility

B =  np.array([[0.4, 0.6, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
M = np.array([[0, 0, 0], [w_inst, 0, 0], [0, w_inst, 0]], dtype=np.float32)
A = special_ortho_group.rvs(3)
wandb_run = wandb.init(
    project="linear-temporal-instantaneous",
    # name=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}-{seed}",
    config={
        "x_dim": x_dim,
        "z_dim": z_dim,
        "lr": lr,
        "wd": wd,
        "B": json.dumps(B.tolist()),
        "A": json.dumps(A.tolist()),
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
)

model = LinearTempInstICA(x_dim=x_dim, z_dim=z_dim, M_mask=M_mask).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

with tqdm(total=total_steps) as pbar:
    for step in range(total_steps):
        try:
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
            B_sparsity_loss = torch.abs(model.B).sum()
            M_sparsity_loss = torch.abs(model.get_M()).sum()
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
                M_est_permutated = M_est[:, mcc_dict["col_ind"]][mcc_dict["col_ind"],:]

                B_err = np.abs(np.abs(B_est_permutated) - B).sum()
                M_err = np.abs(np.abs(M_est_permutated) - M).sum()
                wandb_run.log({
                    "B_err": B_err,
                    "M_err": M_err,
                    "err": B_err + M_err,
                }, step=step)
                if step % (log_interval * 10) == 0:
                    B_fig = create_matrix_figure(
                        B_est, 
                        title="B matrix",
                        vmin=0,
                        vmax=1,
                    )
                    B_fig_permutated = create_matrix_figure(
                        B_est_permutated, 
                        title="B matrix (permutated)",
                        vmin=0,
                        vmax=1,
                    )
                    
                    M_fig = create_matrix_figure(
                        M_est,
                        title="M matrix",
                        vmin=0,
                        vmax=1,
                    )
                    
                    M_fig_permutated = create_matrix_figure(
                        M_est_permutated,
                        title="M matrix (permutated)",
                        vmin=0,
                        vmax=1,
                    )
                    CC_fig = create_matrix_figure(
                        cc, 
                        title="CC matrix",
                        vmin=0,
                        vmax=1,
                    )
                    wandb_run.log({"valid/B_matrix": wandb.Image(B_fig)}, step=step)
                    wandb_run.log({"valid/B_matrix_permutated": wandb.Image(B_fig_permutated)}, step=step)
                    wandb_run.log({"valid/CC_matrix": wandb.Image(CC_fig)}, step=step)
                    wandb_run.log({"valid/M_matrix": wandb.Image(M_fig)}, step=step)
                    wandb_run.log({"valid/M_matrix_permutated": wandb.Image(M_fig_permutated)}, step=step)
                    plt.close(B_fig)
                    plt.close(CC_fig)
                    plt.close(M_fig)
                    plt.close(B_fig_permutated)
                    plt.close(M_fig_permutated)
                pbar.set_postfix({"MCC": mcc, "Loss": loss.item()})
                # if mcc > early_stopping_mcc_threshold:
                #     print(f"Early stopping at step {step} with MCC: {mcc}")
                #     wandb_run.log({"early_stop": True}, step=step)
                #     break
            pbar.update(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            wandb_run.finish()
            break
wandb_run.finish()
# %%