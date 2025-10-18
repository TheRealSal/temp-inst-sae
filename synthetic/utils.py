import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Union, Any
from scipy.optimize import linear_sum_assignment
import warnings
import random
from functools import partial
from scipy.linalg import qr
from scipy.stats import special_ortho_group


def generate_orthogonal_matrix(z_dim):
    return special_ortho_group.rvs(dim=z_dim)
def compute_mcc(A_1, A_2, dict_size = None, return_dict = False):
    """
    Compute the Matthews correlation coefficient between two matrices.
    shape of A_1 and A_2 should be (d, n) where n is the number of samples and d is the dimension.
    """
    assert A_1.shape == A_2.shape, "The two matrices must have the same shape."
        
    if dict_size is not None:
        # Ensure A_1 and A_2 are of the same size and the size match the dict_size
        assert A_1.shape[0] == dict_size, f"A_1 and A_2 must have the {dict_size} rows"

    # Normalize columns of both matrices
    A_1_norm = np.linalg.norm(A_1, axis=1, keepdims=True)
    A_2_norm = np.linalg.norm(A_2, axis=1, keepdims=True)
    
    # Avoid division by zero
    if np.any(A_1_norm == 0) or np.any(A_2_norm == 0):
        warnings.warn("A_1 and A_2 have zero columns, which may affect the MCC calculation.")
        A_1_norm[A_1_norm == 0] = 1
        A_2_norm[A_2_norm == 0] = 1
    
    A_1 = A_1 / A_1_norm
    A_2 = A_2 / A_2_norm

    # Calculate the cost matrix using inner products
    cost_matrix = 1 - np.abs(np.matmul(A_1, A_2.T))

    # Solve the assignment problem using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Calculate the MCC
    mcc = 1 - cost_matrix[row_ind, col_ind].sum() / len(row_ind)
    
    if return_dict:
        return {
            "mcc": mcc, 
            "cc": 1 - cost_matrix,
            "row_ind": row_ind,
            "col_ind": col_ind,
            }
    
    
    return mcc



def get_noise(
    shape: Union[int, Tuple[int, ...]], 
    noise_type: str = "normal", 
    loc: float = 0, 
    scale: float = 0.1,
    threashold: Optional[float] = None
) -> np.ndarray:
    """
    Generate noise of specified type and shape.
    """
    if noise_type == "normal":
        noise = np.random.normal(loc, scale, shape)
    elif noise_type == "laplace":
        noise = np.random.laplace(loc, scale, shape)
    else:
        raise ValueError("Unsupported noise type. Use 'normal' or 'laplace'.")
    if threashold is not None:
        return noise * (np.abs(noise) > threashold)
    return noise


def generate_synthetic_data(
    num_samples: int = 10_000, noise_type: str = "normal"
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic temporal data with ground truth parameters.
    """
    x_dim = 3
    z_dim = 3
    noise_scale = 0.1
    length = 1
    w_inst = 0.2
    w_hist = 1 - w_inst

    # Define transition matrix (ground truth for B)
    transition = np.array([[0.4, 0.6, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    # Initialize first hidden state
    z_0 = np.random.normal(0, noise_scale, (num_samples, z_dim))
    Z = [z_0]

    z_l = z_0
    for t in range(length):
        z_t = np.zeros((num_samples, z_dim))
        z_hist = np.dot(z_l, transition.T)

        # Generate noise with proper shape directly
        noise_generator = partial(
            get_noise, shape=num_samples, noise_type=noise_type, 
            loc=0, scale=noise_scale
        )
        
        # First dimension with historical dependency only
        z_t[:, 0] = z_hist[:, 0] + noise_generator()
        
        # Remaining dimensions with both historical and instantaneous dependencies
        for i in range(1, z_dim):
            z_t[:, i] = (
                z_hist[:, i] * w_hist
                + z_t[:, i - 1] * w_inst
                + noise_generator()
            )

        Z.append(z_t)
        z_l = z_t

    # Convert list to array - shape (num_samples, length+1, z_dim)
    Z = np.array(Z).transpose((1, 0, 2))
    
    # Generate an invertible matrix with good condition number for observation
    # A, _ = generate_well_conditioned_matrix(z_dim)
    # generate orthogonal matrix
    
    A = generate_orthogonal_matrix(z_dim)
    # Create observation matrix X
    X = np.matmul(Z, A.T)

    return {
        "x_dim": x_dim,
        "z_dim": z_dim,
        "length": length,
        "num_samples": num_samples,
        "noise_scale": noise_scale,
        "w_inst": w_inst,
        "w_hist": w_hist,
        "Z": Z,
        "X": X,
        "B": np.array([[0.4, 0.6, 0], [0, 0.8, 0], [0, 0, 0.8]]),
        "M": np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0]]),
    }


def create_matrix_figure(
    matrix: np.ndarray, 
    title: Optional[str] = None, 
    vmin: float = 0, 
    vmax: Optional[float] = None
) -> plt.Figure:
    """Create a matplotlib figure with a matrix heatmap and values."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Determine color range if not provided
    if vmax is None:
        vmax = np.max(np.abs(matrix))
        if vmax == 0:  # Avoid division by zero
            vmax = 1.0
    
    # Create heatmap with consistent light-to-dark coloring for absolute values
    im = ax.imshow(np.abs(matrix), cmap="Blues", vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Absolute Value")
    
    # Add text annotations with actual values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            abs_value = np.abs(value)
            # Use white text for dark background, black text for light background
            text_color = "white" if abs_value > vmax / 2 else "black"
            ax.text(
                j,
                i,
                f"{value:.4f}",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    # Set ticks
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    
    # Label axes
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    
    # Tight layout
    fig.tight_layout()
    
    return fig


def batch_generator(X: torch.Tensor, Z: torch.Tensor, batch_size: int):
    """Generator that yields batches from PyTorch tensors X and Z"""
    B = Z.size(0)  # Get the first dimension (total samples)
    
    # Verify that X and Z have the same batch dimension
    assert X.size(0) == Z.size(0), "X and Z must have the same batch dimension"
    
    while True:  # Loop forever, allowing multiple complete passes
        # Shuffle indices for each complete pass
        indices = torch.randperm(B)
        
        # Iterate over batches
        for start_idx in range(0, B, batch_size):
            # Handle last batch which might be smaller than batch_size
            end_idx = min(start_idx + batch_size, B)
            batch_indices = indices[start_idx:end_idx]
            
            # Yield both X and Z batches
            yield X[batch_indices], Z[batch_indices]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_shd(
    A: np.ndarray, 
    B: np.ndarray, 
    threshold: float = 0.1
) -> int:
    """
    Compute the Structural Hamming Distance (SHD) between two matrices.
    
    Parameters:
    ----------
    A : np.ndarray
        First matrix (ground truth).
    B : np.ndarray
        Second matrix (estimated).
    threshold : float
        Threshold for considering an entry as non-zero.
        
    Returns:
    -------
    int
        The SHD between the two matrices.
    """
    # Ensure both matrices are binary
    A_binary = (np.abs(A) > threshold).astype(int)
    B_binary = (np.abs(B) > threshold).astype(int)
    
    # Compute the SHD
    shd = np.sum(np.abs(A_binary - B_binary))
    
    return shd

import numpy as np

def compute_shd_auto(
    A: np.ndarray, 
    B: np.ndarray, 
    num_iters: int = 10
) -> int:
    """
    Compute the Structural Hamming Distance (SHD) between two matrices using binary search
    to find a threshold that minimizes the SHD.

    Parameters
    ----------
    A : np.ndarray
        First matrix (ground truth).
    B : np.ndarray
        Second matrix (estimated).
    num_iters : int
        Number of binary search iterations to refine the threshold.

    Returns
    -------
    int
        The SHD between the two binarized matrices at the optimal threshold.
    """
    low = 0.0
    high = np.abs(A).min()
    best_shd = np.inf
    best_threshold = 0.0

    for threshold in np.linspace(low, high, num_iters):
        shd = compute_shd(A, B, threshold)
        if shd < best_shd:
            best_shd = shd
            best_threshold = threshold

    # Final SHD computation using best threshold
    return best_shd
