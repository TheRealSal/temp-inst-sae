import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from nnsight import LanguageModel
from linear_idol_model import LinearIDOL

from utils import *

# import disctionary_learning 
sys.path.insert(0, './examples/submodule_dl')
import dictionary_learning.utils as utils
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import get_norm_factor

def get_acts_buffer(model_name,
                    text,
                    layer,
                    buffer_size, 
                    out_batch_ratio,
                    dtype, 
                    device,
                    ctx_len=128):

    n_ctx = buffer_size // ctx_len

    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)
    submodule = utils.get_submodule(model, layer)
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    act_dim = model.config.hidden_size

    generator = hf_dataset_to_generator(text)

    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=n_ctx,
        ctx_len=ctx_len,
        out_batch_size=int(buffer_size * out_batch_ratio),
        io=io,
        d_submodule=act_dim,
        device=device,
    )
    
    return activation_buffer, act_dim

def train(model, 
          activation_buffer, 
          optimizer, 
          args,
          logger,
          device, 
          normalize_activations=False):
    
    # Prepare pbar log 
    n_refreshes = int(args.total_tokens // int(args.buffer_size * args.out_batch_ratio))
    if args.total_tokens % int(args.buffer_size * args.out_batch_ratio) > 0:
        n_refreshes = n_refreshes + 1
    
    # Use this in place of regular tqdm
    tqdm_logger = TqdmToLogger(logger)
    total_steps = n_refreshes
    progress_bar = tqdm(total=total_steps, file=tqdm_logger, desc="Training", dynamic_ncols=True)
    # progress_bar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)

    list_loss_mse_Xt = []
    list_loss_mse_Zt = []
    list_loss_indep = []
    list_loss_sparse_Bs = []
    list_loss_sparse_M = []
    list_loss_sparse_Zt = []

    n_tokens = 0
    model_saving_flags = [0,0,0]
    refresh_idx = 0
    while n_tokens < args.total_tokens:
        activation_buffer.refresh()
        if (~activation_buffer.read).sum() >= activation_buffer.out_batch_size:
            out_batch = next(activation_buffer) # output from activation buffer
            act_batch = gen_window_slicing_batch(out_batch, window_size=args.tau+1)
            # Forward and training 
            model.train() 
            act = act_batch.to(device)

            # Compute tokens
            batch_size, act_dim, p = act.shape
            this_batch_tokens = batch_size + p - 1 # p = tau + 1
            n_tokens = n_tokens + this_batch_tokens

            # Log the mem usage status
            logging_mem_usage(logger=logger)

            # Whether normalize activations
            if normalize_activations:
                norm_factor = get_norm_factor(act, steps=100)
                act = act / norm_factor

            progress_bar.set_description(f"Refresh {refresh_idx+1}/{n_refreshes} | token {n_tokens/1_000_000}/{args.total_tokens/1_000_000}M")
            
            loss_mse_Xt, loss_mse_Zt, loss_indep, loss_sparse_Bs, loss_sparse_M, loss_sparse_Zt = model(act)
            
            l_mse_Zt = 0.
            # Enable loss_mse_Zt or not
            if args.mse_Zt:
                l_mse_Zt = 1.
                
            loss = loss_mse_Xt + l_mse_Zt * loss_mse_Zt + args.l_ind * loss_indep + args.l_spB * loss_sparse_Bs + args.l_spM * loss_sparse_M + args.l_spZ * loss_sparse_Zt

            # pbar log
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "loss_mse_Xt": f"{loss_mse_Xt:.4f}",
                "loss_mse_Zt": f"{loss_mse_Zt:.4f}",
                "loss_indep": f"{loss_indep:.4f}",
                "loss_sp_B": f"{loss_sparse_Bs:.4f}",
                "loss_sp_M": f"{loss_sparse_M:.4f}",
                "loss_sp_Zt": f"{loss_sparse_Zt:.4f}",
            })

            list_loss_mse_Xt.append(loss_mse_Xt.item())
            list_loss_mse_Zt.append(loss_mse_Zt.item())
            list_loss_indep.append(loss_indep.item())
            list_loss_sparse_Bs.append(loss_sparse_Bs.item())
            list_loss_sparse_M.append(loss_sparse_M.item())
            list_loss_sparse_Zt.append(loss_sparse_Zt.item())

            # Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)

            # Save ckps
            if n_tokens > int(args.total_tokens * 0.25) and n_tokens <= int(args.total_tokens * 0.5):
                if model_saving_flags[0] == 0:
                    model_name = os.path.basename(args.model_save_dir) + 'Token_{:.2f}M.ckp'.format(n_tokens / 1_000_000)
                    model_save_path = os.path.join(args.model_save_dir, model_name)
                    logger.info('Saving ckp at training tokens of {:.2f}M at location {:s}...'.format(n_tokens/1_000_000, model_save_path))
                    model.eval()
                    torch.save(model.state_dict(), model_save_path)
                    model_saving_flags[0] = 1
            elif n_tokens > int(args.total_tokens * 0.5) and n_tokens <= int(args.total_tokens * 0.75):
                if model_saving_flags[1] == 0:
                    model_name = os.path.basename(args.model_save_dir) + 'Token_{:.2f}M.ckp'.format(n_tokens / 1_000_000)
                    model_save_path = os.path.join(args.model_save_dir, model_name)
                    logger.info('Saving ckp at training tokens of {:.2f}M at location {:s}...'.format(n_tokens/1_000_000, model_save_path))
                    model.eval()
                    torch.save(model.state_dict(), model_save_path)
                    model_saving_flags[1] = 1
            elif n_tokens > int(args.total_tokens * 0.75):
                if model_saving_flags[2] == 0:
                    model_name = os.path.basename(args.model_save_dir) + 'Token_{:.2f}M.ckp'.format(n_tokens / 1_000_000)
                    model_save_path = os.path.join(args.model_save_dir, model_name)
                    logger.info('Saving ckp at training tokens of {:.2f}M at location {:s}...'.format(n_tokens/1_000_000, model_save_path))
                    model.eval()
                    torch.save(model.state_dict(), model_save_path)
                    model_saving_flags[2] = 1
    
        # refresh_idx ++1
        refresh_idx += 1

    # Save the final ckp, log and plot the losses
    model_name = os.path.basename(args.model_save_dir) + 'Token_{:.2f}M.ckp'.format(n_tokens / 1_000_000)
    model_save_path = os.path.join(args.model_save_dir, model_name)
    logger.info('Saving ckp at training tokens of {:.2f}M at location {:s}...'.format(n_tokens/1_000_000, model_save_path))
    model.eval()
    torch.save(model.state_dict(), model_save_path)

    logger.info('Saving loss to loss.json and plotting to loss.png')
    loss_dict = {
        'loss_mse_Xt': list_loss_mse_Xt,
        'loss_mse_Zt': list_loss_mse_Zt,
        'loss_indep': list_loss_indep,
        'loss_sparse_Bs': list_loss_sparse_Bs,
        'loss_sparse_M': list_loss_sparse_M,
        'loss_sparse_Zt': list_loss_sparse_Zt
    }
    with open(args.loss_path, 'w') as f:
        json.dump(loss_dict, f, indent=4)

    draw_loss(loss_dict=loss_dict, loss_path=args.loss_path, total_tokens=args.total_tokens)
    
                                                                                                                                                               
def main():
    parser = argparse.ArgumentParser()
    # Training configs in general
    parser.add_argument('--seed', default=123, type=int, help='random seed')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    
    # Model configs
    parser.add_argument('--z-dim', default=3072, type=int)
    parser.add_argument('--tau', default=20, type=int)
    parser.add_argument('--w', default=0.5, type=float) # disabled for now
    # use gaussian or laplacian to constrain the noise to be independent from each other
    # notice that the loss function is slightly different regarding the two different modes
    parser.add_argument('--noise-mode', default='lap', type=str, choices=['gau', 'lap'])
    parser.add_argument('--mse-Zt', default=False, action='store_true', help='enable to use mse on Zt when training B and M')
    parser.add_argument('--l-ind', default=0.1, type=float)
    parser.add_argument('--l-spB', default=0.1, type=float)
    parser.add_argument('--l-spM', default=0.1, type=float)
    parser.add_argument('--l-spZ', default=0.1, type=float)
    parser.add_argument('--normalize-activations', default=False, action='store_true')
    parser.add_argument('--topk', default=100, type=int, help='the topk spartisy; if it is set to 0 (by default), then only use l1 sparsity', choices=[0, 25, 50, 100])
    parser.add_argument('--results-dir', required=True, help='The dir to saving results, including logs, losses (plots), and ckps')
    
    # Hugging face login
    parser.add_argument('--hgf-token', default='', type=str, help='The hugging face login tokens, for avoiding request limited when the number of total tokens is big or many conccurnet tasks.')
    
    # General LLM activations datasets configs
    parser.add_argument('--total-tokens', default='10M', type=str, help='(Unit M) the total number of tokens to train on')
    parser.add_argument('--model-name', type=str, default='EleutherAI/pythia-160m-deduped')
    parser.add_argument('--layer', type=int, default=8)
    parser.add_argument('--out-batch-ratio', type=float, default=0.1, help='Specifies the fraction of the activation buffer to use as the output batch size. The out_batch_size is computed as: int(out_batch_ratio * buffer_size)')
    parser.add_argument('--buffer-size', default='0.1M', type=str, help='(Unit M) the size of ActivationBuffer; note that the actual batch_size during training is (buffer_size - tau)')
    parser.add_argument('--text', default='monology/pile-uncopyrighted', type=str, help='the name of the text ot use, by default is the unspecifed corpus monology/pile-uncopyrighted')    
    
    # Contexual LLM datasets configs (synthetic real-world)
    parser.add_argument('--context', type=str, default='unspecified', help='the context of the tokens', choices=['unspecified'])

    # Synthetic datasets configs
    # TODO

    # Read args
    args = parser.parse_args()
    assert args.total_tokens[-1].lower() == 'm'
    assert args.buffer_size[-1].lower() == 'm'
    args.total_tokens = str(float(args.total_tokens[:-1])) + 'M'
    args.buffer_size = str(float(args.buffer_size[:-1])) + 'M'

    # Pick up params to store the ckps and log
    config_name_keys = ['z_dim', 'total_tokens', 'noise_mode', 'topk', 'buffer_size', 'layer', 'tau', 'mse_Zt', 'l_ind', 'l_spB', 'l_spB', 'l_spZ', 'normalize_activations', 'out_batch_ratio', 'lr', 'wd', 'seed'] 

    # Staring logging
    logger, log_path, loss_path, model_save_dir = setup_logging(args, config_name_keys=config_name_keys)
    args.log_path = log_path
    args.loss_path = loss_path
    args.model_save_dir = model_save_dir

    args.total_tokens = int(float(args.total_tokens[:-1]) * 1_000_000)
    args.buffer_size = int(float(args.buffer_size[:-1]) * 1_000_000)

    # Hugging face login
    hugging_face_login(token=args.hgf_token)

    # Set device and random seed
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    logger.info('Setting device: {:s}'.format(str(device)))
    
    logger.info('Setting global random seed: {:d}'.format(args.seed))
    set_seed(args.seed)

    # Init data loader
    logger.info('Initiating data loader...')
    activation_buffer, act_dim = get_acts_buffer(model_name=args.model_name,
                                                text=args.text,
                                                layer=args.layer,
                                                buffer_size=args.buffer_size,
                                                out_batch_ratio=args.out_batch_ratio,
                                                device=device, # buffer is store in CPU by default
                                                dtype=torch.float32)
    logger.info('Total n_tokens to train on: {:d}M, n_refresh: {:d} of the activation buffer with each buffer of {:.2f}M tokens'.format(
        (args.total_tokens // 1_000_000), 
        (args.total_tokens // args.buffer_size),
        (args.buffer_size / 1_000_000)))
    logger.info('Please check the esimated mem usage below to adjust your params carefully:')
    logger.info('Esimated CPU usage for activation buffer: {:.2f}GB, estimated GPU usage of one batch when a full buffer is processed: {:.2f}GB'.format(
        (args.buffer_size * 4 * act_dim / 1024 **3),
        ((int(args.out_batch_ratio * args.buffer_size) - args.tau) * (args.tau + 1) * 4 * act_dim / (1024 **3))))
    
    # Init LinearIDOL
    logger.info('Initiating LinearIDOL model...')
    model = LinearIDOL(x_dim=act_dim, 
                       z_dim=args.z_dim, 
                       w=args.w,
                       tau=args.tau,
                       noise_mode=args.noise_mode,
                       topk_sparsity=args.topk).to(device)
    
    # Optimization
    optimizer = torch.optim.Adam(lr=args.lr, weight_decay=args.wd, params=model.parameters())
    
    logger.info('Training...')
    train(model=model, 
          activation_buffer=activation_buffer, 
          logger=logger, 
          optimizer=optimizer, 
          args=args, 
          device=device,
          normalize_activations=args.normalize_activations)

if __name__ == "__main__":
    main()
