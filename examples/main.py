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

def _fmt_M(n_tokens):
    """50_000_000 -> '50M'; 100_000 -> '0.1M'; trims '.0' suffix."""
    return '{:g}M'.format(n_tokens / 1_000_000)


def build_run_name(args):
    """Short, fixed-form run name for the leaf directory and wandb run."""
    parts = [
        f'mode={args.mode}',
        f'tau={args.tau}',
        f'z={args.z_dim}',
        f'topk={args.topk}',
        args.noise_mode,
        _fmt_M(args.total_tokens_int),
        f'seed={args.seed}',
    ]
    if args.mse_Zt:
        parts.append('mseZt')
    if args.normalize_activations:
        parts.append('norm')
    return '_'.join(parts)

class WandbLogger:
    """
    Thin wrapper so the training loop is wandb-agnostic.

    When wandb is disabled, missing, or fails to init, every method becomes
    a no-op and training proceeds unchanged. Local checkpoints + loss.json
    are the source of truth; wandb is a versioned shadow.
    """

    def __init__(self, args, logger):
        self.enabled = False
        self.run = None
        self.artifact_name = None
        self._wandb = None
        self._logger = logger

        if args.wandb_mode == 'disabled':
            logger.info('wandb logging disabled.')
            return

        try:
            import wandb
        except ImportError:
            logger.warning('wandb not installed; skipping wandb logging. '
                           'pip install wandb to enable.')
            return

        self._wandb = wandb
        try:
            self.run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                group=args.mode,           # cluster ablation arms in the UI
                job_type='train',
                tags=[f'mode:{args.mode}', f'noise:{args.noise_mode}',
                      f'topk:{args.topk}', f'tau:{args.tau}'],
                mode=args.wandb_mode,      # 'online' or 'offline'
                config={k: v for k, v in vars(args).items()
                        if not k.startswith('_')},
                dir=args.model_save_dir,
            )
            # n_tokens is the right x-axis: runs with different buffer sizes
            # still align on the same plot.
            wandb.define_metric('n_tokens')
            wandb.define_metric('train/*', step_metric='n_tokens')
            wandb.define_metric('eval/*', step_metric='n_tokens')

            # Artifact name: one per run; versions = milestones.
            # Sanitize (artifact names disallow '=', '.').
            self.artifact_name = (
                args.run_name.replace('=', '-').replace('.', '_')
            )
            self.enabled = True
            logger.info('wandb run: {} (project={}, mode={})'.format(
                self.run.name, args.wandb_project, args.wandb_mode))
        except Exception as e:
            logger.warning('wandb.init failed ({}); continuing without wandb.'.format(e))
            self.run = None

    def log_metrics(self, metrics, n_tokens):
        if not self.enabled:
            return
        try:
            payload = dict(metrics)
            payload['n_tokens'] = n_tokens
            self._wandb.log(payload)
        except Exception as e:
            self._logger.warning('wandb.log failed ({}); continuing.'.format(e))

    def log_checkpoint(self, ckp_path, n_tokens, aliases=None):
        """Upload `ckp_path` as a new version of the run's model artifact."""
        if not self.enabled:
            return
        try:
            artifact = self._wandb.Artifact(
                name=self.artifact_name,
                type='model',
                metadata={'n_tokens': n_tokens},
            )
            artifact.add_file(ckp_path)
            self.run.log_artifact(artifact, aliases=list(aliases or []))
            self._logger.info('wandb: logged checkpoint artifact {} (aliases={})'.format(
                os.path.basename(ckp_path), aliases))
        except Exception as e:
            self._logger.warning('wandb artifact upload failed for {} ({}); '
                                 'local copy still on disk.'.format(ckp_path, e))

    def finish(self):
        if not self.enabled:
            return
        try:
            self.run.finish()
        except Exception:
            pass


# ---------------- buffer / save ----------------

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


def _save_ckp(model, args, n_tokens, logger, tag=None):
    if tag is not None:
        fname = f'ckpt_{tag}.ckp'
    else:
        fname = 'ckpt_{:.2f}M.ckp'.format(n_tokens / 1_000_000)
    path = os.path.join(args.model_save_dir, fname)
    logger.info('Saving ckp at {:.2f}M tokens -> {}'.format(n_tokens / 1_000_000, path))
    model.eval()
    torch.save(model.state_dict(), path)
    return path

def train(model,
          activation_buffer,
          optimizer,
          args,
          logger,
          device,
          wlog,
          normalize_activations=False):

    from torch.amp import autocast, GradScaler
    scaler = GradScaler()

    use_temporal = args.mode in ('temporal', 'both')
    use_inst = args.mode in ('instantaneous', 'both')
    eff_l_spB = args.l_spB if use_temporal else 0.0
    eff_l_spM = args.l_spM if use_inst else 0.0
    logger.info('Mode: {:s} | l_spB={:g} (temporal {}) | l_spM={:g} (instantaneous {})'.format(
        args.mode, eff_l_spB, 'on' if use_temporal else 'off',
        eff_l_spM, 'on' if use_inst else 'off',
    ))

    n_refreshes = int(args.total_tokens // int(args.buffer_size * args.out_batch_ratio))
    if args.total_tokens % int(args.buffer_size * args.out_batch_ratio) > 0:
        n_refreshes = n_refreshes + 1

    tqdm_logger = TqdmToLogger(logger)
    progress_bar = tqdm(total=n_refreshes, file=tqdm_logger, desc="Training", dynamic_ncols=True)

    list_loss_mse_Xt = []
    list_loss_mse_Zt = []
    list_loss_indep = []
    list_loss_sparse_Bs = []
    list_loss_sparse_M = []
    list_loss_sparse_Zt = []

    n_tokens = 0
    model_saving_flags = [0, 0, 0]
    refresh_idx = 0
    last_ckp_path = None
    while n_tokens < args.total_tokens:
        activation_buffer.refresh()
        if (~activation_buffer.read).sum() >= activation_buffer.out_batch_size:
            out_batch = next(activation_buffer)
            act_batch = gen_window_slicing_batch(out_batch, window_size=args.tau + 1)

            model.train()
            act = act_batch.to(device)

            batch_size, act_dim, p = act.shape
            this_batch_tokens = batch_size + p - 1
            n_tokens = n_tokens + this_batch_tokens

            logging_mem_usage(logger=logger)

            if normalize_activations:
                norm_factor = get_norm_factor(act, steps=100)
                act = act / norm_factor

            progress_bar.set_description(
                f"[{args.mode}] Refresh {refresh_idx+1}/{n_refreshes} | "
                f"token {n_tokens/1_000_000:.2f}/{args.total_tokens/1_000_000:.0f}M"
            )

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                (loss_mse_Xt, loss_mse_Zt, loss_indep,
                 loss_sparse_Bs, loss_sparse_M, loss_sparse_Zt) = model(act)

                l_mse_Zt = 1.0 if args.mse_Zt else 0.0

                loss = (loss_mse_Xt
                        + l_mse_Zt * loss_mse_Zt
                        + args.l_ind * loss_indep
                        + eff_l_spB * loss_sparse_Bs
                        + eff_l_spM * loss_sparse_M
                        + args.l_spZ * loss_sparse_Zt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- bookkeeping ---
            v_loss = loss.item()
            v_mse_Xt = loss_mse_Xt.item()
            v_mse_Zt = loss_mse_Zt.item()
            v_indep = loss_indep.item()
            v_sp_B = loss_sparse_Bs.item()
            v_sp_M = loss_sparse_M.item()
            v_sp_Zt = loss_sparse_Zt.item()

            progress_bar.set_postfix({
                "loss": f"{v_loss:.4f}",
                "mse_Xt": f"{v_mse_Xt:.4f}",
                "mse_Zt": f"{v_mse_Zt:.4f}",
                "indep": f"{v_indep:.4f}",
                "sp_B": f"{v_sp_B:.4f}",
                "sp_M": f"{v_sp_M:.4f}",
                "sp_Zt": f"{v_sp_Zt:.4f}",
            })

            list_loss_mse_Xt.append(v_mse_Xt)
            list_loss_mse_Zt.append(v_mse_Zt)
            list_loss_indep.append(v_indep)
            list_loss_sparse_Bs.append(v_sp_B)
            list_loss_sparse_M.append(v_sp_M)
            list_loss_sparse_Zt.append(v_sp_Zt)

            # ---- wandb metrics ----
            # Raw losses ("is the model becoming sparse?") and weighted
            # contributions ("how much is each term pulling the loss down?").
            metrics = {
                'train/loss_total': v_loss,
                'train/loss_mse_Xt': v_mse_Xt,
                'train/loss_mse_Zt': v_mse_Zt,
                'train/loss_indep': v_indep,
                'train/loss_sparse_Bs': v_sp_B,
                'train/loss_sparse_M': v_sp_M,
                'train/loss_sparse_Zt': v_sp_Zt,

                'train/weighted/mse_Xt': v_mse_Xt,
                'train/weighted/mse_Zt': l_mse_Zt * v_mse_Zt,
                'train/weighted/indep': args.l_ind * v_indep,
                'train/weighted/sparse_Bs': eff_l_spB * v_sp_B,
                'train/weighted/sparse_M': eff_l_spM * v_sp_M,
                'train/weighted/sparse_Zt': args.l_spZ * v_sp_Zt,

                'train/refresh_idx': refresh_idx,
                'train/grad_scale': scaler.get_scale(),
            }
            wlog.log_metrics(metrics, n_tokens=n_tokens)

            progress_bar.update(1)

            # Save ckps at 25/50/75% milestones.
            # `latest` alias floats with each new milestone so eval scripts
            # can pull the most recent without knowing which milestone hit.
            if int(args.total_tokens * 0.25) < n_tokens <= int(args.total_tokens * 0.5):
                if model_saving_flags[0] == 0:
                    last_ckp_path = _save_ckp(model, args, n_tokens, logger)
                    wlog.log_checkpoint(last_ckp_path, n_tokens, aliases=['25pct', 'latest'])
                    model_saving_flags[0] = 1
            elif int(args.total_tokens * 0.5) < n_tokens <= int(args.total_tokens * 0.75):
                if model_saving_flags[1] == 0:
                    last_ckp_path = _save_ckp(model, args, n_tokens, logger)
                    wlog.log_checkpoint(last_ckp_path, n_tokens, aliases=['50pct', 'latest'])
                    model_saving_flags[1] = 1
            elif n_tokens > int(args.total_tokens * 0.75):
                if model_saving_flags[2] == 0:
                    last_ckp_path = _save_ckp(model, args, n_tokens, logger)
                    wlog.log_checkpoint(last_ckp_path, n_tokens, aliases=['75pct', 'latest'])
                    model_saving_flags[2] = 1

        refresh_idx += 1

    # Final ckp + 'final'/'latest' aliases.
    last_ckp_path = _save_ckp(model, args, n_tokens, logger)
    wlog.log_checkpoint(last_ckp_path, n_tokens, aliases=['final', 'latest'])

    final_alias = os.path.join(args.model_save_dir, 'ckpt_final.ckp')
    try:
        if os.path.lexists(final_alias):
            os.remove(final_alias)
        os.symlink(os.path.basename(last_ckp_path), final_alias)
        logger.info('Symlinked {} -> {}'.format(final_alias, os.path.basename(last_ckp_path)))
    except OSError:
        import shutil
        shutil.copyfile(last_ckp_path, final_alias)
        logger.info('Copied final ckp to {} (symlink unsupported)'.format(final_alias))

    logger.info('Saving loss to loss.json and plotting to loss.png')
    loss_dict = {
        'loss_mse_Xt': list_loss_mse_Xt,
        'loss_mse_Zt': list_loss_mse_Zt,
        'loss_indep': list_loss_indep,
        'loss_sparse_Bs': list_loss_sparse_Bs,
        'loss_sparse_M': list_loss_sparse_M,
        'loss_sparse_Zt': list_loss_sparse_Zt,
        'mode': args.mode,
    }
    with open(args.loss_path, 'w') as f:
        json.dump(loss_dict, f, indent=4)

    draw_loss(loss_dict=loss_dict, loss_path=args.loss_path, total_tokens=args.total_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)

    parser.add_argument('--z-dim', default=3072, type=int)
    parser.add_argument('--tau', default=20, type=int)
    parser.add_argument('--w', default=0.5, type=float)
    parser.add_argument('--noise-mode', default='lap', type=str, choices=['gau', 'lap'])
    parser.add_argument('--mse-Zt', default=False, action='store_true')
    parser.add_argument('--l-ind', default=0.1, type=float)
    parser.add_argument('--l-spB', default=0.01, type=float)
    parser.add_argument('--l-spM', default=0.01, type=float)
    parser.add_argument('--l-spZ', default=0.01, type=float)
    parser.add_argument('--normalize-activations', default=False, action='store_true')
    parser.add_argument('--topk', default=100, type=int, choices=[0, 25, 50, 100])

    parser.add_argument('--mode', default='both', type=str,
                        choices=['temporal', 'instantaneous', 'both'])

    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--run-name', default=None, type=str)
    parser.add_argument('--hgf-token', default='', type=str)

    parser.add_argument('--total-tokens', default='50M', type=str)
    parser.add_argument('--model-name', type=str, default='EleutherAI/pythia-160m-deduped')
    parser.add_argument('--layer', type=int, default=8)
    parser.add_argument('--out-batch-ratio', type=float, default=0.1)
    parser.add_argument('--buffer-size', default='0.1M', type=str)
    parser.add_argument('--text', default='monology/pile-uncopyrighted', type=str)

    parser.add_argument('--context', type=str, default='unspecified', choices=['unspecified'])

    # ---- wandb flags ----
    parser.add_argument('--wandb-project', default='coev-linearidol', type=str)
    parser.add_argument('--wandb-entity', default=None, type=str,
                        help='wandb entity (team/user); None uses your default.')
    parser.add_argument('--wandb-mode', default='online', type=str,
                        choices=['online', 'offline', 'disabled'],
                        help="online: live sync; offline: log to disk only "
                             "(sync later with `wandb sync`); disabled: no wandb.")

    args = parser.parse_args()
    assert args.total_tokens[-1].lower() == 'm'
    assert args.buffer_size[-1].lower() == 'm'

    args.total_tokens_int = int(float(args.total_tokens[:-1]) * 1_000_000)
    args.buffer_size_int = int(float(args.buffer_size[:-1]) * 1_000_000)

    args.run_name = args.run_name or build_run_name(args)
    config_name_keys = ['run_name']

    logger, log_path, loss_path, model_save_dir = setup_logging(args, config_name_keys=config_name_keys)
    args.log_path = log_path
    args.loss_path = loss_path
    args.model_save_dir = model_save_dir

    args.total_tokens = args.total_tokens_int
    args.buffer_size = args.buffer_size_int

    config_path = os.path.join(model_save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({k: v for k, v in vars(args).items() if not k.startswith('_')},
                  f, indent=2, default=str)
    logger.info('Wrote full config to {}'.format(config_path))
    logger.info('Run name: {}'.format(args.run_name))
    logger.info('Run dir : {}'.format(model_save_dir))

    if args.hgf_token:
        hugging_face_login(token=args.hgf_token)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Setting device: {:s}'.format(str(device)))

    logger.info('Setting global random seed: {:d}'.format(args.seed))
    set_seed(args.seed)

    # Init wandb BEFORE the model so init failures fail fast, and use a
    # try/finally so the run is closed even if training crashes.
    wlog = WandbLogger(args, logger)

    try:
        logger.info('Initiating data loader...')
        activation_buffer, act_dim = get_acts_buffer(
            model_name=args.model_name, text=args.text, layer=args.layer,
            buffer_size=args.buffer_size, out_batch_ratio=args.out_batch_ratio,
            device=device, dtype=torch.float32,
        )
        logger.info('Total n_tokens to train on: {:d}M, n_refresh: {:d}, buffer {:.2f}M tokens'.format(
            args.total_tokens // 1_000_000,
            args.total_tokens // args.buffer_size,
            args.buffer_size / 1_000_000))
        logger.info('Estimated CPU buffer: {:.2f}GB | Estimated GPU per-batch: {:.2f}GB'.format(
            args.buffer_size * 4 * act_dim / 1024 ** 3,
            (int(args.out_batch_ratio * args.buffer_size) - args.tau) * (args.tau + 1) * 4 * act_dim / (1024 ** 3),
        ))

        logger.info('Initiating LinearIDOL model in mode={:s}...'.format(args.mode))
        model = LinearIDOL(
            x_dim=act_dim,
            z_dim=args.z_dim,
            w=args.w,
            tau=args.tau,
            noise_mode=args.noise_mode,
            topk_sparsity=args.topk,
            mode=args.mode,
        ).to(device)

        optimizer = torch.optim.Adam(lr=args.lr, weight_decay=args.wd, params=model.parameters())

        logger.info('Training...')
        train(model=model,
              activation_buffer=activation_buffer,
              logger=logger,
              optimizer=optimizer,
              args=args,
              device=device,
              wlog=wlog,
              normalize_activations=args.normalize_activations)
    finally:
        wlog.finish()


if __name__ == "__main__":
    main()