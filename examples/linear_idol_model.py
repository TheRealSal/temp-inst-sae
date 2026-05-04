import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LinearIDOL(nn.Module):
    """
    Linear IDOL with selectable causal-structure ablation.

    mode:
        'temporal'      -- only lagged effects (B_1, ..., B_tau); M is disabled.
        'instantaneous' -- only instantaneous effect (M); B_lags are disabled.
        'both'          -- temporal + instantaneous (original behavior).

    Disabled branches return zero-valued losses (as tensors) and do not
    contribute to Zt, so the parameters of disabled branches receive no
    gradient and remain at their init values.
    """

    VALID_MODES = ('temporal', 'instantaneous', 'both')

    def __init__(self,
                 x_dim,
                 z_dim,
                 tau,
                 w,
                 noise_mode,
                 topk_sparsity=0,
                 mode='both'):
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}, got '{mode}'."
            )
        if mode == 'instantaneous' and tau != 0:
            # tau is still used to slice Z_p[:, :, tau] as "the current step",
            # so we keep it; just note that B_lags exist but are unused.
            pass

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.tau = tau
        self.w = w  # time-delayed contribution weight to Zt (used iff enable_w)
        self.noise_mode = noise_mode
        self.topk_sparsity = topk_sparsity  # 0 -> use l1 sparsity instead of topk
        self.mode = mode

        # Encoder / decoder always present (Xt reconstruction is shared).
        self.F_enc = nn.Parameter(torch.ones(self.x_dim, self.z_dim), requires_grad=True)
        self.F_dec = nn.Parameter(torch.ones(self.z_dim, self.x_dim), requires_grad=True)

        # Temporal branch: lagged transition matrices B_1, ..., B_tau.
        # Always allocated for state-dict compatibility, but only used when
        # the temporal branch is enabled.
        self.Bs = nn.ParameterList([
            nn.Parameter(torch.zeros(self.z_dim, self.z_dim), requires_grad=self._uses_temporal())
            for _ in range(tau)
        ])

        # Instantaneous branch: M.
        self.M = nn.Parameter(
            torch.ones(self.z_dim, self.z_dim),
            requires_grad=self._uses_instantaneous(),
        )

        self.init_params()

    def _uses_temporal(self):
        return self.mode in ('temporal', 'both')

    def _uses_instantaneous(self):
        return self.mode in ('instantaneous', 'both')

    def init_params(self):
        nn.init.xavier_normal_(self.F_enc.data)
        nn.init.xavier_normal_(self.F_dec.data)
        if self._uses_instantaneous():
            nn.init.xavier_normal_(self.M.data)
        # Bs are intentionally left at zero init (matches original code).

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def _encode_decode(self, Xp):
        """Returns (Zp, recons_Xp, loss_mse_Xt)."""
        Zp = torch.einsum('hd,bdt->bht', self.F_enc.T, Xp)
        recons_Xp = torch.einsum('dh,bht->bdt', self.F_dec.T, Zp)
        loss_mse_Xt = F.mse_loss(recons_Xp[:, :, -1], Xp[:, :, -1])
        return Zp, recons_Xp, loss_mse_Xt

    def _instantaneous_contribution(self, Zp, _w, device, dtype):
        """
        Returns (Zt_inst, M_used, loss_sparse_M).
        If instantaneous branch is disabled, returns zeros with correct shapes
        so downstream code is uniform.
        """
        if not self._uses_instantaneous():
            B = Zp.shape[0]
            zero_Zt = torch.zeros(B, self.z_dim, device=device, dtype=dtype)
            zero_loss = torch.zeros((), device=device, dtype=dtype)
            return zero_Zt, None, zero_loss

        # NOTE: diagonal=1 keeps the main diagonal AND first super-diagonal,
        # which lets Z_t^{(i)} predict itself instantaneously. If you want a
        # strict DAG (no self-loops), use diagonal=-1. Keeping original
        # behavior here so this refactor is loss-preserving.
        M_used = torch.tril(self.M, diagonal=1)
        Zt_inst = _w * torch.einsum('hd,bd->bh', M_used, Zp[:, :, self.tau])
        loss_sparse_M = F.l1_loss(M_used, torch.zeros_like(M_used))
        return Zt_inst, M_used, loss_sparse_M

    def _temporal_contribution(self, Zp, w, device, dtype):
        """
        Returns (Zt_temp, loss_sparse_Bs).
        If temporal branch is disabled, returns zeros.
        """
        B = Zp.shape[0]
        if not self._uses_temporal():
            zero_Zt = torch.zeros(B, self.z_dim, device=device, dtype=dtype)
            zero_loss = torch.zeros((), device=device, dtype=dtype)
            return zero_Zt, zero_loss

        Zt_temp = torch.zeros(B, self.z_dim, device=device, dtype=dtype)
        loss_sparse_Bs = torch.zeros((), device=device, dtype=dtype)
        for lag in range(1, self.tau + 1):
            B_lag = self.Bs[lag - 1]
            loss_sparse_Bs = loss_sparse_Bs + F.l1_loss(B_lag, torch.zeros_like(B_lag))
            Zt_lag = Zp[:, :, self.tau - lag]
            Zt_temp = Zt_temp + w * torch.einsum('hd,bd->bh', B_lag, Zt_lag)
        return Zt_temp, loss_sparse_Bs

    def _apply_topk(self, Zt):
        if self.topk_sparsity <= 0:
            return Zt
        Zt_abs = torch.abs(Zt)
        _, topk_indices = torch.topk(Zt_abs, self.topk_sparsity, dim=1)
        mask = torch.zeros_like(Zt)
        mask.scatter_(1, topk_indices, 1.0)
        return Zt * mask

    def _independence_loss(self, Et):
        if self.noise_mode == 'gau':
            return torch.trace(torch.cov(Et))  # -logp up to const
        elif self.noise_mode == 'lap':
            return F.l1_loss(Et, torch.zeros_like(Et))
        else:
            raise NotImplementedError(f"noise_mode={self.noise_mode}")

    def forward(self, Xp, enable_w=False):
        if not self.training:
            self.topk_sparsity = 0

        device, dtype = Xp.device, Xp.dtype

        # 1. Encode / decode (always)
        Zp, _, loss_mse_Xt = self._encode_decode(Xp)

        # 2. Branch weights
        if enable_w:
            w_temp = self.w
            w_inst = 1. - self.w
        else:
            w_temp = 1.
            w_inst = 1.

        # 3. Branch contributions
        Zt_inst, _, loss_sparse_M = self._instantaneous_contribution(Zp, w_inst, device, dtype)
        Zt_temp, loss_sparse_Bs = self._temporal_contribution(Zp, w_temp, device, dtype)

        # 4. Combine and (optionally) sparsify Zt
        Zt = Zt_inst + Zt_temp
        Zt = self._apply_topk(Zt)

        # 5. Reconstruction of Z_t and innovations
        loss_mse_Zt = F.mse_loss(Zt, Zp[:, :, self.tau])
        Et = Zp[:, :, self.tau] - Zt
        loss_indep = self._independence_loss(Et)

        # 6. L1 sparsity on Zt
        loss_sparse_Zt = F.l1_loss(Zt, torch.zeros_like(Zt))

        return (loss_mse_Xt, loss_mse_Zt, loss_indep,
                loss_sparse_Bs, loss_sparse_M, loss_sparse_Zt)