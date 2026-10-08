# surrogates/mlp.py
"""Data-only PINN ablation using the same PINNNet architecture.

The MLP shares PINN's ever_epochs, Adam optimizer and exponential learning-rate
schedule; only the loss changes to data MSE. No PDE, IC or BC information is
used. The common HF budget and update policy are retained. setup reads only
query_bounds for input dimensions, without accessing problem.physics.
"""
import numpy as np
import torch
import torch.nn as nn

from surrogates.base import Surrogate
from surrogates.pinn import PINNNet


class MLPSurrogate(Surrogate):
    name = "MLP"

    def __init__(self, config: dict = None):
        config = config or {}
        self.layers = config.get("layers", [2, 80, 80, 80, 80, 1])
        self.ever_epochs = int(config.get("ever_epochs", 300))
        self.lr_max = float(config.get("lr_max", 0.008))
        self.lr_min = float(config.get("lr_min", 0.000001))
        self.lr = float(config.get("lr", 0.008))
        self.lr_schedule = config.get("lr_schedule", "exponential")  # exponential | none

        self.seed_model_init = bool(config.get('seed_model_init', False))
        self.weight_decay = float(config.get('weight_decay', 0.0))
        self.model = None
        self.optimizer = None

    def setup(self, problem, seed: int, config: dict = None):
        if config:
            self.layers = config.get('layers', self.layers)
            self.ever_epochs = int(config.get('ever_epochs', self.ever_epochs))
            self.lr_max = float(config.get('lr_max', self.lr_max))
            self.lr_min = float(config.get('lr_min', self.lr_min))
            self.lr = float(config.get('lr', self.lr))
            self.lr_schedule = config.get('lr_schedule', self.lr_schedule)
            self.seed_model_init = bool(config.get('seed_model_init', self.seed_model_init))
            self.weight_decay = float(config.get('weight_decay', self.weight_decay))
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError('MLP weight_decay must be finite and nonnegative')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dim = int(np.asarray(problem.query_bounds, dtype=float).shape[0])
        if self.seed_model_init:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.model = PINNNet(self.layers).to(self.device)
        if self.weight_decay == 0.0:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam([{'params': [p for name, p in self.model.named_parameters() if name.endswith('weight')], 'weight_decay': self.weight_decay}, {'params': [p for name, p in self.model.named_parameters() if name.endswith('bias')], 'weight_decay': 0.0}], lr=self.lr)

    def _set_lr_by_gen(self, gen: int, maxgen: int) -> float:
        """Use the same exponential learning-rate schedule as PINN."""
        g = float(np.clip(gen, 0, maxgen))
        lr_now = self.lr_max * ((self.lr_min / self.lr_max) ** (g / maxgen))
        for pg in self.optimizer.param_groups:
            pg["lr"] = float(lr_now)
        return float(lr_now)

    def fit(self, Q: np.ndarray, u: np.ndarray, **ctx):
        gen = ctx.get("gen", 0)
        maxgen = ctx.get("maxgen", 1)
        if self.lr_schedule == "exponential":
            self._set_lr_by_gen(gen, maxgen)
        elif self.lr_schedule != "none":
            raise KeyError(f"Unknown lr_schedule: {self.lr_schedule}")

        Q = np.asarray(Q, dtype=float).reshape(-1, self._dim)
        q_cols = [torch.as_tensor(Q[:, j:j+1], dtype=torch.float32,
                                  device=self.device)
                  for j in range(self._dim)]
        # Keep target shape (n,1) to avoid pairwise broadcasting against (n,) labels.
        u_data = torch.as_tensor(np.asarray(u, dtype=np.float32).reshape(-1, 1),
                                 dtype=torch.float32, device=self.device)

        mse = nn.MSELoss()
        self.model.train()
        for epoch in range(self.ever_epochs):
            self.optimizer.zero_grad()
            loss = mse(self.model(*q_cols), u_data)
            loss.backward()
            self.optimizer.step()

    def predict(self, Q: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        Q = np.asarray(Q, dtype=float).reshape(-1, self._dim)
        N = Q.shape[0]
        pred = np.empty(N, dtype=float)
        self.model.eval()
        with torch.no_grad():
            for i in range(0, N, batch_size):
                j = min(i + batch_size, N)
                Qb = torch.tensor(Q[i:j], dtype=torch.float32, device=self.device)
                pred[i:j] = self.model(*[Qb[:, k].unsqueeze(-1) for k in range(Qb.shape[1])]).detach().cpu().numpy().reshape(-1)
        return pred
