from __future__ import annotations
import torch
import torch.nn as nn

class MLPBaseline(nn.Module):
    """A simple non-autoregressive baseline: flatten parameters -> 8 floats."""
    def __init__(self, in_dim: int = 12, out_dim: int = 8, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, 4, 3) -> (B, 12)
        x = x_seq.reshape(x_seq.size(0), -1)
        return self.net(x)

class Seq2SeqAutoreg(nn.Module):
    """
    Encoder: LSTM over gate-parameter sequence (length=4, feature=3)
    Decoder: TransformerDecoder, autoregressively generates y[0..T-1] scalars (T=8).
    Training: teacher forcing with shifted targets, MSE loss.
    """
    def __init__(
        self,
        in_feat: int = 3,
        enc_layers: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        dec_layers: int = 2,
        out_len: int = 8,
    ):
        super().__init__()
        self.out_len = out_len
        self.encoder = nn.LSTM(
            input_size=in_feat,
            hidden_size=d_model,
            num_layers=enc_layers,
            batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        # scalar tokenization for targets
        self.tok_in = nn.Linear(1, d_model)
        self.tok_out = nn.Linear(d_model, 1)

    @staticmethod
    def causal_mask(T: int, device) -> torch.Tensor:
        # True => masked
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x_seq: torch.Tensor, y_teacher: torch.Tensor | None = None) -> torch.Tensor:
        memory, _ = self.encoder(x_seq)  # (B, L, d_model)

        if y_teacher is not None:
            # teacher forcing: decoder input is shifted y
            B, T = y_teacher.shape
            y_in = torch.zeros((B, T), device=y_teacher.device, dtype=y_teacher.dtype)
            y_in[:, 1:] = y_teacher[:, :-1]
            tgt = self.tok_in(y_in.unsqueeze(-1))  # (B, T, d_model)
            mask = self.causal_mask(T, tgt.device)
            h = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)  # (B, T, d_model)
            y_pred = self.tok_out(h).squeeze(-1)  # (B, T)
            return y_pred

        # autoregressive inference
        B = x_seq.size(0)
        ys = torch.zeros((B, 1), device=x_seq.device, dtype=x_seq.dtype)  # start token = 0
        for _ in range(self.out_len):
            tgt = self.tok_in(ys.unsqueeze(-1))  # (B, t+1, d_model)
            mask = self.causal_mask(tgt.size(1), tgt.device)
            h = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)
            next_scalar = self.tok_out(h[:, -1:, :]).squeeze(-1)  # (B, 1)
            ys = torch.cat([ys, next_scalar], dim=1)
        return ys[:, 1:]  # drop start token
