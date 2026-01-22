from __future__ import annotations
import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    A simple MLP baseline: flatten parameters -> 8 floats.
    Strictly non-autoregressive and non-sequential encoder.
    """

    def __init__(self, in_dim: int = 12, out_dim: int = 8, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x_seq: torch.Tensor, y_teacher: torch.Tensor | None = None) -> torch.Tensor:
        # x_seq: (B, 4, 3) -> (B, 12)
        # y_teacher is ignored (dummy arg for compatibility)
        x = x_seq.reshape(x_seq.size(0), -1)
        return self.net(x)


class LSTMNonAutoreg(nn.Module):
    """
    [新增模型] LSTM Regression (Non-Autoreg).
    Encoder: LSTM over gate-parameter sequence (captures circuit structure).
    Decoder: Simple MLP Head -> Outputs all 8 values at once.

    Why this baseline?
    It proves that simply knowing the circuit structure (via LSTM) isn't enough;
    you need the autoregressive output structure (Seq2Seq) to be 'stable'.
    """

    def __init__(self, in_feat: int = 3, hidden: int = 128, layers: int = 2, out_len: int = 8):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_feat,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_len)  # Direct mapping to 8 outputs
        )

    def forward(self, x_seq: torch.Tensor, y_teacher: torch.Tensor | None = None) -> torch.Tensor:
        # x_seq: (B, 4, 3)
        # LSTM output: (B, Seq_Len, Hidden)
        out, _ = self.lstm(x_seq)

        # Take the embedding of the LAST gate step to represent the whole circuit
        last_hidden = out[:, -1, :]

        # Predict all 8 numbers simultaneously
        return self.head(last_hidden)


class Seq2SeqAutoreg(nn.Module):
    """
    Revised: Includes Positional Embedding to fix convergence issues.
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
        self.d_model = d_model

        # Encoder
        self.encoder = nn.LSTM(
            input_size=in_feat,
            hidden_size=d_model,
            num_layers=enc_layers,
            batch_first=True,
        )

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        # Input/Output Projections
        self.tok_in = nn.Linear(1, d_model)
        self.tok_out = nn.Linear(d_model, 1)

        # === FIX: Learnable Positional Embedding ===
        # Create a learnable vector for positions 0..out_len (plus a buffer)
        self.pos_emb = nn.Parameter(torch.randn(1, out_len + 1, d_model))

    @staticmethod
    def causal_mask(T: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x_seq: torch.Tensor, y_teacher: torch.Tensor | None = None) -> torch.Tensor:
        # Encoder output
        memory, _ = self.encoder(x_seq)  # (B, L, d_model)

        # 1. Training Mode
        if y_teacher is not None:
            B, T = y_teacher.shape
            y_in = torch.zeros((B, T), device=y_teacher.device, dtype=y_teacher.dtype)
            y_in[:, 1:] = y_teacher[:, :-1]

            # Embed + Add Position Info
            tgt = self.tok_in(y_in.unsqueeze(-1))  # (B, T, d_model)
            tgt = tgt + self.pos_emb[:, :T, :]  # <--- Add Pos Emb here

            mask = self.causal_mask(T, tgt.device)
            h = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)
            y_pred = self.tok_out(h).squeeze(-1)
            return y_pred

        # 2. Inference Mode
        B = x_seq.size(0)
        ys = torch.zeros((B, 1), device=x_seq.device, dtype=x_seq.dtype)

        for t in range(self.out_len):
            # Embed current sequence
            tgt = self.tok_in(ys.unsqueeze(-1))  # (B, t+1, d_model)

            # Add Position Info (Critical for inference too)
            current_len = tgt.size(1)
            tgt = tgt + self.pos_emb[:, :current_len, :]

            mask = self.causal_mask(current_len, tgt.device)
            h = self.decoder(tgt=tgt, memory=memory, tgt_mask=mask)

            next_scalar = self.tok_out(h[:, -1:, :]).squeeze(-1)
            ys = torch.cat([ys, next_scalar], dim=1)

        return ys[:, 1:]