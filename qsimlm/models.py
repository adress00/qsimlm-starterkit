from __future__ import annotations
import torch
import torch.nn as nn

class MLPBaseline(nn.Module):
    """A simple non-autoregressive baseline: flatten parameters -> 8 floats (statevector)."""
    def __init__(self, in_dim: int = 12, out_dim: int = 8, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, 4, 3) -> (B, 12)
        x = x_seq.reshape(x_seq.size(0), -1)
        return self.net(x)


class MLPDensityMatrix(nn.Module):
    """MLP for density matrix prediction (lower triangular + real/imag parts)."""
    def __init__(self, in_dim: int = 12, n_qubits: int = 2, hidden: int = 512):
        super().__init__()
        dim = 2 ** n_qubits
        self.out_dim = dim * (dim + 1)  # Lower triangular: D*(D+1)/2 complex = D*(D+1) real
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, 4, 3) -> (B, 12)
        x = x_seq.reshape(x_seq.size(0), -1)
        return self.net(x)

class MLPFlex(nn.Module):
    """
    Flexible-depth MLP: builds Linear/ReLU stacks from a provided hidden list.

    This allows restoring checkpoints trained with different hidden sizes or
    depths than the default baseline. Input is the flattened 12-dim parameters
    and output is 8 real/imag amplitudes.
    """
    def __init__(self, in_dim: int = 12, out_dim: int = 8, hidden_layers: list[int] | None = None):
        super().__init__()
        if hidden_layers is None or len(hidden_layers) == 0:
            hidden_layers = [256, 256]
        layers: list[nn.Module] = []
        dim_in = in_dim
        for h in hidden_layers:
            layers.append(nn.Linear(dim_in, h))
            layers.append(nn.ReLU())
            dim_in = h
        layers.append(nn.Linear(dim_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x = x_seq.reshape(x_seq.size(0), -1)
        return self.net(x)

class Seq2SeqAutoreg(nn.Module):
    """
    Encoder-Decoder architecture with detailed specifications:
    
    Encoder: 3-layer LSTM with 399 hidden units per layer.
    - Input: quantum circuit parameter sequence (batch, seq_len, param_dim)
    - Output: context vector from the last time step
    
    Decoder: Multi-head Self-Attention + Autoregressive generation.
    - Uses Self-Attention to identify relevant features in context vector
    - Combines context with previously generated sequence via FFN
    - Generates quantum state components autoregressively
    
    Training: Teacher Forcing strategy (ground truth as input)
    Inference: Autoregressive iterative generation
    Loss: Mean Squared Error (MSE)
    """
    def __init__(
        self,
        in_feat: int = 3,
        enc_hidden: int = 399,
        enc_layers: int = 3,
        d_model: int = 399,
        nhead: int = 3,
        dec_layers: int = 2,
        dim_feedforward: int = 1024,
        out_len: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_len = out_len
        self.d_model = d_model
        
        # Encoder: 3-layer LSTM with 399 hidden units
        self.encoder = nn.LSTM(
            input_size=in_feat,
            hidden_size=enc_hidden,
            num_layers=enc_layers,
            batch_first=True,
            dropout=dropout if enc_layers > 1 else 0,
        )
        self.enc_hidden = enc_hidden
        
        # Context projection: from encoder hidden state to decoder d_model
        self.context_proj = nn.Linear(enc_hidden, d_model)
        
        # Decoder: Self-Attention based generation
        # Scalar tokenization for input
        self.tok_in = nn.Linear(1, d_model)
        
        # Decoder layers with self-attention and cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu',
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        
        # Output projection
        self.tok_out = nn.Linear(d_model, 1)

    @staticmethod
    def causal_mask(T: int, device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation (upper triangular)."""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x_seq: torch.Tensor, y_teacher: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass with Teacher Forcing (training) or Autoregressive (inference).
        
        Args:
            x_seq: (batch, seq_len=4, param_dim=3) - encoded quantum circuit parameters
            y_teacher: (batch, out_len) - ground truth target sequence for teacher forcing
                      None for autoregressive inference
        
        Returns:
            Predicted quantum state components (batch, out_len)
        """
        batch_size = x_seq.size(0)
        
        # Encoder: process parameter sequence and get context vector
        encoder_output, (h_n, c_n) = self.encoder(x_seq)
        # h_n: (num_layers=3, batch, hidden_size=399)
        # Use last layer's hidden state as context
        context = h_n[-1]  # (batch, 399)
        context = self.context_proj(context)  # (batch, d_model)
        
        # Expand context to match decoder sequence length for cross-attention
        # memory: (batch, 1, d_model) - used as cross-attention key/value
        memory = context.unsqueeze(1)  # (batch, 1, d_model)
        
        if y_teacher is not None:
            # Training mode: Teacher Forcing
            # Shift targets: y_in[t] = y_teacher[t-1] (start with 0)
            T = y_teacher.shape[1]
            y_in = torch.zeros((batch_size, T), device=y_teacher.device, dtype=y_teacher.dtype)
            y_in[:, 1:] = y_teacher[:, :-1]
            
            # Tokenize input scalars
            tgt = self.tok_in(y_in.unsqueeze(-1))  # (batch, T, d_model)
            
            # Generate causal mask for self-attention
            mask = self.causal_mask(T, tgt.device)
            
            # Decoder with self-attention on generated sequence + cross-attention on context
            h = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=mask,
                memory_mask=None,
            )  # (batch, T, d_model)
            
            # Output projection
            y_pred = self.tok_out(h).squeeze(-1)  # (batch, T)
            return y_pred
        
        else:
            # Inference mode: Autoregressive generation
            # Iteratively generate each token using previously generated tokens
            ys = torch.zeros((batch_size, 1), device=x_seq.device, dtype=x_seq.dtype)
            
            for step in range(self.out_len):
                # Tokenize current sequence
                tgt = self.tok_in(ys.unsqueeze(-1))  # (batch, step+1, d_model)
                
                # Generate causal mask
                mask = self.causal_mask(tgt.size(1), tgt.device)
                
                # Decoder forward pass
                h = self.decoder(
                    tgt=tgt,
                    memory=memory,
                    tgt_mask=mask,
                    memory_mask=None,
                )  # (batch, step+1, d_model)
                
                # Predict next token from last position
                next_scalar = self.tok_out(h[:, -1:, :]).squeeze(-1)  # (batch, 1)
                
                # Append to sequence
                ys = torch.cat([ys, next_scalar], dim=1)
            
            return ys[:, 1:]  # drop start token, return (batch, out_len)


class Seq2SeqAutoregDensityMatrix(nn.Module):
    """
    Encoder-Decoder architecture for density matrix prediction (lower triangular).
    
    Similar to Seq2SeqAutoreg but outputs D*(D+1) values for a DxD density matrix.
    For 2-qubit (D=4): 20 values
    For 3-qubit (D=8): 72 values
    """
    def __init__(
        self,
        in_feat: int = 3,
        enc_hidden: int = 399,
        enc_layers: int = 3,
        d_model: int = 399,
        nhead: int = 3,
        dec_layers: int = 2,
        dim_feedforward: int = 1024,
        n_qubits: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim = 2 ** n_qubits
        self.out_len = dim * (dim + 1)  # Output length for lower triangular
        self.d_model = d_model
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=in_feat,
            hidden_size=enc_hidden,
            num_layers=enc_layers,
            batch_first=True,
            dropout=dropout if enc_layers > 1 else 0,
        )
        self.enc_hidden = enc_hidden
        self.context_proj = nn.Linear(enc_hidden, d_model)
        
        # Decoder
        self.tok_in = nn.Linear(1, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu',
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        self.tok_out = nn.Linear(d_model, 1)

    @staticmethod
    def causal_mask(T: int, device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation (upper triangular)."""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x_seq: torch.Tensor, y_teacher: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for density matrix prediction.
        
        Args:
            x_seq: (batch, 4, 3) circuit parameters
            y_teacher: (batch, D*(D+1)) ground truth lower triangular DM
                      None for autoregressive inference
        
        Returns:
            (batch, D*(D+1)) predicted lower triangular density matrix
        """
        batch_size = x_seq.size(0)
        
        # Encoder
        encoder_output, (h_n, c_n) = self.encoder(x_seq)
        context = h_n[-1]
        context = self.context_proj(context)
        memory = context.unsqueeze(1)
        
        if y_teacher is not None:
            # Teacher forcing
            T = y_teacher.shape[1]
            y_in = torch.zeros((batch_size, T), device=y_teacher.device, dtype=y_teacher.dtype)
            y_in[:, 1:] = y_teacher[:, :-1]
            
            tgt = self.tok_in(y_in.unsqueeze(-1))
            mask = self.causal_mask(T, tgt.device)
            
            h = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=mask,
                memory_mask=None,
            )
            
            y_pred = self.tok_out(h).squeeze(-1)
            return y_pred
        
        else:
            # Autoregressive generation
            ys = torch.zeros((batch_size, 1), device=x_seq.device, dtype=x_seq.dtype)
            
            for step in range(self.out_len):
                tgt = self.tok_in(ys.unsqueeze(-1))
                mask = self.causal_mask(tgt.size(1), tgt.device)
                
                h = self.decoder(
                    tgt=tgt,
                    memory=memory,
                    tgt_mask=mask,
                    memory_mask=None,
                )
                
                next_scalar = self.tok_out(h[:, -1:, :]).squeeze(-1)
                ys = torch.cat([ys, next_scalar], dim=1)
            
            return ys[:, 1:]
