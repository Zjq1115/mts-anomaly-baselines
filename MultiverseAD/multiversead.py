import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class TemporalDecomposition(nn.Module):
    """
    Temporal Decomposition using FFT to find periodic patterns.

    Equations (1)-(3):
    AM = AVG(AMP(FFT(X)))
    f = argmax(AM)
    P = floor(T/f)
    """

    def __init__(self, min_period: int = 10, max_period: int = 100):
        super().__init__()
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, x: torch.Tensor) -> int:
        """
        Find the dominant period in the time series.

        Args:
            x: Input tensor (B, T, N) or (T, N)

        Returns:
            Period P for time series decomposition
        """
        if x.dim() == 3:
            x = x.mean(dim=0)  # Average over batch

        T, N = x.shape

        # Apply FFT
        x_fft = torch.fft.rfft(x, dim=0)

        # Compute amplitude and average over features
        amplitude = torch.abs(x_fft).mean(dim=1)  # (T//2+1,)

        # Find frequency with maximum amplitude (skip DC component)
        valid_range = min(T // 2, self.max_period)
        start_idx = max(1, T // self.max_period)

        if start_idx >= valid_range:
            return self.min_period

        freq_idx = torch.argmax(amplitude[start_idx:valid_range]) + start_idx

        # Convert frequency to period
        period = max(self.min_period, min(self.max_period, T // max(1, freq_idx.item())))

        return int(period)


class SpatialTemporalCausalGraph(nn.Module):
    """
    Spatial-Temporal Causal Graph Construction.

    The STCG is a 3d×3d matrix combining:
    - B0: instantaneous causal relationships (t_n → t_n)
    - B1: lag-1 causal relationships (t_{n-1} → t_n)
    - B2: lag-2 causal relationships (t_{n-2} → t_n)
    """

    def __init__(self, n_features: int, k: int = 2):
        super().__init__()
        self.n_features = n_features
        self.k = k  # Number of time lags
        self.graph_size = (k + 1) * n_features  # 3d for k=2

        # Learnable causal weight matrices (approximating VAR-LiNGAM)
        # B0: t_n → t_n (instantaneous)
        self.B0 = nn.Parameter(torch.randn(n_features, n_features) * 0.01)
        # B1: t_{n-1} → t_n (lag 1)
        self.B1 = nn.Parameter(torch.randn(n_features, n_features) * 0.01)
        # B2: t_{n-2} → t_n (lag 2)
        self.B2 = nn.Parameter(torch.randn(n_features, n_features) * 0.01)

        # Sparsity mask (to enforce DAG structure)
        self.register_buffer('temporal_mask', self._create_temporal_mask())

    def _create_temporal_mask(self) -> torch.Tensor:
        """
        Create mask to enforce temporal causality (no reverse time dependencies).

        The spatial-temporal causal graph has blank regions
        for reverse causal relationships.
        """
        d = self.n_features
        size = self.graph_size
        mask = torch.zeros(size, size)

        # Block structure (3x3 blocks of d×d):
        # [B0, B1, B2]     t_{n-2} row
        # [0,  B0, B1]     t_{n-1} row
        # [0,  0,  B0]     t_n row

        # Upper triangular block structure (causal direction: past → future)
        # Row 0 (t_{n-2}): can affect t_{n-2}, t_{n-1}, t_n
        mask[0:d, 0:d] = 1  # B0
        mask[0:d, d:2 * d] = 1  # B1
        mask[0:d, 2 * d:3 * d] = 1  # B2

        # Row 1 (t_{n-1}): can affect t_{n-1}, t_n
        mask[d:2 * d, d:2 * d] = 1  # B0
        mask[d:2 * d, 2 * d:3 * d] = 1  # B1

        # Row 2 (t_n): can only affect t_n
        mask[2 * d:3 * d, 2 * d:3 * d] = 1  # B0

        return mask

    def forward(self) -> torch.Tensor:
        """
        Construct the spatial-temporal causal graph.

        Returns:
            A_cst: Spatial-temporal causal graph (3d, 3d)
        """
        d = self.n_features
        size = self.graph_size

        # Initialize causal graph
        A_cst = torch.zeros(size, size, device=self.B0.device)

        # Fill in the block structure according to Figure 6
        # Row 0 (t_{n-2} nodes)
        A_cst[0:d, 0:d] = self.B0
        A_cst[0:d, d:2 * d] = self.B1
        A_cst[0:d, 2 * d:3 * d] = self.B2

        # Row 1 (t_{n-1} nodes)
        A_cst[d:2 * d, d:2 * d] = self.B0
        A_cst[d:2 * d, 2 * d:3 * d] = self.B1

        # Row 2 (t_n nodes)
        A_cst[2 * d:3 * d, 2 * d:3 * d] = self.B0

        # Apply temporal mask
        A_cst = A_cst * self.temporal_mask

        return A_cst


class SpatialTemporalSynchronousAttention(nn.Module):
    """
    Spatial-Temporal Synchronous Attention Module (STSAM).

    Key operations:
    1. Vanilla graph attention on 3d nodes
    2. Fusion with spatial-temporal causal graph
    3. Multi-head attention aggregation
    """

    def __init__(self, n_features: int, d_model: int, n_heads: int = 4,
                 alpha: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.alpha = alpha  # Balance between vanilla attention and causal attention
        self.graph_size = 3 * n_features  # 3 time steps

        # Linear transformations for attention
        self.W = nn.Linear(d_model, d_model * n_heads, bias=False)

        # Attention weight vector (Equation 7)
        self.attention_w = nn.Parameter(torch.randn(n_heads, 2 * d_model))

        # 2D Convolution for causal knowledge fusion (Equation 9)
        self.causal_fusion = nn.Conv2d(
            in_channels=2,  # vanilla attention + causal graph
            out_channels=n_heads,
            kernel_size=3,
            padding=1
        )

        # GLU activation
        self.glu = nn.GLU(dim=-1)
        self.fc_glu = nn.Linear(d_model, d_model * 2)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def compute_vanilla_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute vanilla graph attention scores.

        Equations (7)-(8):
        e_{uv}^{ij} = LeakyReLU(w^T · (x_u^i ⊕ x_v^j))
        α_{uv}^{ij} = softmax(e_{uv}^{ij})

        Args:
            x: Input features (B, 3d, d_model)

        Returns:
            Attention scores (B, n_heads, 3d, 3d)
        """
        B, N, D = x.shape  # N = 3d

        # Transform features
        x_transformed = self.W(x)  # (B, N, d_model * n_heads)
        x_transformed = x_transformed.view(B, N, self.n_heads, D)  # (B, N, H, D)

        # Compute pairwise attention scores
        # For each pair of nodes, concatenate features
        attention_scores = torch.zeros(B, self.n_heads, N, N, device=x.device)

        for h in range(self.n_heads):
            x_h = x_transformed[:, :, h, :]  # (B, N, D)

            # Expand for pairwise computation
            x_i = x_h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
            x_j = x_h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)

            # Concatenate and compute attention
            x_concat = torch.cat([x_i, x_j], dim=-1)  # (B, N, N, 2D)
            e = F.leaky_relu(torch.einsum('bnmd,d->bnm', x_concat, self.attention_w[h]))

            attention_scores[:, h] = e

        return attention_scores

    def forward(self, x: torch.Tensor, causal_graph: torch.Tensor,
                prev_attention: Optional[torch.Tensor] = None,
                beta: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of STSAM.

        Args:
            x: Input features (B, 3d, d_model)
            causal_graph: Spatial-temporal causal graph (3d, 3d)
            prev_attention: Attention from previous step for skip connection
            beta: Weight for skip connection (Equation 12)

        Returns:
            - Output features (B, 3d, d_model)
            - Attention graph (B, n_heads, 3d, 3d)
        """
        B, N, D = x.shape

        # 1. Compute vanilla attention (Equations 7-8)
        A_va = self.compute_vanilla_attention(x)  # (B, H, N, N)

        # 2. Apply skip connection if available (Equation 12)
        if prev_attention is not None:
            A_eva = (1 - beta) * prev_attention + beta * A_va
        else:
            A_eva = A_va

        # 3. Fuse with causal graph (Equation 9)
        # Stack vanilla attention (averaged over heads) and causal graph
        A_va_mean = A_eva.mean(dim=1)  # (B, N, N)
        causal_expanded = causal_graph.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        # Channel-wise concatenation
        A_stacked = torch.stack([A_va_mean, causal_expanded], dim=1)  # (B, 2, N, N)

        # Apply 2D convolution
        A_st = self.causal_fusion(A_stacked)  # (B, H, N, N)

        # 4. Combine vanilla and causal attention (Equation 10)
        A = F.softmax(self.alpha * A_eva + (1 - self.alpha) * A_st, dim=-1)

        # 5. Apply attention to features (Equation 11)
        x_transformed = self.W(x).view(B, N, self.n_heads, D)  # (B, N, H, D)

        # Aggregate using attention weights
        out = torch.zeros(B, N, self.n_heads, D, device=x.device)
        for h in range(self.n_heads):
            out[:, :, h, :] = torch.bmm(A[:, h], x_transformed[:, :, h, :])

        # Average over heads
        out = out.mean(dim=2)  # (B, N, D)

        # Apply GLU activation
        out = self.fc_glu(out)
        out = self.glu(out)

        # Residual connection and layer norm
        out = self.layer_norm(out + x)

        return out, A


class SpatialTemporalSynchronousAttentionLayer(nn.Module):
    """
    Spatial-Temporal Synchronous Attention Layer (STSAL).

    From paper Section 3.6.2:
    "To capture long-term spatial-temporal dependencies throughout the entire
    time series, a sliding window mechanism is employed."

    The STSAM slides along the time axis with window size 3 and step 1.
    """

    def __init__(self, n_features: int, d_model: int, n_heads: int = 4,
                 alpha: float = 0.1, beta: float = 0.8, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.window_size = 3  # k + 1 where k = 2

        # Shared STSAM for sliding window
        self.stsam = SpatialTemporalSynchronousAttention(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            alpha=alpha,
            dropout=dropout
        )

        self.beta = beta

    def forward(self, x: torch.Tensor, causal_graph: torch.Tensor) -> torch.Tensor:
        """
        Apply sliding STSAM along the temporal dimension.

        Args:
            x: Input features (B, L, d, c) where L is number of patches
            causal_graph: Spatial-temporal causal graph (3d, 3d)

        Returns:
            Output features (B, L, d, c)
        """
        B, L, d, c = x.shape

        # Causal padding: add 2 zeros to the beginning (Section 3.6.2)
        padding = torch.zeros(B, self.window_size - 1, d, c, device=x.device)
        x_padded = torch.cat([padding, x], dim=1)  # (B, L+2, d, c)

        outputs = []
        prev_attention = None

        # Slide window along time dimension
        for i in range(L):
            # Extract window of 3 consecutive time steps
            window = x_padded[:, i:i + self.window_size, :, :]  # (B, 3, d, c)

            # Reshape for STSAM: (B, 3*d, c)
            window_flat = window.reshape(B, self.window_size * d, c)

            # Apply STSAM
            out, attention = self.stsam(
                window_flat, causal_graph,
                prev_attention=prev_attention,
                beta=self.beta
            )

            prev_attention = attention

            out_pruned = out[:, -d:, :]  # (B, d, c)
            outputs.append(out_pruned)

        # Concatenate outputs
        V_con = torch.stack(outputs, dim=1)  # (B, L, d, c)

        return V_con


class Patching(nn.Module):
    """
    Patching module to segment input time series.

    """

    def __init__(self, patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Segment input into patches.

        Args:
            x: Input tensor (B, T, d)

        Returns:
            Patched tensor (B, N_patches, d, patch_len)
        """
        B, T, d = x.shape

        # Pad the last value S times
        pad_len = self.stride
        x_padded = F.pad(x, (0, 0, 0, pad_len), mode='replicate')

        # Calculate number of patches
        N = (T - self.patch_len) // self.stride + 2

        # Extract patches
        patches = []
        for i in range(N):
            start = i * self.stride
            end = start + self.patch_len
            if end <= x_padded.shape[1]:
                patch = x_padded[:, start:end, :]  # (B, patch_len, d)
                patches.append(patch.transpose(1, 2))  # (B, d, patch_len)

        # Stack patches
        patches = torch.stack(patches, dim=1)  # (B, N, d, patch_len)

        return patches


class GRU_VAE(nn.Module):
    """
    GRU-based Variational Autoencoder for reconstruction.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 100, latent_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution."""
        _, h = self.encoder_gru(x)  # h: (1, B, hidden_dim)
        h = h.squeeze(0)  # (B, hidden_dim)

        mu = self.fc_mu(h)
        log_var = self.fc_var(h)

        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent to reconstruction."""
        h = self.fc_decode(z)  # (B, hidden_dim)
        h = h.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, hidden_dim)

        out, _ = self.decoder_gru(h)  # (B, T, hidden_dim)
        out = self.fc_out(out)  # (B, T, input_dim)

        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            - Reconstruction
            - mu
            - log_var
            - Reconstruction probability
        """
        B, T, D = x.shape

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, T)

        # Compute reconstruction probability
        recon_error = F.mse_loss(recon, x, reduction='none').mean(dim=-1)  # (B, T)
        prob = torch.exp(-recon_error)  # Higher prob = lower error

        return recon, mu, log_var, prob


class ForecastingModel(nn.Module):
    """
    Forecasting model using fully connected layers.

    """

    def __init__(self, input_dim: int, hidden_dim: int = 150, seq_len: int = 100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict next time step.

        Args:
            x: Input features (B, T, d)

        Returns:
            Predictions (B, T, d)
        """
        return self.fc(x)


class MultiverseAD(nn.Module):
    """
    MultiverseAD: Spatial-Temporal Synchronous Attention Network with Causal Knowledge.

    Main components:
    1. Spatial-Temporal Causal Graph (STCG)
    2. 1D Convolution preprocessing
    3. Patching and Linear projection
    4. Spatial-Temporal Synchronous Attention Network (STSAN)
    5. GRU temporal encoder
    6. Joint forecasting and reconstruction models

    Args:
        feats: Number of input features (d)
        seq_len: Sequence length (T)
        d_model: Model dimension (c)
        n_heads: Number of attention heads
        patch_len: Patch length (L)
        stride: Stride for patching (S)
        alpha: Balance for causal attention fusion
        beta: Skip connection weight
        gamma: Balance for anomaly score
        dropout: Dropout rate
    """

    def __init__(self,
                 feats: int = 25,
                 lr: float = 0.001,
                 batch_size: int = 64,
                 seq_len: int = 100,
                 enc_in: int = 25,
                 c_out: int = 25,
                 e_layers: int = 1,
                 d_model: int = 32,
                 n_heads: int = 4,
                 patch_len: int = 16,
                 stride: int = 8,
                 alpha: float = 0.1,
                 beta: float = 0.8,
                 gamma: float = 1.0,
                 dropout: float = 0.1):
        super(MultiverseAD, self).__init__()

        self.name = "MultiverseAD"
        self.lr = lr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_feats = feats
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_window = seq_len
        self.gamma = gamma

        # 1. Spatial-Temporal Causal Graph
        self.stcg = SpatialTemporalCausalGraph(n_features=enc_in, k=2)

        # 2. Preprocessing: 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=enc_in,
            out_channels=enc_in,
            kernel_size=7,
            padding=3
        )

        # 3. Patching
        self.patching = Patching(patch_len=patch_len, stride=stride)

        # Calculate number of patches
        self.n_patches = (seq_len - patch_len) // stride + 2

        # 4. Linear projection
        self.linear_proj = nn.Linear(patch_len, d_model)

        # 5. Spatial-Temporal Synchronous Attention Layer
        self.stsal = SpatialTemporalSynchronousAttentionLayer(
            n_features=enc_in,
            d_model=d_model,
            n_heads=n_heads,
            alpha=alpha,
            beta=beta,
            dropout=dropout
        )

        # 6. Flatten and restore dimensions
        self.flatten_proj = nn.Linear(self.n_patches * d_model, seq_len)

        # 7. GRU for temporal patterns
        self.gru = nn.GRU(
            input_size=enc_in * 2,  # concatenated conv and STSAN outputs
            hidden_size=d_model * 2,
            num_layers=1,
            batch_first=True
        )

        # 8. Forecasting model
        self.forecast_model = ForecastingModel(
            input_dim=d_model * 2,
            hidden_dim=150,
            seq_len=seq_len
        )
        self.forecast_proj = nn.Linear(d_model * 2, c_out)

        # 9. Reconstruction model (GRU-VAE)
        self.recon_model = GRU_VAE(
            input_dim=d_model * 2,
            hidden_dim=100,
            latent_dim=32
        )
        self.recon_proj = nn.Linear(d_model * 2, c_out)

    def forward(self, x_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of MultiverseAD.

        Args:
            x_enc: Input tensor (B, T, C)

        Returns:
            - Reconstructed output (B, T, C)
            - Anomaly score (B, T)
        """
        B, T, C = x_enc.shape

        # Store original for loss computation
        x_original = x_enc.clone()

        # Normalize input (Equation 5)
        x_min = x_enc.min(dim=1, keepdim=True)[0]
        x_max = x_enc.max(dim=1, keepdim=True)[0]
        x_norm = (x_enc - x_min) / (x_max - x_min + 1e-8)

        # 1. Get spatial-temporal causal graph
        causal_graph = self.stcg()  # (3d, 3d)

        # 2. 1D Convolution preprocessing
        x_conv = self.conv1d(x_norm.transpose(1, 2)).transpose(1, 2)  # (B, T, C)

        # 3. Patching
        x_patches = self.patching(x_norm)  # (B, N, d, patch_len)

        # 4. Linear projection
        x_proj = self.linear_proj(x_patches)  # (B, N, d, c)

        # 5. Spatial-Temporal Synchronous Attention Layer
        V_con = self.stsal(x_proj, causal_graph)  # (B, L, d, c)

        # 6. Flatten and restore dimensions
        B, L, d, c = V_con.shape
        V_flat = V_con.transpose(2, 3).reshape(B, d, -1)  # (B, d, L*c)
        V_out = self.flatten_proj(V_flat).transpose(1, 2)  # (B, T, d)

        # 7. Concatenate conv and STSAN outputs
        combined = torch.cat([x_conv, V_out], dim=-1)  # (B, T, 2*d)

        # 8. GRU for temporal patterns
        gru_out, _ = self.gru(combined)  # (B, T, 2*d_model)

        # 9. Forecasting
        forecast_hidden = self.forecast_model(gru_out)  # (B, T, 2*d_model)
        forecast = self.forecast_proj(forecast_hidden)  # (B, T, C)

        # 10. Reconstruction
        recon_hidden, mu, log_var, prob = self.recon_model(gru_out)
        reconstruction = self.recon_proj(recon_hidden)  # (B, T, C)

        # Denormalize outputs
        forecast = forecast * (x_max - x_min) + x_min
        reconstruction = reconstruction * (x_max - x_min) + x_min

        # 11. Compute anomaly score (Equation 18)
        # score = Σ [(x - x̂)² + γ(1 - p)] / (1 + γ)
        forecast_error = (x_original - forecast) ** 2  # (B, T, C)
        recon_prob = prob.unsqueeze(-1).expand(-1, -1, C)  # (B, T, C)

        anomaly_score = (forecast_error.sum(dim=-1) + self.gamma * (1 - prob)) / (1 + self.gamma)

        # Return reconstruction as main output
        output = (forecast + reconstruction) / 2

        return output, anomaly_score

    def compute_loss(self, x: torch.Tensor, forecast: torch.Tensor,
                     mu: torch.Tensor, log_var: torch.Tensor,
                     reconstruction: torch.Tensor) -> torch.Tensor:
        """
        Compute joint loss for training.

        Loss = Loss_for + Loss_rec

        Args:
            x: Original input
            forecast: Forecasted output
            mu, log_var: VAE latent parameters
            reconstruction: Reconstructed output
        """
        # Forecasting loss (RMSE - Equation 16)
        loss_for = torch.sqrt(F.mse_loss(forecast, x))

        # Reconstruction loss (VAE loss - Equation 17)
        recon_loss = F.mse_loss(reconstruction, x)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss_rec = recon_loss + kl_loss

        return loss_for + loss_rec


def create_multiversead(n_features: int = 25, seq_len: int = 100, **kwargs) -> MultiverseAD:
    """
    Create MultiverseAD model with standard configuration.

    Args:
        n_features: Number of input features
        seq_len: Sequence length
        **kwargs: Additional arguments

    Returns:
        Configured MultiverseAD model
    """
    default_config = {
        'feats': n_features,
        'enc_in': n_features,
        'c_out': n_features,
        'seq_len': seq_len,
        'd_model': 32,
        'n_heads': 4,
        'patch_len': 16,
        'stride': 8,
        'alpha': 0.1,
        'beta': 0.8,
        'gamma': 1.0,
        'dropout': 0.1,
        'lr': 0.001,
        'batch_size': 64
    }
    default_config.update(kwargs)
    return MultiverseAD(**default_config)

# Only for testing
if __name__ == "__main__":

    # Create model
    model = create_multiversead(n_features=25, seq_len=100)
    print(f"Model: {model.name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 64
    seq_len = 100
    n_features = 25

    x = torch.randn(batch_size, seq_len, n_features)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    model.train()
    reconstruction, anomaly_score = model(x)
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Anomaly score shape: {anomaly_score.shape}")
    print(f"Anomaly score range: [{anomaly_score.min().item():.4f}, {anomaly_score.max().item():.4f}]")

    # Verify shapes match
    assert reconstruction.shape == x.shape, "Reconstruction shape mismatch!"
    assert anomaly_score.shape == (batch_size, seq_len), "Anomaly score shape mismatch!"

    # Test causal graph
    causal_graph = model.stcg()
    print(f"\nCausal graph shape: {causal_graph.shape}")
    print(f"Causal graph sparsity: {(causal_graph == 0).float().mean().item():.2%}")

    print("\n✓ All tests passed!")