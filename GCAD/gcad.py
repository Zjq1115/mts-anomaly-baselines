import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class MixerPredictorLayer(nn.Module):
    """
    Single Mixer Predictor Layer with Time Mixing and Feature Mixing.

    From paper: "The predictor consists of L stacked Mixer Predictor Layers,
    each containing interleaved temporal mixing and feature mixing MLPs."
    """

    def __init__(self, n_features: int, seq_len: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

        # Time Mixing MLP (shared across all N features)
        # Operates on the time dimension
        self.time_mixing = nn.Sequential(
            nn.Linear(seq_len, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, seq_len),
            nn.Dropout(dropout)
        )

        # Feature Mixing MLP (shared across all time steps)
        # Operates on the feature dimension
        self.feature_mixing = nn.Sequential(
            nn.Linear(n_features, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_features),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, N) where T=seq_len, N=n_features

        Returns:
            Output tensor (B, T, N)
        """
        # Time Mixing: transpose to (B, N, T), apply MLP, transpose back
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)  # (B, N, T)
        x = self.time_mixing(x)  # (B, N, T)
        x = x.transpose(1, 2)  # (B, T, N)
        x = x + residual

        # Feature Mixing: apply MLP on feature dimension
        residual = x
        x = self.norm2(x)
        x = self.feature_mixing(x)  # (B, T, N)
        x = x + residual

        return x


class MixerPredictor(nn.Module):
    """
    Mixer Predictor (Gradient Generator) for GCAD.

    Takes input X_{t-1} = {x_{t-τ}, ..., x_{t-1}} and predicts ŷ_t.
    """

    def __init__(self, n_features: int, seq_len: int, n_layers: int = 2,
                 d_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_layers = n_layers

        # Stack of Mixer Predictor Layers
        self.layers = nn.ModuleList([
            MixerPredictorLayer(n_features, seq_len, d_hidden, dropout)
            for _ in range(n_layers)
        ])

        # Skip connection projections for each layer
        self.skip_projections = nn.ModuleList([
            nn.Linear(n_features, n_features)
            for _ in range(n_layers)
        ])

        self.fc = nn.Linear(seq_len * n_features, n_features)

        # Layer norm before final projection
        self.final_norm = nn.LayerNorm(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sliding window (B, T, N) where T=τ (max time lag)

        Returns:
            Prediction ŷ_t (B, N)
        """
        B, T, N = x.shape

        # Accumulate skip connections
        skip_sum = torch.zeros(B, T, N, device=x.device)

        # Pass through Mixer layers with skip connections
        h = x
        for layer, skip_proj in zip(self.layers, self.skip_projections):
            h = layer(h)
            skip_sum = skip_sum + skip_proj(h)

        # Final output
        h = self.final_norm(skip_sum)
        h = h.reshape(B, -1)  # (B, T*N)
        output = self.fc(h)  # (B, N)

        return output


class ChannelSeparatedErrorDetector(nn.Module):
    """
    Channel-separated Error Detector.

    Equation (1): L_{t,j} = (ŷ_{t,j} - y_{t,j})²

    Computes per-channel squared error for gradient computation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: Predicted values ŷ_t (B, N)
            target: Ground truth y_t (B, N)

        Returns:
            Channel-separated loss L_t (B, N)
        """
        return (prediction - target) ** 2


class GrangerCausalityDiscovery(nn.Module):
    """
    Granger Causality Discovery from gradients.

    - Compute gradients G_{t,j} ∈ R^{N×τ} by backpropagating each channel loss
    - Quantify Granger causality as integral of gradients over time lag

    Equation (5): a_{i,j} = ∫_{t-τ}^{t-1} |∂L_{t,j}/∂x_{φ,i}| P(x_{φ,i}) dx_{φ,i}

    For simplicity, P is uniform distribution, so this becomes sum of absolute gradients.
    """

    def __init__(self, n_features: int, seq_len: int):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

    def compute_causality_matrix(self, x: torch.Tensor, channel_losses: torch.Tensor,
                                 predictor: nn.Module) -> torch.Tensor:
        """
        Compute Granger causality matrix from gradients.

        Args:
            x: Input tensor (B, T, N) with requires_grad=True
            channel_losses: Per-channel losses (B, N)
            predictor: The predictor model for gradient computation

        Returns:
            Causality matrix A (B, N, N) where A[i,j] = effect of i on j
        """
        B, T, N = x.shape
        device = x.device

        # Initialize causality matrix
        causality_matrix = torch.zeros(B, N, N, device=device)

        # For each target channel j, compute gradients w.r.t. all input channels
        for j in range(N):
            # Get loss for channel j
            loss_j = channel_losses[:, j].sum()  # Sum over batch for gradient

            # Compute gradients of loss_j w.r.t. input x
            if x.grad is not None:
                x.grad.zero_()

            # Backpropagate
            grad_j = torch.autograd.grad(
                loss_j, x,
                create_graph=False,
                retain_graph=True,
                allow_unused=True
            )[0]  # (B, T, N)

            if grad_j is not None:
                causality_effect = torch.abs(grad_j).sum(dim=1)  # (B, N)
                causality_matrix[:, :, j] = causality_effect

        return causality_matrix

    def forward(self, causality_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply the causality matrix (identity forward, used for module compatibility).
        """
        return causality_matrix


class CausalityGraphSparsification(nn.Module):
    """
    Causality Graph Sparsification.

    From paper Equation (6):
    Ã_{i,j} = max(0, A_{i,j} - A^T_{i,j}), i ≠ j
    Ã_{i,i} = A_{i,i}

    This eliminates bidirectional symmetric similarities while preserving
    unidirectional Granger causality.
    """

    def __init__(self, threshold: float = 0.01):
        super().__init__()
        self.threshold = threshold

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Apply sparsification to causality matrix.

        Args:
            A: Causality matrix (B, N, N)

        Returns:
            Sparsified causality matrix Ã (B, N, N)
        """
        B, N, _ = A.shape

        # Transpose for subtraction
        A_T = A.transpose(-2, -1)

        # Apply sparsification: max(0, A - A^T) for off-diagonal
        A_sparse = F.relu(A - A_T)

        # Preserve diagonal elements
        diag_mask = torch.eye(N, device=A.device).bool().unsqueeze(0).expand(B, -1, -1)
        A_sparse = torch.where(diag_mask, A, A_sparse)

        # Apply threshold to remove insignificant causal relationships
        A_sparse = torch.where(A_sparse > self.threshold, A_sparse, torch.zeros_like(A_sparse))

        return A_sparse


class CausalDeviationScoring(nn.Module):
    """
    Causal Deviation Scoring for anomaly detection.

    From Equations (10), (11), (12):
    - Causal deviation: Sc_i = Σ |Ã_{test,i} - Ā_norm| / (Ā_norm + ε)
    - Time pattern deviation: St_i = Σ |diag(Ã_{test,i} - Ā_norm)| / (diag(Ā_norm) + ε)
    - Final score: S = Sc + β * St
    """

    def __init__(self, n_features: int, beta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.n_features = n_features
        self.beta = beta
        self.eps = eps

        # Register buffer for normal pattern (will be set during training)
        self.register_buffer('A_norm', torch.zeros(n_features, n_features))
        self.register_buffer('norm_initialized', torch.tensor(False))

    def update_normal_pattern(self, A_samples: torch.Tensor):
        """
        Update the normal causality pattern from training samples.

        From paper Equation (8): Ā_norm = (1/n) Σ Ã_{norm,i}

        Args:
            A_samples: Sampled causality matrices from training (n, N, N)
        """
        self.A_norm = A_samples.mean(dim=0)
        self.norm_initialized = torch.tensor(True)

    def forward(self, A_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute anomaly scores based on causal pattern deviation.

        Args:
            A_test: Test causality matrix (B, N, N)

        Returns:
            - Total anomaly score S (B,)
            - Causal deviation score Sc (B,)
            - Time pattern deviation St (B,)
        """
        B, N, _ = A_test.shape

        # Use current batch mean as normal pattern if not initialized
        if not self.norm_initialized:
            A_norm = A_test.mean(dim=0)
        else:
            A_norm = self.A_norm

        # Equation (10): Causal deviation score
        deviation = torch.abs(A_test - A_norm.unsqueeze(0))
        Sc = (deviation / (A_norm.unsqueeze(0) + self.eps)).sum(dim=(-2, -1))  # (B,)

        # Equation (11): Time pattern deviation (diagonal elements)
        diag_deviation = torch.abs(
            torch.diagonal(A_test, dim1=-2, dim2=-1) -
            torch.diagonal(A_norm.unsqueeze(0).expand(B, -1, -1), dim1=-2, dim2=-1)
        )
        diag_norm = torch.diagonal(A_norm.unsqueeze(0).expand(B, -1, -1), dim1=-2, dim2=-1)
        St = (diag_deviation / (diag_norm + self.eps)).sum(dim=-1)  # (B,)

        # Equation (12): Final anomaly score
        S = Sc + self.beta * St

        return S, Sc, St


class CausalityAwareReconstructor(nn.Module):
    """
    Reconstruction module that uses causality information.

    This module reconstructs the input time series using the learned
    causal relationships to provide reconstruction-based anomaly scores.
    """

    def __init__(self, n_features: int, seq_len: int, d_hidden: int = 64):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

        # Causality-aware encoder
        self.causality_encoder = nn.Sequential(
            nn.Linear(n_features * n_features, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden)
        )

        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_features + d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_features)
        )

    def forward(self, x: torch.Tensor, causality_matrix: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct time series using causality information.

        Args:
            x: Input tensor (B, T, N)
            causality_matrix: Causality matrix (B, N, N)

        Returns:
            Reconstructed tensor (B, T, N)
        """
        B, T, N = x.shape

        # Encode causality information
        causality_flat = causality_matrix.reshape(B, -1)  # (B, N*N)
        causality_embed = self.causality_encoder(causality_flat)  # (B, d_hidden)
        causality_embed = causality_embed.unsqueeze(1).expand(-1, T, -1)  # (B, T, d_hidden)

        # Concatenate with input and decode
        combined = torch.cat([x, causality_embed], dim=-1)  # (B, T, N+d_hidden)
        reconstructed = self.decoder(combined)  # (B, T, N)

        return reconstructed


class GCAD(nn.Module):
    """
    GCAD: Granger Causality-based Anomaly Detection for Multivariate Time Series.

    Main components:
    1. Mixer Predictor (Gradient Generator)
    2. Channel-separated Error Detector
    3. Granger Causality Discovery
    4. Causality Graph Sparsification
    5. Causal Deviation Scoring
    6. Causality-aware Reconstruction (for interface compatibility)

    Args:
        feats: Number of input features (N)
        seq_len: Sequence length / max time lag (τ)
        n_layers: Number of Mixer layers
        d_hidden: Hidden dimension
        dropout: Dropout rate
        sparsity_threshold: Threshold for causality graph sparsification
        beta: Weight for time pattern deviation in anomaly score
        lr: Learning rate (stored for reference)
        batch_size: Batch size (stored for reference)
    """

    def __init__(self,
                 feats: int = 25,
                 lr: float = 0.0001,
                 batch_size: int = 64,
                 seq_len: int = 100,
                 enc_in: int = 25,
                 c_out: int = 25,
                 e_layers: int = 2,
                 d_model: int = 64,
                 dropout: float = 0.1,
                 sparsity_threshold: float = 0.01,
                 beta: float = 1.0):
        super(GCAD, self).__init__()

        self.name = "GCAD"
        self.lr = lr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_feats = feats
        self.enc_in = enc_in
        self.c_out = c_out
        self.e_layers = e_layers
        self.d_model = d_model
        self.dropout = dropout
        self.n_window = seq_len
        self.sparsity_threshold = sparsity_threshold
        self.beta = beta

        # 1. Mixer Predictor (Gradient Generator)
        self.predictor = MixerPredictor(
            n_features=enc_in,
            seq_len=seq_len,
            n_layers=e_layers,
            d_hidden=d_model,
            dropout=dropout
        )

        # 2. Channel-separated Error Detector
        self.error_detector = ChannelSeparatedErrorDetector()

        # 3. Granger Causality Discovery
        self.causality_discovery = GrangerCausalityDiscovery(
            n_features=enc_in,
            seq_len=seq_len
        )

        # 4. Causality Graph Sparsification
        self.sparsification = CausalityGraphSparsification(
            threshold=sparsity_threshold
        )

        # 5. Causal Deviation Scoring
        self.deviation_scoring = CausalDeviationScoring(
            n_features=enc_in,
            beta=beta
        )

        # 6. Causality-aware Reconstruction (for interface compatibility)
        self.reconstructor = CausalityAwareReconstructor(
            n_features=enc_in,
            seq_len=seq_len,
            d_hidden=d_model
        )

        # Buffer to store normal causality patterns
        self.register_buffer('causality_samples', None)
        self.register_buffer('n_samples', torch.tensor(0))

    def compute_causality_matrix_efficient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efficiently compute Granger causality matrix using gradient information.

        This is a more efficient implementation that computes all gradients
        in a single backward pass by using per-channel losses.

        Args:
            x: Input tensor (B, T, N)

        Returns:
            Causality matrix (B, N, N)
        """
        B, T, N = x.shape
        device = x.device

        # Enable gradient computation for input
        x_grad = x.clone().detach().requires_grad_(True)

        x_input = x_grad[:, :-1, :]  # (B, T-1, N)
        target = x[:, -1, :]  # (B, N)

        # Pad to match expected seq_len if necessary
        if x_input.shape[1] < self.seq_len:
            pad_len = self.seq_len - x_input.shape[1]
            x_input = F.pad(x_input, (0, 0, pad_len, 0), mode='replicate')
        elif x_input.shape[1] > self.seq_len:
            x_input = x_input[:, -self.seq_len:, :]

        # Get prediction
        prediction = self.predictor(x_input)  # (B, N)

        # Compute channel-separated losses
        channel_losses = self.error_detector(prediction, target)  # (B, N)

        # Initialize causality matrix
        causality_matrix = torch.zeros(B, N, N, device=device)

        # Compute gradients for each target channel
        for j in range(N):
            # Sum loss over batch for this channel
            loss_j = channel_losses[:, j].sum()

            # Compute gradient
            grad = torch.autograd.grad(
                loss_j, x_input,
                create_graph=False,
                retain_graph=True,
                allow_unused=True
            )[0]

            if grad is not None:
                # a_{i,j} = integral of |∂L_j/∂x_i| over time lag
                causality_effect = torch.abs(grad).sum(dim=1)  # (B, N)
                causality_matrix[:, :, j] = causality_effect

        return causality_matrix

    def compute_causality_matrix_fast(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast approximation of causality matrix using Jacobian computation.

        This method approximates Granger causality using the sensitivity
        of predictions to input perturbations.

        Args:
            x: Input tensor (B, T, N)

        Returns:
            Causality matrix (B, N, N)
        """
        B, T, N = x.shape
        device = x.device

        # Prepare input
        x_input = x[:, :-1, :] if T > 1 else x
        target = x[:, -1, :] if T > 1 else x[:, 0, :]

        # Pad/truncate to match seq_len
        if x_input.shape[1] < self.seq_len:
            pad_len = self.seq_len - x_input.shape[1]
            x_input = F.pad(x_input, (0, 0, pad_len, 0), mode='replicate')
        elif x_input.shape[1] > self.seq_len:
            x_input = x_input[:, -self.seq_len:, :]

        x_input = x_input.requires_grad_(True)

        # Forward pass
        prediction = self.predictor(x_input)  # (B, N)

        # Shape: (B, N_out, T, N_in)
        causality_matrix = torch.zeros(B, N, N, device=device)

        for j in range(N):
            # Gradient of j-th output w.r.t. input
            grad_outputs = torch.zeros_like(prediction)
            grad_outputs[:, j] = 1.0

            grad = torch.autograd.grad(
                prediction, x_input,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=True,
                allow_unused=True
            )[0]

            if grad is not None:
                # Sum over time dimension to get causality from each input channel
                causality_matrix[:, :, j] = torch.abs(grad).sum(dim=1)

        return causality_matrix

    def forward(self, x_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of GCAD.

        Args:
            x_enc: Input tensor (B, T, C)

        Returns:
            - Reconstructed output (B, T, C)
            - Anomaly score (B,)
        """
        B, T, C = x_enc.shape

        # Store original statistics for denormalization
        means = x_enc.mean(1, keepdim=True).detach()
        stds = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)

        # Normalize input
        x_normalized = (x_enc - means) / stds
        x_normalized = torch.clamp(x_normalized, -1e6, 1e6)

        # Compute Granger causality matrix
        if self.training:
            # During training, use efficient gradient-based method
            causality_matrix = self.compute_causality_matrix_fast(x_normalized)
        else:
            # During inference, can use the same method
            with torch.enable_grad():
                causality_matrix = self.compute_causality_matrix_fast(x_normalized)

        # Apply sparsification (Equation 6)
        causality_sparse = self.sparsification(causality_matrix)

        # Compute anomaly scores (Equations 10, 11, 12)
        anomaly_score, Sc, St = self.deviation_scoring(causality_sparse)

        # Reconstruct using causality information
        reconstruction = self.reconstructor(x_normalized, causality_sparse)

        # Denormalize output
        reconstruction = reconstruction * stds + means

        # Normalize anomaly score to [0, 1] range approximately
        anomaly_score = torch.sigmoid(anomaly_score / (C * C))

        return reconstruction, anomaly_score

    def update_normal_pattern(self, x_train: torch.Tensor, sample_ratio: float = 0.1):
        """
        Update normal causality pattern from training data.

        Sample training windows using Bernoulli distribution and compute
        mean causality matrix as typical normal pattern.

        Args:
            x_train: Training data (N_train, T, C)
            sample_ratio: Sampling probability (Bernoulli parameter p)
        """
        self.eval()
        device = x_train.device
        n_train = x_train.shape[0]

        # Sample using Bernoulli distribution
        sample_mask = torch.bernoulli(torch.full((n_train,), sample_ratio, device=device)).bool()
        x_sampled = x_train[sample_mask]

        if x_sampled.shape[0] == 0:
            # If no samples, use all data
            x_sampled = x_train

        # Compute causality matrices for sampled windows
        with torch.no_grad():
            # Process in batches
            batch_size = min(32, x_sampled.shape[0])
            causality_matrices = []

            for i in range(0, x_sampled.shape[0], batch_size):
                batch = x_sampled[i:i + batch_size]

                # Normalize
                means = batch.mean(1, keepdim=True)
                stds = torch.sqrt(torch.var(batch, dim=1, keepdim=True, unbiased=False) + 1e-5)
                batch_norm = (batch - means) / stds

                # Need gradients for causality computation
                with torch.enable_grad():
                    causality = self.compute_causality_matrix_fast(batch_norm)
                    causality_sparse = self.sparsification(causality)
                    causality_matrices.append(causality_sparse)

            # Concatenate all causality matrices
            all_causality = torch.cat(causality_matrices, dim=0)

            # Update normal pattern (Equation 8)
            self.deviation_scoring.update_normal_pattern(all_causality)

        self.train()


def create_gcad(n_features: int = 25, seq_len: int = 100, **kwargs) -> GCAD:
    """
    Create GCAD model with standard configuration.

    Args:
        n_features: Number of input features
        seq_len: Sequence length
        **kwargs: Additional arguments passed to GCAD

    Returns:
        Configured GCAD model
    """
    default_config = {
        'feats': n_features,
        'enc_in': n_features,
        'c_out': n_features,
        'seq_len': seq_len,
        'd_model': 64,
        'e_layers': 2,
        'dropout': 0.1,
        'sparsity_threshold': 0.01,
        'beta': 1.0,
        'lr': 0.0001,
        'batch_size': 64
    }
    default_config.update(kwargs)
    return GCAD(**default_config)

# Only for testing
if __name__ == "__main__":

    # Create model
    model = create_gcad(n_features=25, seq_len=100)
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
    assert anomaly_score.shape == (batch_size,), "Anomaly score shape mismatch!"

    # Test normal pattern update
    print("\nTesting normal pattern update...")
    train_data = torch.randn(100, seq_len, n_features)
    model.update_normal_pattern(train_data, sample_ratio=0.2)
    print("Normal pattern updated successfully!")

    # Test inference mode
    model.eval()
    with torch.no_grad():
        reconstruction_eval, score_eval = model(x)
    print(f"Inference reconstruction shape: {reconstruction_eval.shape}")
    print(f"Inference anomaly score range: [{score_eval.min().item():.4f}, {score_eval.max().item():.4f}]")

    print("\n✓ All tests passed!")