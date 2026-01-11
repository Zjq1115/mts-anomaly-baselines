import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SPCModule(nn.Module):
    """
    Statistical Process Control Module implementing Moving Average Moving Range (MAMR).

    - UCL (Upper Control Limit) for X chart: X̄ + 2.66R̄
    - LCL (Lower Control Limit) for X chart: X̄ - 2.66R̄
    - UCL for mR chart: 3.27R̄

    Points outside control limits are deemed "out of statistical control" (anomalous).
    """

    def __init__(self, window_size: int = 5, ucl_multiplier: float = 2.66,
                 mr_multiplier: float = 3.27):
        super().__init__()
        self.window_size = window_size
        self.ucl_multiplier = ucl_multiplier
        self.mr_multiplier = mr_multiplier

    def compute_moving_average(self, x: torch.Tensor) -> torch.Tensor:
        """Compute moving average over the sequence dimension."""
        # x: (B, T, C)
        B, T, C = x.shape
        if T < self.window_size:
            return x.mean(dim=1, keepdim=True).expand(-1, T, -1)

        # Use 1D convolution for moving average
        kernel = torch.ones(1, 1, self.window_size, device=x.device) / self.window_size
        x_padded = F.pad(x.transpose(1, 2), (self.window_size - 1, 0), mode='replicate')
        ma = F.conv1d(x_padded, kernel.expand(C, -1, -1), groups=C)
        return ma.transpose(1, 2)

    def compute_moving_range(self, x: torch.Tensor) -> torch.Tensor:
        """Compute moving range (absolute difference between consecutive points)."""
        # x: (B, T, C)
        diff = torch.abs(x[:, 1:, :] - x[:, :-1, :])
        # Pad to maintain sequence length
        diff = F.pad(diff, (0, 0, 1, 0), mode='replicate')
        return diff

    def compute_control_limits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute control limits based on MAMR.
        Returns: (center_line, upper_control_limit, lower_control_limit)
        """
        ma = self.compute_moving_average(x)
        mr = self.compute_moving_range(x)

        # Center line (X̄)
        center = ma.mean(dim=1, keepdim=True)
        # Average moving range (R̄)
        avg_mr = mr.mean(dim=1, keepdim=True)

        # Control limits
        ucl = center + self.ucl_multiplier * avg_mr
        lcl = center - self.ucl_multiplier * avg_mr

        return center, ucl, lcl

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply SPC preprocessing.

        Args:
            x: Input tensor (B, T, C)

        Returns:
            - Processed input with out-of-control points masked
            - Control mask (1 for in-control, 0 for out-of-control)
            - Moving average
            - Control limits info (center, ucl, lcl stacked)
        """
        center, ucl, lcl = self.compute_control_limits(x)

        # Identify out-of-control points
        out_of_control = (x > ucl) | (x < lcl)
        in_control_mask = ~out_of_control

        # Replace out-of-control points with interpolated values
        ma = self.compute_moving_average(x)
        x_processed = torch.where(in_control_mask, x, ma)

        control_limits = torch.stack([center.squeeze(1), ucl.squeeze(1), lcl.squeeze(1)], dim=-1)

        return x_processed, in_control_mask.float(), ma, control_limits


class StatisticalFeatureExtractor(nn.Module):
    """
    Extract statistical derivatives as described in Section IV-B of the paper.

    Features extracted:
    - Z-score: (value - mean) / std
    - Maximum value
    - Mean and standard deviation for reference
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Project statistical features to model dimension
        # Features: [z_score, max, mean, std, range] per channel
        self.stat_projection = nn.Linear(n_features * 5, d_model)

    def forward(self, x: torch.Tensor, comparison: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Extract statistical features from input and comparison dataset.

        Args:
            x: Query window (B, T, C)
            comparison: Comparison dataset representing normal behavior (B, T, C)

        Returns:
            - Statistical feature embeddings (B, d_model)
            - Dictionary of raw statistics for analysis
        """
        B, T, C = x.shape

        # Statistics for query window
        x_mean = x.mean(dim=1)  # (B, C)
        x_std = x.std(dim=1) + 1e-8  # (B, C)
        x_max = x.max(dim=1)[0]  # (B, C)
        x_range = x.max(dim=1)[0] - x.min(dim=1)[0]  # (B, C)

        # Statistics for comparison (normal baseline)
        c_mean = comparison.mean(dim=1)  # (B, C)
        c_std = comparison.std(dim=1) + 1e-8  # (B, C)

        # Z-score of query relative to comparison baseline
        z_score = (x_mean - c_mean) / c_std  # (B, C)

        # Concatenate all statistical features
        stats = torch.cat([z_score, x_max, x_mean, x_std, x_range], dim=-1)  # (B, C*5)

        # Project to model dimension
        stat_embed = self.stat_projection(stats)  # (B, d_model)

        raw_stats = {
            'z_score': z_score,
            'max': x_max,
            'mean': x_mean,
            'std': x_std,
            'range': x_range,
            'comparison_mean': c_mean,
            'comparison_std': c_std
        }

        return stat_embed, raw_stats


class TemplateEncoder(nn.Module):
    """
    Encode statistical derivatives into template-like embeddings.

    From paper Section IV-B:
    "To enable structured understanding and improved performance, we create
    a set of text templates with placeholders for statistical values."

    This module creates learnable template embeddings that are modulated
    by the statistical features, mimicking the text template injection process.
    """

    def __init__(self, n_features: int, d_model: int, n_templates: int = 4):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_templates = n_templates

        # Learnable template embeddings (mimicking text templates)
        # Templates for: z-score info, max info, normal conditions, comparison
        self.template_embeddings = nn.Parameter(torch.randn(n_templates, d_model))

        # Feature-specific modulation
        self.feature_modulation = nn.ModuleList([
            nn.Linear(n_features, d_model) for _ in range(n_templates)
        ])

        # Template fusion
        self.fusion = nn.Linear(n_templates * d_model, d_model)

    def forward(self, stats: dict) -> torch.Tensor:
        """
        Create template-based embeddings from statistical features.

        Args:
            stats: Dictionary of statistical features

        Returns:
            Template embeddings (B, d_model)
        """
        B = stats['z_score'].shape[0]

        # Modulate templates with different statistical features
        modulated_templates = []

        # Template 1: Z-score template
        z_mod = self.feature_modulation[0](stats['z_score'])
        t1 = self.template_embeddings[0].unsqueeze(0).expand(B, -1) + z_mod
        modulated_templates.append(t1)

        # Template 2: Maximum value template
        max_mod = self.feature_modulation[1](stats['max'])
        t2 = self.template_embeddings[1].unsqueeze(0).expand(B, -1) + max_mod
        modulated_templates.append(t2)

        # Template 3: Normal conditions template (from comparison)
        normal_mod = self.feature_modulation[2](stats['comparison_mean'])
        t3 = self.template_embeddings[2].unsqueeze(0).expand(B, -1) + normal_mod
        modulated_templates.append(t3)

        # Template 4: Deviation template (comparing current to normal)
        deviation = stats['mean'] - stats['comparison_mean']
        dev_mod = self.feature_modulation[3](deviation)
        t4 = self.template_embeddings[3].unsqueeze(0).expand(B, -1) + dev_mod
        modulated_templates.append(t4)

        # Fuse all templates
        combined = torch.cat(modulated_templates, dim=-1)  # (B, n_templates * d_model)
        template_embed = self.fusion(combined)  # (B, d_model)

        return template_embed


class DomainContextEncoder(nn.Module):
    """
    Encode domain-specific context and expert knowledge.

    From paper Section IV-B:
    "To facilitate collaboration with plant operators, we first construct
    a domain-specific context file, which enables the LLM to understand
    the context of our time series data."

    This module creates learnable domain context embeddings that capture
    correlations and domain-specific patterns.
    """

    def __init__(self, n_features: int, d_model: int, n_context_tokens: int = 8):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_context_tokens = n_context_tokens

        # Learnable domain context tokens
        self.context_tokens = nn.Parameter(torch.randn(n_context_tokens, d_model))

        # Feature correlation encoder (captures relationships between variables)
        self.correlation_encoder = nn.Sequential(
            nn.Linear(n_features * n_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Context fusion layer
        self.context_fusion = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor, stats: dict) -> torch.Tensor:
        """
        Generate domain context embeddings.

        Args:
            x: Input tensor (B, T, C)
            stats: Statistical features dictionary

        Returns:
            Domain context embedding (B, d_model)
        """
        B, T, C = x.shape

        # Compute feature correlations (domain knowledge about variable relationships)
        x_normalized = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
        correlation = torch.bmm(x_normalized.transpose(1, 2), x_normalized) / T  # (B, C, C)
        correlation_flat = correlation.reshape(B, -1)  # (B, C*C)

        # Encode correlations
        corr_embed = self.correlation_encoder(correlation_flat)  # (B, d_model)

        # Combine with learnable context tokens
        context_tokens = self.context_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, n_tokens, d_model)
        query = corr_embed.unsqueeze(1)  # (B, 1, d_model)

        # Attention over context tokens
        context_out, _ = self.context_fusion(query, context_tokens, context_tokens)
        context_embed = context_out.squeeze(1)  # (B, d_model)

        return context_embed


class ReasoningTransformer(nn.Module):
    """
    Transformer-based reasoning module mimicking LLM reasoning capabilities.

    From paper Section III-A:
    "An LLM's advanced reasoning and pattern recognition capabilities could
    then be leveraged to achieve high precision and efficiency."

    This module processes the template embeddings, domain context, and
    time series features to reason about anomalies.
    """

    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 4,
                 dropout: float = 0.1, dim_feedforward: int = 256):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Position encoding for sequence tokens
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, sequence_embed: torch.Tensor, stat_embed: torch.Tensor,
                template_embed: torch.Tensor, context_embed: torch.Tensor) -> torch.Tensor:
        """
        Perform reasoning over all input embeddings.

        Args:
            sequence_embed: Time series embeddings (B, T, d_model)
            stat_embed: Statistical feature embeddings (B, d_model)
            template_embed: Template embeddings (B, d_model)
            context_embed: Domain context embeddings (B, d_model)

        Returns:
            Reasoned embeddings (B, T, d_model)
        """
        B, T, D = sequence_embed.shape

        # Add special tokens: [STAT], [TEMPLATE], [CONTEXT], sequence tokens
        special_tokens = torch.stack([stat_embed, template_embed, context_embed], dim=1)  # (B, 3, d_model)

        # Concatenate all tokens
        all_tokens = torch.cat([special_tokens, sequence_embed], dim=1)  # (B, 3+T, d_model)

        # Add position encoding
        pos_enc = self.pos_encoding[:, :3 + T, :]
        all_tokens = all_tokens + pos_enc

        # Apply transformer reasoning
        reasoned = self.transformer(all_tokens)  # (B, 3+T, d_model)

        # Extract sequence part (skip special tokens)
        sequence_out = reasoned[:, 3:, :]  # (B, T, d_model)

        return self.layer_norm(sequence_out)


class AdaptiveComparisonModule(nn.Module):
    """
    Adaptive mechanism for updating the comparison dataset.

    From paper Section IV-C:
    "The next step of our proposed algorithm is to initialize datasets Ci that
    represent normal process behavior... the model then updates its understanding
    of normalcy as each new query window is ingested."

    From Section IV-F:
    "In addition to Ci constantly updating as each new query window is ingested,
    the process of re-initializing Ci is done for each new instance Q."
    """

    def __init__(self, n_features: int, seq_len: int, d_model: int,
                 memory_size: int = 10):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.memory_size = memory_size

        # Learnable memory bank for normal behavior patterns
        self.memory_bank = nn.Parameter(torch.randn(memory_size, seq_len, n_features) * 0.1)

        # Gate for determining if current window should update memory
        self.update_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Memory attention for retrieving relevant normal patterns
        self.memory_attention = nn.MultiheadAttention(n_features, num_heads=1, batch_first=True)

        # Adaptation layer
        self.adaptation_layer = nn.Sequential(
            nn.Linear(n_features * 2, n_features),
            nn.GELU(),
            nn.Linear(n_features, n_features)
        )

    def forward(self, x: torch.Tensor, reasoned_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve comparison dataset and compute adaptation weights.

        Args:
            x: Input tensor (B, T, C)
            reasoned_embed: Reasoned embeddings from transformer (B, T, d_model)

        Returns:
            - Comparison dataset (B, T, C)
            - Update gate values (B, 1) indicating if window is normal
        """
        B, T, C = x.shape

        # Expand memory bank for batch
        memory = self.memory_bank.unsqueeze(0).expand(B, -1, -1, -1)  # (B, M, T, C)
        memory = memory.reshape(B, self.memory_size * T, C)  # (B, M*T, C)

        # Query memory with current input
        x_query = x  # (B, T, C)
        attended_memory, _ = self.memory_attention(x_query, memory, memory)  # (B, T, C)

        # Combine attended memory with input for comparison
        combined = torch.cat([x, attended_memory], dim=-1)  # (B, T, 2C)
        comparison = self.adaptation_layer(combined)  # (B, T, C)

        # Compute update gate (should this window be considered normal?)
        pooled_embed = reasoned_embed.mean(dim=1)  # (B, d_model)
        update_weight = self.update_gate(pooled_embed)  # (B, 1)

        return comparison, update_weight


class TimeSeriesEmbedding(nn.Module):
    """
    Embed time series input into model dimension.

    Includes temporal encoding and feature projection.
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.feature_projection = nn.Linear(n_features, d_model)
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # Learnable temporal encoding
        self.temporal_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed time series input.

        Args:
            x: Input tensor (B, T, C)

        Returns:
            Embedded tensor (B, T, d_model)
        """
        B, T, C = x.shape

        # Project features
        embed = self.feature_projection(x)  # (B, T, d_model)

        # Add temporal convolution
        embed_conv = self.temporal_conv(embed.transpose(1, 2)).transpose(1, 2)  # (B, T, d_model)
        embed = embed + embed_conv

        # Add temporal encoding
        embed = embed + self.temporal_encoding[:, :T, :]

        return self.layer_norm(self.dropout(embed))


class ReconstructionDecoder(nn.Module):
    """
    Decode reasoned embeddings back to time series for reconstruction-based anomaly detection.
    """

    def __init__(self, d_model: int, n_features: int, dropout: float = 0.1):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode to original feature space.

        Args:
            x: Reasoned embeddings (B, T, d_model)

        Returns:
            Reconstructed time series (B, T, C)
        """
        return self.decoder(x)


class AADLLM(nn.Module):
    """
    AAD-LLM: Adaptive Anomaly Detection Using Large Language Models

    Main components:
    1. SPC Module: Statistical Process Control preprocessing
    2. Statistical Feature Extractor: Z-score, max, mean, std
    3. Template Encoder: Mimics text template injection
    4. Domain Context Encoder: Captures domain knowledge
    5. Reasoning Transformer: Mimics LLM reasoning
    6. Adaptive Comparison Module: Updates normal baseline
    7. Reconstruction Decoder: Outputs reconstruction for anomaly scoring

    Args:
        feats: Number of input features (C)
        seq_len: Sequence length (T)
        d_model: Model dimension
        n_heads: Number of attention heads
        e_layers: Number of transformer layers
        dropout: Dropout rate
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
                 e_layers: int = 4,
                 d_model: int = 64,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super(AADLLM, self).__init__()

        self.name = "AAD-LLM"
        self.lr = lr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_feats = feats
        self.enc_in = enc_in
        self.c_out = c_out
        self.e_layers = e_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_window = seq_len

        # 1. SPC Module (Section II - MAMR charts)
        self.spc_module = SPCModule(window_size=5)

        # 2. Time series embedding
        self.embedding = TimeSeriesEmbedding(enc_in, d_model, dropout)

        # 3. Statistical Feature Extractor (Section IV-B)
        self.stat_extractor = StatisticalFeatureExtractor(enc_in, d_model)

        # 4. Template Encoder (Section IV-B - text templates)
        self.template_encoder = TemplateEncoder(enc_in, d_model)

        # 5. Domain Context Encoder (Section IV-B - domain knowledge)
        self.context_encoder = DomainContextEncoder(enc_in, d_model)

        # 6. Adaptive Comparison Module (Section IV-C, IV-F)
        self.adaptive_module = AdaptiveComparisonModule(enc_in, seq_len, d_model)

        # 7. Reasoning Transformer (mimics frozen LLM - Section IV)
        self.reasoning = ReasoningTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=e_layers,
            dropout=dropout,
            dim_feedforward=d_model * 4
        )

        # 8. Reconstruction Decoder
        self.decoder = ReconstructionDecoder(d_model, c_out, dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of AAD-LLM.

        Args:
            x_enc: Input tensor (B, T, C)

        Returns:
            - Reconstructed output (B, T, C)
            - Anomaly indicator (adaptive gate values indicating normalcy)
        """
        # Store original statistics for denormalization
        means = x_enc.mean(1, keepdim=True).detach()
        stds = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)

        # Normalize input
        x_normalized = (x_enc - means) / stds
        x_normalized = torch.clamp(x_normalized, -1e6, 1e6)

        # 1. Apply SPC preprocessing (Section II - identify out-of-control points)
        x_processed, control_mask, moving_avg, control_limits = self.spc_module(x_normalized)

        # 2. Get comparison dataset from adaptive module (preliminary)
        # First, embed the sequence for initial comparison retrieval
        sequence_embed = self.embedding(x_processed)  # (B, T, d_model)

        # Initial reasoned embedding for adaptive module
        initial_reasoned = sequence_embed
        comparison, _ = self.adaptive_module(x_processed, initial_reasoned)

        # 3. Extract statistical features (Section IV-B)
        stat_embed, raw_stats = self.stat_extractor(x_processed, comparison)

        # 4. Create template embeddings (Section IV-B - text template injection)
        template_embed = self.template_encoder(raw_stats)

        # 5. Encode domain context (Section IV-B - domain knowledge)
        context_embed = self.context_encoder(x_processed, raw_stats)

        # 6. Apply reasoning transformer (mimics LLM reasoning)
        reasoned = self.reasoning(
            sequence_embed,
            stat_embed,
            template_embed,
            context_embed
        )  # (B, T, d_model)

        # 7. Get updated comparison and anomaly indicator (Section IV-E, IV-F)
        _, update_gate = self.adaptive_module(x_processed, reasoned)

        # 8. Apply layer normalization
        reasoned = self.layer_norm(reasoned)

        # 9. Decode to reconstruction
        dec_out = self.decoder(reasoned)  # (B, T, C)

        # Denormalize output
        dec_out = dec_out * stds + means

        # The update_gate can be used as anomaly indicator:
        # High gate value = likely normal (should update comparison)
        # Low gate value = likely anomalous (should not update comparison)
        anomaly_score = 1 - update_gate  # Invert so high = anomalous

        return dec_out, anomaly_score

def create_aadllm(n_features: int = 25, seq_len: int = 100, **kwargs) -> AADLLM:
    """
    Create AAD-LLM model with standard configuration.

    Args:
        n_features: Number of input features
        seq_len: Sequence length
        **kwargs: Additional arguments passed to AADLLM

    Returns:
        Configured AADLLM model
    """
    default_config = {
        'feats': n_features,
        'enc_in': n_features,
        'c_out': n_features,
        'seq_len': seq_len,
        'd_model': 64,
        'n_heads': 8,
        'e_layers': 4,
        'dropout': 0.1,
        'lr': 0.0001,
        'batch_size': 64
    }
    default_config.update(kwargs)
    return AADLLM(**default_config)

# Only For Testing
if __name__ == "__main__":
    print("Testing AAD-LLM implementation...")

    # Create model
    model = create_aadllm(n_features=25, seq_len=100)
    print(f"Model: {model.name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 64
    seq_len = 100
    n_features = 25

    x = torch.randn(batch_size, seq_len, n_features)
    print(f"\nInput shape: {x.shape}")

    reconstruction, anomaly_score = model(x)
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Anomaly score shape: {anomaly_score.shape}")

    # Verify shapes match
    assert reconstruction.shape == x.shape, "Reconstruction shape mismatch!"
    print("\n✓ All tests passed!")