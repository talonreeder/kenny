"""
YELENA v2 — ModelManager
Loads all 44 ML models into RAM for in-process inference.

Architecture: EC2 in-process (NOT SageMaker endpoints)
- All models loaded at startup (~54MB, <5 seconds)
- 10-50ms total prediction latency for full ensemble
- Hot-swap via reload() method

Actual file inventory (44 files):
  Per timeframe (10 files × 4 TFs = 40):
    - xgboost_call_{tf}_v2.json, xgboost_put_{tf}_v2.json
    - transformer_call_{tf}_v2.pt, transformer_put_{tf}_v2.pt
    - cnn_call_{tf}_v2.pt, cnn_put_{tf}_v2.pt
    - rl_call_{tf}_v2.pt, rl_put_{tf}_v2.pt  (PPO Actor-Critic, 4 actions)
    - scaler_call_{tf}_v2.pkl, scaler_put_{tf}_v2.pkl  (shared across model types)

  Config files (4, in config/ dir, from 5min training, shared across TFs):
    - transformer_5min_v2_config.json  (n_features=156, d_model=128, 4 heads, 3 layers)
    - xgboost_5min_v2_config.json      (n_features=156, best params per direction)
    - rl_5min_v2_config.json            (state_dim=159, n_actions=4, hidden=256)
    - ensemble_config_5min_v2.json      (method=simple_average, threshold=0.55)

  Feature notes:
    - Base Feature Engine v2 produces 163 features
    - 7 self-referential f_htf_5min_* features dropped → 156 used by models
    - RL has 3 extra state features (position info) → 159 state_dim
"""

import os
import json
import time
import logging
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field

import xgboost as xgb
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# MODEL ARCHITECTURE DEFINITIONS
# Must match architectures used during training on Colab.
# If predictions fail or produce garbage, verify these against training notebooks.
# ============================================================================

class YelenaTransformer(nn.Module):
    """
    Custom Transformer for time-series prediction.
    Architecture from checkpoint state_dict:
      - input_proj: Linear(n_features, d_model)
      - pos_encoding: Parameter([1, 500, d_model])  ← max_seq_len=500
      - encoder: TransformerEncoder (3 layers, 4 heads)  ← named 'encoder' not 'transformer'
      - head: Sequential(LayerNorm, Linear, GELU, Dropout, Linear, Sigmoid)  ← named 'head' not 'classifier'

    n_features varies per timeframe (HTF self-ref features dropped):
      1min=163, 5min=156, 15min=149, 1hr=139
    """
    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 256, dropout: float = 0.15,
                 max_seq_len: int = 500):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.encoder(x)
        x = x[:, -1, :]  # Take last timestep
        return self.head(x)




class YelenaCNN(nn.Module):
    """
    Multi-scale 1D CNN with SE attention — matches training checkpoint exactly.
    Architecture from checkpoint state_dict:
      - 3 branches (kernel 3,5,7), each with 2 Conv1d layers + BatchNorm + ReLU
      - SE: flat Sequential(AdaptiveAvgPool1d, Flatten, Linear, ReLU, Linear, Sigmoid)
      - head: Sequential(AdaptiveAvgPool1d, Flatten, Linear, ReLU, Dropout, Linear, Sigmoid)

    n_features varies per timeframe (same as Transformer).
    """
    def __init__(self, n_features: int, hidden_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        # Each branch: 2 conv layers
        self.branch3 = nn.Sequential(
            nn.Conv1d(n_features, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(n_features, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(n_features, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )

        combined_channels = hidden_channels * 3  # 192

        # SE block as flat Sequential (matches checkpoint se.0-se.5)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(combined_channels, combined_channels // 4),  # se.2
            nn.ReLU(),                                              # se.3
            nn.Linear(combined_channels // 4, combined_channels),   # se.4
            nn.Sigmoid()                                            # se.5
        )

        # Classification head (matches checkpoint head.0-head.6)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),    # head.0
            nn.Flatten(),               # head.1
            nn.Linear(combined_channels, 128),  # head.2
            nn.ReLU(),                  # head.3
            nn.Dropout(dropout),        # head.4
            nn.Linear(128, 1),          # head.5
            nn.Sigmoid()                # head.6
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features) → transpose to (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        combined = torch.cat([b3, b5, b7], dim=1)  # (batch, 192, seq_len)

        # SE attention: channel-wise reweighting
        se_weights = self.se(combined)  # (batch, 192)
        combined = combined * se_weights.unsqueeze(-1)  # broadcast to (batch, 192, seq_len)

        return self.head(combined)


class YelenaRL(nn.Module):
    """
    PPO Actor-Critic for trade action selection.
    Per-direction model (separate call/put files).

    Config: state_dim=159, n_actions=4, hidden=256
    Actions: 0=HOLD, 1=CALL, 2=PUT, 3=EXIT

    NOTE: state_dim (159) > n_features (156) because RL training adds
    3 extra state features (e.g., position info, unrealized PnL, hold duration).
    During inference, pad the extra 3 features with 0.0 if not available.

    Input: (batch, state_dim) → Output: (batch, n_actions) action probabilities
    """
    def __init__(self, state_dim: int = 159, n_actions: int = 4, hidden_size: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # Actor head: action probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        # Critic head: state value (not used during inference but needed for loading weights)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """Returns action probabilities."""
        shared = self.shared(x)
        return self.actor(shared)

    def get_value(self, x):
        """Returns state value estimate (not used in inference)."""
        shared = self.shared(x)
        return self.critic(shared)


# ============================================================================
# PREDICTION RESULT DATA CLASSES
# ============================================================================

@dataclass
class ModelPrediction:
    """Single model prediction result."""
    model_name: str
    direction: str       # "CALL" or "PUT"
    probability: float   # 0.0 - 1.0
    confidence: float    # abs(probability - 0.5) * 2
    signal: str          # "BUY", "SELL", "HOLD"


@dataclass
class EnsemblePrediction:
    """Combined ensemble prediction."""
    direction: str              # "CALL", "PUT", or "HOLD"
    probability: float          # Simple average probability
    confidence: float           # Confidence score 0-100
    grade: str                  # "A+", "A", "B+", "B", "C"
    models_agreeing: int        # How many of 4 models agree (XGB, Transformer, CNN, RL)
    unanimous: bool             # All agree
    individual: Dict[str, ModelPrediction] = field(default_factory=dict)
    timeframe: str = ""
    symbol: str = ""
    timestamp: Optional[str] = None


@dataclass
class CrossTFPrediction:
    """Cross-timeframe agreement prediction."""
    primary_direction: str
    primary_confidence: float
    primary_grade: str
    tf_agreement: Dict[str, str]  # timeframe → direction
    agreement_count: int
    total_tfs: int
    confidence_bonus: float
    adjusted_confidence: float
    predictions: Dict[str, EnsemblePrediction] = field(default_factory=dict)


# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """
    Manages all 44 YELENA v2 ML models for in-process inference.

    Directory structure on EC2 (after S3 sync):
        /home/ubuntu/yelena/models/v2/
        ├── config/
        │   ├── transformer_5min_v2_config.json
        │   ├── xgboost_5min_v2_config.json
        │   ├── rl_5min_v2_config.json
        │   └── ensemble_config_5min_v2.json
        ├── 5min/
        │   ├── xgboost_call_5min_v2.json
        │   ├── xgboost_put_5min_v2.json
        │   ├── transformer_call_5min_v2.pt
        │   ├── transformer_put_5min_v2.pt
        │   ├── cnn_call_5min_v2.pt
        │   ├── cnn_put_5min_v2.pt
        │   ├── rl_call_5min_v2.pt
        │   ├── rl_put_5min_v2.pt
        │   ├── scaler_call_5min_v2.pkl
        │   └── scaler_put_5min_v2.pkl
        ├── 15min/ (same pattern)
        ├── 1hr/  (same pattern)
        └── 1min/ (same pattern)

    Usage:
        manager = ModelManager(model_dir="/home/ubuntu/yelena/models/v2")
        manager.load_all()
        pred = manager.predict("SPY", "15min", features, sequence)
        cross = manager.predict_cross_tf("SPY", features_by_tf, sequences_by_tf)
        manager.reload("15min")  # Hot-swap
    """

    TIMEFRAMES = ["1min", "5min", "15min", "1hr"]
    DIRECTIONS = ["call", "put"]
    # All 4 model types are used for ensemble voting
    MODEL_TYPES = ["xgboost", "transformer", "cnn", "rl"]
    # Only XGB, Transformer, CNN vote on direction; RL is supplementary
    VOTING_MODELS = ["xgboost", "transformer", "cnn"]
    THRESHOLD = 0.55

    GRADE_THRESHOLDS = {
        "A+": 85,
        "A": 75,
        "B+": 65,
        "B": 55,
        "C": 0
    }

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.config_dir = self.model_dir / "config"
        self.device = torch.device(device)

        # Storage: {timeframe: {model_key: model_object}}
        self.models: Dict[str, Dict[str, Any]] = {}
        self.scalers: Dict[str, Dict[str, Any]] = {}

        # Per-TF metadata: feature names and counts (vary by TF)
        # Each TF drops self-referential HTF features:
        #   1min=163, 5min=156, 15min=149, 1hr=139
        self.tf_metadata: Dict[str, Dict[str, Any]] = {}

        # Configs loaded from 5min config files (shared across TFs)
        self.configs: Dict[str, Any] = {}

        # Metadata
        self.loaded_timeframes: List[str] = []
        self.load_times: Dict[str, float] = {}
        self.model_count = 0
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_all(self) -> Dict[str, Any]:
        """Load all models for all timeframes. Returns load summary."""
        start = time.time()
        summary = {"timeframes": {}, "total_models": 0, "total_time_ms": 0}

        # Load shared configs first
        self._load_configs()

        for tf in self.TIMEFRAMES:
            tf_dir = self.model_dir / tf
            if not tf_dir.exists():
                logger.warning(f"Model directory not found: {tf_dir}")
                continue

            try:
                tf_start = time.time()
                self._load_timeframe(tf)
                tf_time = (time.time() - tf_start) * 1000
                self.load_times[tf] = tf_time
                self.loaded_timeframes.append(tf)

                model_count = len(self.models.get(tf, {}))
                summary["timeframes"][tf] = {
                    "models_loaded": model_count,
                    "load_time_ms": round(tf_time, 1)
                }
                summary["total_models"] += model_count
                logger.info(f"Loaded {model_count} models for {tf} in {tf_time:.1f}ms")
            except Exception as e:
                logger.error(f"Failed to load {tf}: {e}", exc_info=True)
                summary["timeframes"][tf] = {"error": str(e)}

        total_time = (time.time() - start) * 1000
        summary["total_time_ms"] = round(total_time, 1)
        self.model_count = summary["total_models"]
        self._is_loaded = self.model_count > 0

        logger.info(f"ModelManager: {self.model_count} models in {total_time:.1f}ms")
        return summary

    def _load_configs(self):
        """Load config files from config/ directory (5min configs shared across TFs)."""
        config_sources = {
            "transformer": self.config_dir / "transformer_5min_v2_config.json",
            "xgboost": self.config_dir / "xgboost_5min_v2_config.json",
            "rl": self.config_dir / "rl_5min_v2_config.json",
            "ensemble": self.config_dir / "ensemble_config_5min_v2.json",
        }

        for name, path in config_sources.items():
            if path.exists():
                self.configs[name] = json.loads(path.read_text())
                logger.debug(f"Loaded config: {name}")
            else:
                logger.warning(f"Config not found: {path}")
                self.configs[name] = {}

    def _load_timeframe(self, tf: str):
        """Load all 10 model files for a single timeframe."""
        tf_dir = self.model_dir / tf
        self.models[tf] = {}
        self.scalers[tf] = {}
        self.tf_metadata[tf] = {"n_features": None, "feature_names": None}

        # --- Scalers (shared across model types) ---
        for direction in self.DIRECTIONS:
            scaler_path = tf_dir / f"scaler_{direction}_{tf}_v2.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                    self.scalers[tf][direction] = scaler
                    # Extract feature count from scaler
                    if hasattr(scaler, "n_features_in_"):
                        self.tf_metadata[tf]["n_features"] = scaler.n_features_in_
                logger.debug(f"  Loaded scaler {direction} {tf}")

        # --- XGBoost ---
        for direction in self.DIRECTIONS:
            path = tf_dir / f"xgboost_{direction}_{tf}_v2.json"
            if path.exists():
                model = xgb.Booster()
                model.load_model(str(path))
                self.models[tf][f"xgboost_{direction}"] = model
                # Extract feature names from XGBoost model
                if model.feature_names and self.tf_metadata[tf]["feature_names"] is None:
                    self.tf_metadata[tf]["feature_names"] = model.feature_names
                logger.debug(f"  Loaded XGBoost {direction} {tf}")

        # --- Transformer ---
        # n_features varies per TF: infer from state_dict input_proj.weight shape
        transformer_cfg = self.configs.get("transformer", {})

        for direction in self.DIRECTIONS:
            path = tf_dir / f"transformer_{direction}_{tf}_v2.pt"
            if path.exists():
                # Load state dict first to infer architecture
                sd = torch.load(str(path), map_location=self.device)
                tf_n_features = sd["input_proj.weight"].shape[1]  # [d_model, n_features]
                max_seq_len = sd["pos_encoding"].shape[1]          # [1, max_seq_len, d_model]

                model = YelenaTransformer(
                    n_features=tf_n_features,
                    d_model=transformer_cfg.get("d_model", 128),
                    n_heads=transformer_cfg.get("n_heads", 4),
                    n_layers=transformer_cfg.get("n_layers", 3),
                    d_ff=transformer_cfg.get("d_ff", 256),
                    dropout=transformer_cfg.get("dropout", 0.15),
                    max_seq_len=max_seq_len
                )
                model.load_state_dict(sd)
                model.eval()
                self.models[tf][f"transformer_{direction}"] = model
                logger.debug(f"  Loaded Transformer {direction} {tf} (n_features={tf_n_features})")

        # --- CNN ---
        # Also infer n_features from state_dict branch3.0.weight shape
        for direction in self.DIRECTIONS:
            path = tf_dir / f"cnn_{direction}_{tf}_v2.pt"
            if path.exists():
                sd = torch.load(str(path), map_location=self.device)
                cnn_n_features = sd["branch3.0.weight"].shape[1]  # [out_ch, in_ch, kernel]

                model = YelenaCNN(
                    n_features=cnn_n_features,
                    hidden_channels=64,
                    dropout=0.2
                )
                model.load_state_dict(sd)
                model.eval()
                self.models[tf][f"cnn_{direction}"] = model
                logger.debug(f"  Loaded CNN {direction} {tf} (n_features={cnn_n_features})")

        # --- RL (per-direction PPO Actor-Critic) ---
        # Infer state_dim from state_dict
        for direction in self.DIRECTIONS:
            path = tf_dir / f"rl_{direction}_{tf}_v2.pt"
            if path.exists():
                sd = torch.load(str(path), map_location=self.device)
                rl_state_dim = sd["shared.0.weight"].shape[1]  # [hidden, state_dim]
                rl_n_actions = sd["actor.2.weight"].shape[0]   # [n_actions, 64]
                rl_hidden = sd["shared.0.weight"].shape[0]     # [hidden, state_dim]

                model = YelenaRL(
                    state_dim=rl_state_dim,
                    n_actions=rl_n_actions,
                    hidden_size=rl_hidden
                )
                model.load_state_dict(sd)
                model.eval()
                self.models[tf][f"rl_{direction}"] = model
                logger.debug(f"  Loaded RL {direction} {tf} (state_dim={rl_state_dim})")

    def predict(self, symbol: str, timeframe: str, features: np.ndarray,
                sequence: Optional[np.ndarray] = None,
                feature_names: Optional[List[str]] = None) -> EnsemblePrediction:
        """
        Run full ensemble prediction for a single timeframe.

        Args:
            symbol: e.g., "SPY"
            timeframe: e.g., "15min"
            features: 1D array of features (can be full 163 or TF-specific count)
            sequence: 2D array (seq_len, n_features) for Transformer/CNN
            feature_names: Feature column names (required if features > TF count)

        Returns:
            EnsemblePrediction with individual and combined results

        Feature handling:
            Each TF uses a different feature subset (1min=163, 5min=156, 15min=149, 1hr=139).
            If you pass all 163 features + feature_names, the manager auto-selects
            the right subset for the TF. If you pass the exact TF count, it's used as-is.
        """
        if timeframe not in self.models:
            raise ValueError(f"No models for {timeframe}. Available: {self.loaded_timeframes}")

        tf_models = self.models[timeframe]
        tf_scalers = self.scalers.get(timeframe, {})
        tf_meta = self.tf_metadata.get(timeframe, {})
        tf_feature_names = tf_meta.get("feature_names")
        tf_n_features = tf_meta.get("n_features")

        # --- Feature selection: map full feature set to TF-specific subset ---
        tf_features = features
        tf_sequence = sequence
        used_feature_names = feature_names

        if tf_feature_names and feature_names:
            if len(features) != len(tf_feature_names):
                # Caller sent full feature set — select only what this TF needs
                name_to_idx = {name: i for i, name in enumerate(feature_names)}
                indices = [name_to_idx[fn] for fn in tf_feature_names if fn in name_to_idx]

                if len(indices) != len(tf_feature_names):
                    missing = set(tf_feature_names) - set(feature_names)
                    raise ValueError(f"{timeframe} requires features not in input: {missing}")

                tf_features = features[indices]
                if sequence is not None:
                    tf_sequence = sequence[:, indices]
                used_feature_names = tf_feature_names

            elif feature_names != tf_feature_names:
                # Same count but different order — reorder to match model's expected order
                name_to_idx = {name: i for i, name in enumerate(feature_names)}
                indices = [name_to_idx[fn] for fn in tf_feature_names]

                tf_features = features[indices]
                if sequence is not None:
                    tf_sequence = sequence[:, indices]
                used_feature_names = tf_feature_names

        elif tf_n_features and len(features) != tf_n_features:
            raise ValueError(
                f"{timeframe} expects {tf_n_features} features, got {len(features)}. "
                f"Pass feature_names to auto-select, or pass exact count."
            )

        call_probs = {}
        put_probs = {}

        for direction in self.DIRECTIONS:
            # Get scaler for this direction (shared across model types)
            scaler = tf_scalers.get(direction)

            # --- XGBoost ---
            key = f"xgboost_{direction}"
            if key in tf_models:
                scaled = tf_features.copy()
                if scaler is not None:
                    scaled = scaler.transform(scaled.reshape(1, -1)).flatten()

                # XGBoost requires feature names if model was trained with them
                xgb_names = used_feature_names or tf_feature_names
                if xgb_names:
                    dmat = xgb.DMatrix(scaled.reshape(1, -1), feature_names=xgb_names)
                else:
                    dmat = xgb.DMatrix(scaled.reshape(1, -1))

                prob = float(tf_models[key].predict(dmat)[0])
                target = call_probs if direction == "call" else put_probs
                target["xgboost"] = prob

            # --- Transformer (needs sequence) ---
            if tf_sequence is not None:
                key = f"transformer_{direction}"
                if key in tf_models:
                    seq_input = self._scale_sequence(tf_sequence, scaler)
                    seq_tensor = torch.FloatTensor(seq_input).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        prob = float(tf_models[key](seq_tensor).item())

                    target = call_probs if direction == "call" else put_probs
                    target["transformer"] = prob

            # --- CNN (needs sequence) ---
            if tf_sequence is not None:
                key = f"cnn_{direction}"
                if key in tf_models:
                    seq_input = self._scale_sequence(tf_sequence, scaler)
                    seq_tensor = torch.FloatTensor(seq_input).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        prob = float(tf_models[key](seq_tensor).item())

                    target = call_probs if direction == "call" else put_probs
                    target["cnn"] = prob

            # --- RL (PPO Actor-Critic, uses flat features, outputs action probs) ---
            # RL state_dim may differ from feature_dim: pad with zeros if needed
            # Actions: 0=HOLD, 1=CALL, 2=PUT, 3=EXIT
            key = f"rl_{direction}"
            if key in tf_models:
                scaled = tf_features.copy()
                if scaler is not None:
                    scaled = scaler.transform(scaled.reshape(1, -1)).flatten()

                # Get state_dim from the model's first layer weight shape
                rl_state_dim = tf_models[key].shared[0].weight.shape[1]
                if len(scaled) < rl_state_dim:
                    padded = np.zeros(rl_state_dim, dtype=np.float32)
                    padded[:len(scaled)] = scaled
                    scaled = padded

                feat_tensor = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_probs = tf_models[key](feat_tensor).squeeze()
                    # Extract probability for the relevant action
                    if direction == "call":
                        prob = float(action_probs[1])  # CALL action
                    else:
                        prob = float(action_probs[2])  # PUT action

                target = call_probs if direction == "call" else put_probs
                target["rl"] = prob

        return self._build_ensemble(call_probs, put_probs, symbol, timeframe)

    def _scale_sequence(self, sequence: np.ndarray, scaler) -> np.ndarray:
        """Scale a sequence using the shared scaler."""
        if scaler is None:
            return sequence
        seq_2d = sequence.reshape(-1, sequence.shape[-1])
        seq_scaled = scaler.transform(seq_2d)
        return seq_scaled.reshape(sequence.shape)

    def _build_ensemble(self, call_probs: Dict[str, float],
                        put_probs: Dict[str, float],
                        symbol: str, timeframe: str) -> EnsemblePrediction:
        """Build ensemble from individual model outputs using simple average."""

        individual = {}

        # Build individual predictions
        for name, prob in call_probs.items():
            signal = "BUY" if prob >= self.THRESHOLD else "HOLD"
            individual[f"{name}_call"] = ModelPrediction(
                model_name=f"{name}_call", direction="CALL",
                probability=round(prob, 4),
                confidence=round(abs(prob - 0.5) * 2, 4),
                signal=signal
            )

        for name, prob in put_probs.items():
            signal = "BUY" if prob >= self.THRESHOLD else "HOLD"
            individual[f"{name}_put"] = ModelPrediction(
                model_name=f"{name}_put", direction="PUT",
                probability=round(prob, 4),
                confidence=round(abs(prob - 0.5) * 2, 4),
                signal=signal
            )

        # Simple average across voting models (XGB, Transformer, CNN)
        # RL participates in ensemble but can be weighted differently later
        avg_call = np.mean(list(call_probs.values())) if call_probs else 0.5
        avg_put = np.mean(list(put_probs.values())) if put_probs else 0.5

        # Determine direction and count agreement
        if avg_call >= self.THRESHOLD and avg_call > avg_put:
            direction = "CALL"
            probability = avg_call
            models_agreeing = sum(1 for p in call_probs.values() if p >= self.THRESHOLD)
            total_models = len(call_probs)
        elif avg_put >= self.THRESHOLD and avg_put > avg_call:
            direction = "PUT"
            probability = avg_put
            models_agreeing = sum(1 for p in put_probs.values() if p >= self.THRESHOLD)
            total_models = len(put_probs)
        else:
            direction = "HOLD"
            probability = max(avg_call, avg_put)
            models_agreeing = 0
            total_models = max(len(call_probs), len(put_probs))

        unanimous = models_agreeing == total_models and total_models > 0
        confidence = round(probability * 100, 1)

        # Grade assignment
        grade = "C"
        for g, threshold in self.GRADE_THRESHOLDS.items():
            if confidence >= threshold:
                grade = g
                break

        # Boost grade if unanimous (3+ models)
        if unanimous and models_agreeing >= 3 and grade != "A+":
            grade_order = ["C", "B", "B+", "A", "A+"]
            idx = grade_order.index(grade)
            if idx < len(grade_order) - 1:
                grade = grade_order[idx + 1]

        return EnsemblePrediction(
            direction=direction,
            probability=round(probability, 4),
            confidence=confidence,
            grade=grade,
            models_agreeing=models_agreeing,
            unanimous=unanimous,
            individual=individual,
            timeframe=timeframe,
            symbol=symbol
        )

    def predict_cross_tf(self, symbol: str,
                         features_by_tf: Dict[str, np.ndarray],
                         sequences_by_tf: Optional[Dict[str, np.ndarray]] = None,
                         feature_names: Optional[List[str]] = None,
                         primary_tf: str = "15min") -> CrossTFPrediction:
        """
        Run predictions across multiple timeframes and compute agreement bonus.

        Unanimous across all TFs: +7 pts
        3/4 agree: +4 pts
        2/4: +2 pts
        """
        predictions = {}
        tf_directions = {}

        for tf, features in features_by_tf.items():
            if tf not in self.loaded_timeframes:
                continue
            sequence = sequences_by_tf.get(tf) if sequences_by_tf else None
            pred = self.predict(symbol, tf, features, sequence, feature_names)
            predictions[tf] = pred
            tf_directions[tf] = pred.direction

        if primary_tf not in predictions:
            raise ValueError(f"Primary TF {primary_tf} not in predictions")

        primary_pred = predictions[primary_tf]
        primary_dir = primary_pred.direction

        agreement_count = sum(1 for d in tf_directions.values() if d == primary_dir)
        total_tfs = len(tf_directions)

        if agreement_count == total_tfs and total_tfs >= 3:
            bonus = 7.0
        elif agreement_count >= 3:
            bonus = 4.0
        elif agreement_count >= 2:
            bonus = 2.0
        else:
            bonus = 0.0

        adjusted_confidence = min(100.0, primary_pred.confidence + bonus)

        return CrossTFPrediction(
            primary_direction=primary_dir,
            primary_confidence=primary_pred.confidence,
            primary_grade=primary_pred.grade,
            tf_agreement=tf_directions,
            agreement_count=agreement_count,
            total_tfs=total_tfs,
            confidence_bonus=bonus,
            adjusted_confidence=adjusted_confidence,
            predictions=predictions
        )

    def reload(self, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """Hot-swap models by reloading from disk."""
        if timeframe:
            logger.info(f"Reloading {timeframe}...")
            if timeframe in self.loaded_timeframes:
                self.loaded_timeframes.remove(timeframe)
            self._load_timeframe(timeframe)
            self.loaded_timeframes.append(timeframe)
            return {"reloaded": timeframe, "models": len(self.models.get(timeframe, {}))}
        else:
            logger.info("Reloading ALL models...")
            self.loaded_timeframes = []
            self.models = {}
            self.scalers = {}
            self.configs = {}
            return self.load_all()

    def health(self) -> Dict[str, Any]:
        """Return health/status information."""
        return {
            "is_loaded": self._is_loaded,
            "total_models": self.model_count,
            "loaded_timeframes": self.loaded_timeframes,
            "load_times_ms": self.load_times,
            "models_per_tf": {
                tf: list(models.keys()) for tf, models in self.models.items()
            },
            "tf_metadata": {
                tf: {
                    "n_features": meta.get("n_features"),
                    "has_feature_names": meta.get("feature_names") is not None,
                    "feature_count_from_names": len(meta["feature_names"]) if meta.get("feature_names") else None
                } for tf, meta in self.tf_metadata.items()
            },
            "configs_loaded": list(self.configs.keys()),
            "model_dir": str(self.model_dir)
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global ModelManager instance."""
    global _manager
    if _manager is None:
        raise RuntimeError("ModelManager not initialized. Call init_model_manager() first.")
    return _manager


def init_model_manager(model_dir: str, device: str = "cpu") -> ModelManager:
    """Initialize and load the global ModelManager."""
    global _manager
    _manager = ModelManager(model_dir=model_dir, device=device)
    summary = _manager.load_all()
    logger.info(f"ModelManager initialized: {summary}")
    return _manager
