import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings

@dataclass
class CazzyConfig:
    """Configuration for Cazzy Aporbo model"""
    vocab_size: int = 1400  # Expanded for more conditions
    hidden_dim: int = 256  # Doubled from Delphi
    num_layers: int = 16  # Increased depth
    num_heads: int = 16  # More attention heads
    ff_dim: int = 1024  # Larger feedforward
    max_age_years: int = 120
    dropout: float = 0.1
    use_memory_bank: bool = True  # Novel feature
    use_uncertainty: bool = True  # Novel feature
    use_graph_attention: bool = True  # Novel feature
    biomarker_dim: int = 64  # For lab values
    genetic_dim: int = 128  # For PRS scores
    temporal_resolution: str = 'adaptive'  # Novel: adaptive vs fixed
    
class AdaptiveTemporalEncoding(nn.Module):
    """Novel adaptive temporal encoding that learns optimal frequencies"""
    def __init__(self, hidden_dim: int, max_age_years: int = 120):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_age_days = max_age_years * 365.25
        
        # Learnable frequency parameters
        self.freq_weights = nn.Parameter(torch.randn(hidden_dim // 2))
        self.phase_shifts = nn.Parameter(torch.zeros(hidden_dim // 2))
        self.amplitude_modulation = nn.Parameter(torch.ones(hidden_dim // 2))
        
        # Adaptive resolution network
        self.resolution_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, ages_days: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = ages_days.shape
        ages_norm = ages_days / self.max_age_days
        
        # Create frequency spectrum
        base_freqs = torch.exp(self.freq_weights) / 365.25
        base_freqs = base_freqs.unsqueeze(0).unsqueeze(0)
        
        # Compute encodings
        angles = 2 * math.pi * ages_norm.unsqueeze(-1) * base_freqs
        angles = angles + self.phase_shifts.unsqueeze(0).unsqueeze(0)
        
        sin_enc = torch.sin(angles) * self.amplitude_modulation.unsqueeze(0).unsqueeze(0)
        cos_enc = torch.cos(angles) * self.amplitude_modulation.unsqueeze(0).unsqueeze(0)
        
        # Adaptive resolution based on age
        resolution_weights = self.resolution_net(ages_norm.unsqueeze(-1))
        
        temporal_encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        temporal_encoding = temporal_encoding * resolution_weights
        
        return temporal_encoding

class DiseaseGraphAttention(nn.Module):
    """Novel graph attention for disease co-occurrence patterns"""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.graph_key = nn.Linear(hidden_dim, hidden_dim)
        self.graph_query = nn.Linear(hidden_dim, hidden_dim)
        self.graph_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Learned disease relationship matrix
        self.disease_affinity = nn.Parameter(torch.randn(1400, 1400) * 0.01)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, disease_ids: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Multi-head graph attention
        q = self.graph_query(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.graph_key(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.graph_value(x).reshape(B, T, self.num_heads, self.head_dim)
        
        # Incorporate disease affinity matrix
        if disease_ids is not None:
            affinity_weights = self.disease_affinity[disease_ids]
            affinity_weights = affinity_weights.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            affinity_weights = 0
        
        # Scaled dot-product attention with graph bias
        scores = torch.einsum('bthd,bshd->bhts', q, k) / math.sqrt(self.head_dim)
        scores = scores + affinity_weights * 0.1  # Add graph structure
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.einsum('bhts,bshd->bthd', attn_weights, v)
        
        attn_output = attn_output.reshape(B, T, C)
        return self.output_proj(attn_output)

class UncertaintyQuantification(nn.Module):
    """Novel uncertainty estimation module"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mean_head = nn.Linear(hidden_dim, hidden_dim)
        self.log_var_head = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_ensemble = nn.ModuleList([
            nn.Dropout(p=0.1 + i*0.05) for i in range(5)
        ])
        
    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(x)
        log_var = self.log_var_head(x)
        
        if training:
            # Reparameterization trick
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sampled = mean + eps * std
        else:
            # Monte Carlo dropout for uncertainty
            samples = []
            for dropout in self.dropout_ensemble:
                samples.append(dropout(mean))
            sampled = torch.stack(samples).mean(dim=0)
            uncertainty = torch.stack(samples).std(dim=0)
            
        return sampled, torch.exp(log_var)

class LongTermMemoryBank(nn.Module):
    """Novel memory mechanism for long-term dependencies"""
    def __init__(self, hidden_dim: int, memory_slots: int = 256):
        super().__init__()
        self.memory_slots = memory_slots
        self.hidden_dim = hidden_dim
        
        # Persistent memory bank
        self.memory_bank = nn.Parameter(torch.randn(memory_slots, hidden_dim) * 0.1)
        
        # Memory controller
        self.write_gate = nn.Linear(hidden_dim, memory_slots)
        self.read_gate = nn.Linear(hidden_dim, memory_slots)
        self.erase_gate = nn.Linear(hidden_dim, memory_slots)
        
    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        
        # Read from memory
        read_weights = F.softmax(self.read_gate(x), dim=-1)
        memory_read = torch.einsum('bts,sc->btc', read_weights, self.memory_bank)
        
        if update:
            # Update memory
            write_weights = torch.sigmoid(self.write_gate(x))
            erase_weights = torch.sigmoid(self.erase_gate(x))
            
            # Erase
            erase_term = torch.einsum('bts,sc->sc', erase_weights.mean(0), self.memory_bank)
            self.memory_bank.data = self.memory_bank.data * (1 - erase_term)
            
            # Write
            write_term = torch.einsum('bts,btc->sc', write_weights, x).mean(0)
            self.memory_bank.data = self.memory_bank.data + write_term
            
        return x + memory_read

class MultiModalFusion(nn.Module):
    """Novel multi-modal fusion for diverse data types"""
    def __init__(self, config: CazzyConfig):
        super().__init__()
        
        # Encoders for different modalities
        self.biomarker_encoder = nn.Sequential(
            nn.Linear(config.biomarker_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        self.genetic_encoder = nn.Sequential(
            nn.Linear(config.genetic_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, batch_first=True
        )
        
        # Gated fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, disease_emb: torch.Tensor, 
                biomarkers: Optional[torch.Tensor] = None,
                genetics: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        fused = disease_emb
        modal_features = [disease_emb]
        
        if biomarkers is not None:
            bio_features = self.biomarker_encoder(biomarkers)
            modal_features.append(bio_features)
            
        if genetics is not None:
            gen_features = self.genetic_encoder(genetics)
            modal_features.append(gen_features)
            
        if len(modal_features) > 1:
            # Cross-modal attention fusion
            combined = torch.cat(modal_features, dim=1)
            attended, _ = self.cross_attention(disease_emb, combined, combined)
            
            # Gated fusion
            gate_input = torch.cat([disease_emb, attended, combined.mean(dim=1, keepdim=True).expand_as(disease_emb)], dim=-1)
            gate = self.fusion_gate(gate_input)
            fused = gate * disease_emb + (1 - gate) * attended
            
        return fused

class CazzyAporboTransformerBlock(nn.Module):
    """Enhanced transformer block with novel features"""
    def __init__(self, config: CazzyConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, 
            dropout=config.dropout, batch_first=True
        )
        
        self.graph_attention = DiseaseGraphAttention(
            config.hidden_dim, config.num_heads
        ) if config.use_graph_attention else None
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        self.memory_bank = LongTermMemoryBank(
            config.hidden_dim
        ) if config.use_memory_bank else None
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                disease_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = residual + attn_out
        
        # Graph attention (if enabled)
        if self.graph_attention is not None:
            residual = x
            x = self.norm2(x)
            graph_out = self.graph_attention(x, disease_ids)
            x = residual + graph_out
        
        # Memory bank (if enabled)
        if self.memory_bank is not None:
            x = self.memory_bank(x)
        
        # Feedforward
        residual = x
        x = self.norm3(x)
        x = residual + self.ffn(x)
        
        return x

class CazzyAporbo(nn.Module):
    """Cazzy Aporbo: Advanced Health Trajectory Transformer"""
    
    def __init__(self, config: CazzyConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings with weight tying
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Novel adaptive temporal encoding
        self.temporal_encoding = AdaptiveTemporalEncoding(
            config.hidden_dim, config.max_age_years
        )
        
        # Multi-modal fusion
        self.multimodal_fusion = MultiModalFusion(config)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            CazzyAporboTransformerBlock(config) 
            for _ in range(config.num_layers)
        ])
        
        # Uncertainty quantification
        self.uncertainty = UncertaintyQuantification(
            config.hidden_dim
        ) if config.use_uncertainty else None
        
        # Output heads
        self.disease_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.time_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus()  # Ensures positive output
        )
        
        # Risk stratification head
        self.risk_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 5),  # 5 risk levels
            nn.Softmax(dim=-1)
        )
        
        # Survival analysis head
        self.survival_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 20)  # 20 year horizon
        )
        
        # Weight tying
        self.disease_head.weight = self.token_embedding.weight
        
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
                
    def create_causal_mask(self, seq_len: int, ages: torch.Tensor) -> torch.Tensor:
        """Enhanced causal mask that considers simultaneous events"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        
        # Mask simultaneous events
        age_diff = ages.unsqueeze(-1) - ages.unsqueeze(-2)
        simultaneous = (torch.abs(age_diff) < 1).float()  # Within 1 day
        mask = torch.maximum(mask, simultaneous)
        
        return mask.bool()
    
    def forward(self, 
                tokens: torch.Tensor,
                ages: torch.Tensor,
                biomarkers: Optional[torch.Tensor] = None,
                genetics: Optional[torch.Tensor] = None,
                return_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        
        B, T = tokens.shape
        device = tokens.device
        
        # Embeddings
        token_emb = self.token_embedding(tokens)
        temporal_emb = self.temporal_encoding(ages)
        
        x = token_emb + temporal_emb
        
        # Multi-modal fusion
        x = self.multimodal_fusion(x, biomarkers, genetics)
        
        # Create causal mask
        mask = self.create_causal_mask(T, ages).to(device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask, tokens)
        
        # Apply uncertainty quantification
        if self.uncertainty is not None:
            x, uncertainty = self.uncertainty(x, self.training)
        else:
            uncertainty = None
        
        # Generate outputs
        disease_logits = self.disease_head(x)
        time_to_event = self.time_head(x)
        risk_levels = self.risk_head(x)
        survival_curves = torch.sigmoid(self.survival_head(x))
        
        outputs = {
            'disease_logits': disease_logits,
            'time_to_event': time_to_event,
            'risk_stratification': risk_levels,
            'survival_curves': survival_curves
        }
        
        if return_uncertainty and uncertainty is not None:
            outputs['uncertainty'] = uncertainty
            
        return outputs
    
    def sample_trajectory(self, 
                         initial_tokens: torch.Tensor,
                         initial_ages: torch.Tensor,
                         max_age: float = 80.0,
                         temperature: float = 1.0,
                         top_k: int = 50,
                         top_p: float = 0.95) -> Tuple[List[int], List[float]]:
        """Generate synthetic health trajectory"""
        
        self.eval()
        with torch.no_grad():
            tokens = initial_tokens.clone()
            ages = initial_ages.clone()
            
            trajectory_tokens = []
            trajectory_times = []
            
            current_age = ages[-1].item()
            
            while current_age < max_age * 365.25:
                outputs = self.forward(
                    tokens.unsqueeze(0), 
                    ages.unsqueeze(0)
                )
                
                # Get next token probabilities
                logits = outputs['disease_logits'][0, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                logits = self._top_k_top_p_filtering(logits, top_k, top_p)
                probs = F.softmax(logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1).item()
                
                # Get time to next event
                time_to_next = outputs['time_to_event'][0, -1, 0].item()
                
                # Ensure minimum time between events
                time_to_next = max(time_to_next, 30)  # At least 30 days
                
                # Update trajectory
                trajectory_tokens.append(next_token)
                trajectory_times.append(current_age + time_to_next)
                
                # Update inputs for next iteration
                tokens = torch.cat([tokens, torch.tensor([next_token])])
                ages = torch.cat([ages, torch.tensor([current_age + time_to_next])])
                
                current_age += time_to_next
                
                # Limit sequence length
                if len(tokens) > 512:
                    tokens = tokens[-512:]
                    ages = ages[-512:]
        
        return trajectory_tokens, trajectory_times
    
    @staticmethod
    def _top_k_top_p_filtering(logits: torch.Tensor, 
                               top_k: int = 0, 
                               top_p: float = 0.0) -> torch.Tensor:
        """Filter logits using top-k and/or top-p (nucleus) filtering"""
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
            
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
        return logits

class CazzyLoss(nn.Module):
    """Advanced loss function with multiple objectives"""
    def __init__(self, config: CazzyConfig):
        super().__init__()
        self.config = config
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        losses = {}
        
        # Disease prediction loss (cross-entropy)
        if 'disease_targets' in targets:
            disease_loss = F.cross_entropy(
                outputs['disease_logits'].reshape(-1, self.config.vocab_size),
                targets['disease_targets'].reshape(-1),
                ignore_index=-100
            )
            losses['disease'] = disease_loss
        
        # Time-to-event loss (negative log-likelihood of exponential)
        if 'time_targets' in targets:
            rates = 1.0 / (outputs['time_to_event'] + 1e-6)
            time_loss = -torch.mean(
                torch.log(rates + 1e-6) - rates * targets['time_targets']
            )
            losses['time'] = time_loss
        
        # Risk stratification loss (ordinal regression)
        if 'risk_targets' in targets:
            risk_loss = F.cross_entropy(
                outputs['risk_stratification'].reshape(-1, 5),
                targets['risk_targets'].reshape(-1)
            )
            losses['risk'] = risk_loss
        
        # Survival loss (concordance)
        if 'survival_targets' in targets:
            survival_loss = self._concordance_loss(
                outputs['survival_curves'],
                targets['survival_targets'],
                targets.get('event_indicators')
            )
            losses['survival'] = survival_loss
        
        # Uncertainty regularization
        if 'uncertainty' in outputs:
            uncertainty_reg = torch.mean(outputs['uncertainty'])
            losses['uncertainty'] = uncertainty_reg * 0.01
        
        # Total weighted loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _concordance_loss(self, pred_survival: torch.Tensor, 
                         true_times: torch.Tensor,
                         events: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Concordance index based loss for survival analysis"""
        if events is None:
            events = torch.ones_like(true_times)
            
        # Pairwise comparisons
        n = pred_survival.shape[0]
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                if events[i] == 1 and true_times[i] < true_times[j]:
                    pairs.append((i, j))
                    
        if len(pairs) == 0:
            return torch.tensor(0.0, device=pred_survival.device)
        
        concordance = 0
        for i, j in pairs:
            # Higher risk should have lower survival
            if pred_survival[i].mean() < pred_survival[j].mean():
                concordance += 1
            elif pred_survival[i].mean() == pred_survival[j].mean():
                concordance += 0.5
                
        return 1.0 - (concordance / len(pairs))

def create_model(custom_config: Optional[Dict] = None) -> CazzyAporbo:
    """Factory function to create model with custom configuration"""
    config = CazzyConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return CazzyAporbo(config)

# Example usage and training loop
if __name__ == "__main__":
    # Create model
    model = create_model({
        'hidden_dim': 256,
        'num_layers': 16,
        'use_uncertainty': True,
        'use_memory_bank': True,
        'use_graph_attention': True
    })
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Example forward pass
    batch_size = 32
    seq_len = 100
    
    # Mock data
    tokens = torch.randint(0, 1400, (batch_size, seq_len))
    ages = torch.cumsum(torch.randint(30, 365, (batch_size, seq_len)), dim=1).float()
    
    # Forward pass
    outputs = model(tokens, ages, return_uncertainty=True)
    
    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
    
    # Example trajectory sampling
    initial_tokens = torch.tensor([0, 10, 25])  # Starting conditions
    initial_ages = torch.tensor([0., 1825., 3650.])  # Ages in days
    
    trajectory_tokens, trajectory_times = model.sample_trajectory(
        initial_tokens, initial_ages, max_age=80
    )
    
    print(f"\nGenerated trajectory length: {len(trajectory_tokens)} events")
    print(f"Final age: {trajectory_times[-1]/365.25:.1f} years")
