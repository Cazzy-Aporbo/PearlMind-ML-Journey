import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import wandb
import os
from datetime import datetime

class HealthTrajectoryDataset(Dataset):
    """Dataset for health trajectories with multi-modal data support"""
    
    def __init__(self, 
                 trajectories: List[Dict],
                 max_seq_length: int = 512,
                 include_biomarkers: bool = True,
                 include_genetics: bool = True,
                 augment: bool = True):
        
        self.trajectories = trajectories
        self.max_seq_length = max_seq_length
        self.include_biomarkers = include_biomarkers
        self.include_genetics = include_genetics
        self.augment = augment
        
        # Preprocessing for continuous features
        self.biomarker_scaler = StandardScaler()
        self.genetic_scaler = StandardScaler()
        
        if include_biomarkers:
            all_biomarkers = []
            for traj in trajectories:
                if 'biomarkers' in traj:
                    all_biomarkers.extend(traj['biomarkers'])
            if all_biomarkers:
                self.biomarker_scaler.fit(all_biomarkers)
                
        if include_genetics:
            all_genetics = []
            for traj in trajectories:
                if 'genetics' in traj:
                    all_genetics.append(traj['genetics'])
            if all_genetics:
                self.genetic_scaler.fit(all_genetics)
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        
        # Extract base trajectory
        tokens = torch.tensor(traj['disease_codes'], dtype=torch.long)
        ages = torch.tensor(traj['ages_days'], dtype=torch.float32)
        
        # Data augmentation
        if self.augment and self.training:
            tokens, ages = self._augment_trajectory(tokens, ages)
        
        # Truncate if needed
        if len(tokens) > self.max_seq_length:
            start_idx = np.random.randint(0, len(tokens) - self.max_seq_length)
            tokens = tokens[start_idx:start_idx + self.max_seq_length]
            ages = ages[start_idx:start_idx + self.max_seq_length]
        
        # Pad if needed
        seq_len = len(tokens)
        if seq_len < self.max_seq_length:
            pad_len = self.max_seq_length - seq_len
            tokens = torch.cat([tokens, torch.zeros(pad_len, dtype=torch.long)])
            ages = torch.cat([ages, torch.zeros(pad_len, dtype=torch.float32)])
        
        # Prepare targets
        disease_targets = tokens[1:].clone()
        time_targets = torch.diff(ages, prepend=torch.tensor([0.]))
        
        # Risk stratification targets (example: based on disease burden)
        risk_targets = self._compute_risk_levels(traj)
        
        # Survival targets
        survival_targets = self._compute_survival_targets(traj)
        
        sample = {
            'tokens': tokens[:-1],
            'ages': ages[:-1],
            'disease_targets': disease_targets,
            'time_targets': time_targets[1:],
            'risk_targets': risk_targets,
            'survival_targets': survival_targets,
            'seq_len': seq_len - 1
        }
        
        # Add biomarkers if available
        if self.include_biomarkers and 'biomarkers' in traj:
            biomarkers = self.biomarker_scaler.transform(traj['biomarkers'])
            sample['biomarkers'] = torch.tensor(biomarkers, dtype=torch.float32)
        
        # Add genetics if available
        if self.include_genetics and 'genetics' in traj:
            genetics = self.genetic_scaler.transform([traj['genetics']])[0]
            sample['genetics'] = torch.tensor(genetics, dtype=torch.float32)
            
        return sample
    
    def _augment_trajectory(self, tokens: torch.Tensor, ages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation techniques"""
        
        # Time jittering
        if np.random.random() < 0.3:
            noise = torch.randn_like(ages) * 7  # +/- 7 days
            ages = ages + noise
            ages = torch.maximum(ages, torch.zeros_like(ages))
            ages = torch.sort(ages)[0]  # Ensure temporal order
        
        # Token dropout (simulate missing diagnoses)
        if np.random.random() < 0.2:
            mask = torch.rand(len(tokens)) > 0.1
            tokens = tokens[mask]
            ages = ages[mask]
        
        # Add synthetic padding tokens
        if np.random.random() < 0.3 and len(tokens) > 2:
            num_pads = np.random.randint(1, min(5, len(tokens)))
            pad_positions = torch.randperm(len(tokens))[:num_pads]
            for pos in pad_positions:
                tokens = torch.cat([tokens[:pos], torch.tensor([0]), tokens[pos:]])
                ages = torch.cat([ages[:pos], ages[pos:pos+1], ages[pos:]])
        
        return tokens, ages
    
    def _compute_risk_levels(self, traj: Dict) -> torch.Tensor:
        """Compute risk stratification levels"""
        # Simple heuristic based on disease count and severity
        num_diseases = len(traj['disease_codes'])
        severity_score = sum([1 if code < 500 else 2 for code in traj['disease_codes']])
        
        # Map to 5 risk levels
        if severity_score < 5:
            risk_level = 0  # Very low
        elif severity_score < 10:
            risk_level = 1  # Low
        elif severity_score < 20:
            risk_level = 2  # Medium
        elif severity_score < 30:
            risk_level = 3  # High
        else:
            risk_level = 4  # Very high
            
        return torch.tensor([risk_level] * (len(traj['disease_codes']) - 1), dtype=torch.long)
    
    def _compute_survival_targets(self, traj: Dict) -> torch.Tensor:
        """Compute survival probability targets"""
        # Simple exponential decay based on age and disease burden
        ages_years = np.array(traj['ages_days']) / 365.25
        max_age = 100
        
        survival_probs = []
        for age in ages_years:
            # Basic Gompertz survival function
            base_survival = np.exp(-0.001 * np.exp(0.08 * age))
            
            # Adjust for disease burden
            disease_modifier = 0.95 ** len([a for a in ages_years if a <= age])
            
            # 20-year survival curve
            future_survivals = []
            for years_ahead in range(1, 21):
                future_age = age + years_ahead
                if future_age < max_age:
                    future_survival = np.exp(-0.001 * np.exp(0.08 * future_age)) * disease_modifier
                else:
                    future_survival = 0
                future_survivals.append(future_survival)
            
            survival_probs.append(future_survivals)
        
        return torch.tensor(survival_probs[:-1], dtype=torch.float32)

class CazzyTrainer:
    """Advanced training pipeline for Cazzy Aporbo model"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer with different learning rates for different parts
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Loss function
        from cazzy_aporbo_model import CazzyLoss
        self.criterion = CazzyLoss(model.config)
        
        # Gradient accumulation
        self.accumulation_steps = config.get('gradient_accumulation', 1)
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and device == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize wandb
        if config.get('use_wandb', False):
            wandb.init(
                project="cazzy-aporbo",
                config=config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_concordance': []
        }
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with parameter groups"""
        param_groups = [
            # Embeddings - lower learning rate
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'embedding' in n],
                'lr': self.config.get('embedding_lr', 1e-4)
            },
            # Temporal encoding - higher learning rate for adaptation
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'temporal' in n],
                'lr': self.config.get('temporal_lr', 5e-4)
            },
            # Rest of the model
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'embedding' not in n and 'temporal' not in n],
                'lr': self.config.get('base_lr', 3e-4)
            }
        ]
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Mixed precision training
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self._forward_pass(batch)
                    losses = self.criterion(outputs, batch)
                    loss = losses['total'] / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('max_grad_norm', 1.0)
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self._forward_pass(batch)
                losses = self.criterion(outputs, batch)
                loss = losses['total'] / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += losses['total'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'disease': f"{losses.get('disease', 0):.4f}",
                'time': f"{losses.get('time', 0):.4f}"
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False) and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': losses['total'].item(),
                    'train/disease_loss': losses.get('disease', 0),
                    'train/time_loss': losses.get('time', 0),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / num_batches
        self.metrics_history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        all_survival_preds = []
        all_survival_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self._forward_pass(batch)
                losses = self.criterion(outputs, batch)
                
                total_loss += losses['total'].item()
                num_batches += 1
                
                # Collect predictions for metrics
                disease_probs = torch.softmax(outputs['disease_logits'], dim=-1)
                all_predictions.append(disease_probs.cpu())
                all_targets.append(batch['disease_targets'].cpu())
                
                if 'survival_curves' in outputs:
                    all_survival_preds.append(outputs['survival_curves'].cpu())
                    all_survival_targets.append(batch['survival_targets'].cpu())
        
        avg_loss = total_loss / num_batches
        
        # Calculate metrics
        metrics = {
            'val_loss': avg_loss,
            'val_auc': self._calculate_auc(all_predictions, all_targets),
            'val_concordance': self._calculate_concordance(
                all_survival_preds, all_survival_targets
            ) if all_survival_preds else 0.0
        }
        
        # Update history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        return metrics
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with proper handling of optional inputs"""
        return self.model(
            tokens=batch['tokens'],
            ages=batch['ages'],
            biomarkers=batch.get('biomarkers'),
            genetics=batch.get('genetics'),
            return_uncertainty=True
        )
    
    def _calculate_auc(self, predictions: List[torch.Tensor], 
                      targets: List[torch.Tensor]) -> float:
        """Calculate average AUC across disease predictions"""
        from sklearn.metrics import roc_auc_score
        
        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        
        # Calculate AUC for each disease
        aucs = []
        for disease_idx in range(predictions.shape[-1]):
            if len(np.unique(targets == disease_idx)) > 1:  # Skip if only one class
                try:
                    auc = roc_auc_score(
                        targets == disease_idx,
                        predictions[:, :, disease_idx].reshape(-1)
                    )
                    aucs.append(auc)
                except:
                    pass
        
        return np.mean(aucs) if aucs else 0.5
    
    def _calculate_concordance(self, predictions: List[torch.Tensor],
                              targets: List[torch.Tensor]) -> float:
        """Calculate concordance index for survival predictions"""
        from lifelines.utils import concordance_index
        
        predictions = torch.cat(predictions, dim=0).mean(dim=-1).numpy()
        targets = torch.cat(targets, dim=0).mean(dim=-1).numpy()
        
        try:
            c_index = concordance_index(
                targets.reshape(-1),
                -predictions.reshape(-1)  # Higher risk = lower survival
            )
            return c_index
        except:
            return 0.5
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              save_dir: str = './checkpoints'):
        """Full training loop"""
        
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print('='*50)
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validation
            val_metrics = self.validate(val_loader)
            print(f"Validation Metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'metrics_history': self.metrics_history,
                    'config': self.config
                }
                
                torch.save(
                    checkpoint,
                    os.path.join(save_dir, 'best_model.pt')
                )
                print(f"Saved best model with val_loss: {best_val_loss:.4f}")
            
            # Regular checkpoint
            if epoch % self.config.get('save_every', 10) == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'metrics_history': self.metrics_history,
                    'config': self.config
                }
                
                torch.save(
                    checkpoint,
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                )
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    **{f'val/{k}': v for k, v in val_metrics.items()}
                })
        
        print("\nTraining Complete!")
        return self.metrics_history

def load_and_preprocess_data(data_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load and preprocess health trajectory data"""
    # This is a placeholder - implement based on your data format
    # Expected format: List of dictionaries with keys:
    # - 'disease_codes': List of ICD-10 codes
    # - 'ages_days': List of ages in days
    # - 'biomarkers': Optional list of biomarker measurements
    # - 'genetics': Optional genetic risk scores
    
    print(f"Loading data from {data_path}...")
    
    # Example synthetic data generation for demonstration
    np.random.seed(42)
    trajectories = []
    
    for i in range(1000):  # Generate 1000 synthetic trajectories
        num_events = np.random.randint(5, 50)
        
        # Generate disease codes (0-1399 range for ICD codes)
        disease_codes = np.random.randint(1, 1400, size=num_events).tolist()
        
        # Generate ages (cumulative, in days)
        age_increments = np.random.exponential(365, size=num_events)
        ages_days = np.cumsum(age_increments).tolist()
        
        # Generate biomarkers (64-dimensional)
        biomarkers = [
            np.random.randn(64).tolist() 
            for _ in range(num_events)
        ]
        
        # Generate genetics (128-dimensional polygenic risk scores)
        genetics = np.random.randn(128).tolist()
        
        trajectories.append({
            'disease_codes': disease_codes,
            'ages_days': ages_days,
            'biomarkers': biomarkers,
            'genetics': genetics
        })
    
    # Split into train and validation
    split_idx = int(0.8 * len(trajectories))
    train_data = trajectories[:split_idx]
    val_data = trajectories[split_idx:]
    
    print(f"Loaded {len(train_data)} training and {len(val_data)} validation trajectories")
    
    return train_data, val_data

def main():
    """Main training script"""
    
    # Configuration
    config = {
        'base_lr': 3e-4,
        'embedding_lr': 1e-4,
        'temporal_lr': 5e-4,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'gradient_accumulation': 2,
        'use_amp': True,
        'use_wandb': False,  # Set to True if you have wandb configured
        'save_every': 10,
        'batch_size': 32,
        'num_epochs': 100,
        'num_workers': 4,
        'max_seq_length': 256,
        'model_config': {
            'vocab_size': 1400,
            'hidden_dim': 256,
            'num_layers': 16,
            'num_heads': 16,
            'ff_dim': 1024,
            'dropout': 0.1,
            'use_memory_bank': True,
            'use_uncertainty': True,
            'use_graph_attention': True,
            'biomarker_dim': 64,
            'genetic_dim': 128
        }
    }
    
    # Load data
    train_data, val_data = load_and_preprocess_data('path/to/your/data')
    
    # Create datasets
    train_dataset = HealthTrajectoryDataset(
        train_data,
        max_seq_length=config['max_seq_length'],
        augment=True
    )
    train_dataset.training = True
    
    val_dataset = HealthTrajectoryDataset(
        val_data,
        max_seq_length=config['max_seq_length'],
        augment=False
    )
    val_dataset.training = False
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    from cazzy_aporbo_model import create_model
    model = create_model(config['model_config'])
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = CazzyTrainer(model, config)
    
    # Train
    metrics_history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['num_epochs']
    )
    
    # Save final metrics
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
