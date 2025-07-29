import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    """
    Improved Deep Neural Network for IoT botnet detection with better regularization
    and architecture following Popoola et al.'s optimal configuration.
    """
    def __init__(self, input_size, output_size, hidden_size=100, num_hidden_layers=4, dropout_rate=0.3):
        super(Net, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        self.hidden_dropouts = nn.ModuleList()
        
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_size))
            self.hidden_dropouts.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Initialize weights using He initialization (good for ReLU)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Validate input
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Input layer with batch norm and dropout
        x = self.input_layer(x)
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.input_bn(x)
        x = F.relu(x)
        x = self.input_dropout(x)
        
        # Hidden layers
        for i in range(self.num_hidden_layers):
            x = self.hidden_layers[i](x)
            if x.size(0) > 1:  # Only apply batch norm if batch size > 1
                x = self.hidden_bns[i](x)
            x = F.relu(x)
            x = self.hidden_dropouts[i](x)
        
        # Output layer (no activation, will be handled by loss function)
        x = self.output_layer(x)
        
        return x

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in cybersecurity datasets.
    Helps the model focus on hard examples and minority classes.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_model(model, train_loader, device, epochs=1, learning_rate=0.001, use_focal_loss=True):
    """
    Enhanced training function with better loss function and optimization.
    FIXED: Removed verbose parameter from ReduceLROnPlateau for compatibility.
    """
    model.train()
    
    # Use focal loss for better handling of imbalanced data
    if use_focal_loss:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer with weight decay for better generalization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler - FIXED: Removed verbose parameter
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Skip batch if it contains invalid data
            if torch.isnan(data).any() or torch.isinf(data).any():
                continue
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Skip if loss is invalid
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_samples += data.size(0)
        
        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples
        
        # Update learning rate
        if epoch_samples > 0:
            avg_epoch_loss = epoch_loss / epoch_samples
            scheduler.step(avg_epoch_loss)
    
    # Calculate final metrics
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    
    return avg_loss, accuracy

def test_model(model, test_loader, device, detailed_metrics=True):
    """
    Enhanced testing function with detailed metrics for cybersecurity evaluation.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    # For detailed metrics
    all_predictions = []
    all_targets = []
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Skip invalid data
            if torch.isnan(data).any() or torch.isinf(data).any():
                continue
            
            output = model(data)
            
            # Skip invalid output
            if torch.isnan(output).any() or torch.isinf(output).any():
                continue
            
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if detailed_metrics:
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
                
                # Per-class accuracy
                for i in range(len(target)):
                    label = target[i].item()
                    class_total[label] = class_total.get(label, 0) + 1
                    if pred[i].item() == label:
                        class_correct[label] = class_correct.get(label, 0) + 1
    
    # Calculate metrics
    avg_loss = test_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'total_samples': total
    }
    
    if detailed_metrics and total > 0:
        # Calculate per-class metrics
        class_accuracies = {}
        for class_id in class_total:
            class_acc = class_correct.get(class_id, 0) / class_total[class_id]
            class_accuracies[class_id] = class_acc
        
        results['class_accuracies'] = class_accuracies
        results['predictions'] = np.array(all_predictions)
        results['targets'] = np.array(all_targets)
    
    return results

def calculate_zero_day_metrics(predictions, targets, missing_attack_class_id):
    """
    Calculate specific metrics for zero-day attack detection.
    """
    metrics = {}
    
    # Overall metrics
    accuracy = np.mean(predictions == targets)
    metrics['overall_accuracy'] = accuracy
    
    # Zero-day specific metrics
    zero_day_mask = targets == missing_attack_class_id
    if np.any(zero_day_mask):
        zero_day_targets = targets[zero_day_mask]
        zero_day_predictions = predictions[zero_day_mask]
        
        # Zero-day detection accuracy
        zero_day_accuracy = np.mean(zero_day_predictions == zero_day_targets)
        metrics['zero_day_accuracy'] = zero_day_accuracy
        
        # False positive rate for zero-day class
        non_zero_day_mask = targets != missing_attack_class_id
        if np.any(non_zero_day_mask):
            non_zero_day_predictions = predictions[non_zero_day_mask]
            false_positives = np.sum(non_zero_day_predictions == missing_attack_class_id)
            total_non_zero_day = len(non_zero_day_predictions)
            fp_rate = false_positives / total_non_zero_day if total_non_zero_day > 0 else 0
            metrics['zero_day_fp_rate'] = fp_rate
        
        # Detection rate (recall for zero-day class)
        metrics['zero_day_detection_rate'] = zero_day_accuracy
        
        # Number of zero-day samples
        metrics['zero_day_samples'] = len(zero_day_targets)
        
        # Precision for zero-day class
        zero_day_predicted = predictions == missing_attack_class_id
        if np.any(zero_day_predicted):
            true_positives = np.sum((predictions == missing_attack_class_id) & (targets == missing_attack_class_id))
            precision = true_positives / np.sum(zero_day_predicted)
            metrics['zero_day_precision'] = precision
        else:
            metrics['zero_day_precision'] = 0.0
        
        # F1 score for zero-day class
        if metrics['zero_day_precision'] > 0 or metrics['zero_day_detection_rate'] > 0:
            f1_score = 2 * (metrics['zero_day_precision'] * metrics['zero_day_detection_rate']) / \
                      (metrics['zero_day_precision'] + metrics['zero_day_detection_rate'])
            metrics['zero_day_f1_score'] = f1_score
        else:
            metrics['zero_day_f1_score'] = 0.0
    else:
        metrics['zero_day_accuracy'] = 0.0
        metrics['zero_day_fp_rate'] = 0.0
        metrics['zero_day_detection_rate'] = 0.0
        metrics['zero_day_precision'] = 0.0
        metrics['zero_day_f1_score'] = 0.0
        metrics['zero_day_samples'] = 0
    
    return metrics

def get_model_complexity():
    """Return model complexity information for comparison with literature."""
    return {
        'architecture': '4-layer DNN',
        'hidden_units': 100,
        'activation': 'ReLU',
        'regularization': ['Dropout', 'BatchNorm', 'WeightDecay'],
        'loss_function': 'FocalLoss',
        'optimizer': 'AdamW',
        'initialization': 'He Normal',
        'batch_normalization': True,
        'dropout_rate': 0.3
    }

# Legacy functions for backward compatibility
def train(model, train_loader, device, epochs):
    """Legacy training function for backward compatibility."""
    avg_loss, accuracy = train_model(model, train_loader, device, epochs)
    print(f"🧪 Training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def test(model, test_loader, device):
    """Legacy testing function for backward compatibility."""
    results = test_model(model, test_loader, device, detailed_metrics=False)
    return results['loss'], results['accuracy']