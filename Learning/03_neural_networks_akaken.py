"""
Neural Networks: Where Math Meets Magic (But It's Still Just Math)
==================================================================
Author: Cazandra Aporbo (becaziam@gmail.com)
Date: January 2025

The moment I understood backpropagation, I literally couldn't sleep.
Not from confusion - from excitement. Here was calculus, the chain rule
I'd learned years ago, creating something that felt like intelligence.

My neural network journey has been humbling:
- Started thinking they were magic (they're not)
- Tried to use them for everything (terrible idea)  
- Watched them fail spectacularly (vanishing gradients are real)
- Finally understood when to use them (and when not to)

The biggest revelation: Neural networks are just function composition.
Layer after layer of simple operations, creating complexity. It's like
building a symphony from individual notes - simple units, infinite possibilities.

What Makes Neural Networks Special (and Frustrating):
    - Universal approximators (can learn any function... theoretically)
    - Feature learning (they create their own representations)
    - Composition (simple operations create complexity)
    - Black boxes (good luck explaining that prediction to your boss)

Hard-Won Production Wisdom:
    - They need LOTS of data (I mean LOTS)
    - Initialization matters more than you think (Xavier/He for life)
    - Batch norm is not optional for deep networks
    - The "magic" is really just matrix multiplication
    - When they work, they REALLY work
    - When they fail, good luck debugging
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Continuing our aesthetic journey
PALETTE = {
    "blossom": "#FFCFE7",
    "lilac": "#F6EAFE", 
    "mint": "#A8E6CF",
    "lavender": "#6B5B95",
    "fog": "#FDF3FF",
    "dusk": "#6E6E80",
    "rose": "#FFE4F1",
    "pearl": "#E8D5FF"
}


class SingleNeuron:
    """
    A single neuron - the fundamental unit.
    
    Understanding a single neuron changed everything for me.
    It's just a linear combination followed by a non-linearity.
    That's it. No magic. Yet from this simplicity, we can
    approximate any function. Still blows my mind.
    
    Personal story: I once explained a neuron to my grandmother as
    "a very opinionated calculator" - it takes inputs, has strong
    opinions (weights), and decides yes/no (activation). She got it.
    """
    
    def __init__(self, n_inputs, activation='sigmoid'):
        """
        Initialize a neuron with small random weights.
        
        The activation function is what makes it non-linear.
        Without it, stacking neurons would be pointless -
        linear functions composed are still linear.
        
        I learned this the hard way when my first "deep" network
        with no activation functions performed exactly like linear regression.
        """
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0
        self.activation = activation
        
    def forward(self, x):
        """
        The neuron's thought process, if you will.
        
        1. Weighted sum (like a biased opinion)
        2. Add bias (the neuron's predisposition)
        3. Activation (the decision - fire or not?)
        
        This simple process, repeated millions of times in parallel,
        can recognize faces, understand language, play Go. Incredible.
        """
        z = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'sigmoid':
            # Squashes to [0,1] - like probability
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'relu':
            # If positive, keep it. If negative, zero. Simple but powerful.
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            # Squashes to [-1,1] - allows negative outputs
            return np.tanh(z)
        else:
            return z  # Linear - basically no activation
            
    def explain(self):
        """
        Tell us what this neuron learned.
        
        In my early days, I'd train networks without ever looking at
        what individual neurons learned. Now I always check - you'd be
        surprised what patterns emerge.
        """
        print(f"Neuron with {len(self.weights)} inputs")
        print(f"Activation: {self.activation}")
        print(f"Weights: {self.weights}")
        print(f"Bias: {self.bias:.4f}")
        print("\nWhat this means:")
        print("- Positive weights: 'I like this feature'")
        print("- Negative weights: 'I dislike this feature'")
        print("- Large magnitude: 'I have strong opinions'")
        print("- Bias: 'My default tendency'")


class GentleNeuralNetwork(nn.Module):
    """
    A neural network that explains itself as it learns.
    
    After years of treating neural networks as black boxes,
    I started building them with interpretation in mind.
    Yes, they're still complex, but we can understand pieces.
    
    Built with PyTorch because in production, you don't
    implement backprop from scratch (I tried once, never again).
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        """
        Create a network layer by layer.
        
        Architecture choices I've learned through pain:
        - Start with 2-3 hidden layers (deeper isn't always better)
        - Hidden sizes often decrease (funnel shape)
        - Dropout 0.2-0.5 (0.2 for small networks, 0.5 for large)
        - ReLU for hidden layers (simple, effective, no vanishing gradient)
        
        Args:
            input_size: How many features we start with
            hidden_sizes: List like [64, 32, 16] - the architecture
            output_size: How many outputs (classes for classification)
            dropout_rate: Regularization (prevents memorization)
        """
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())  # ReLU changed everything in deep learning
            layers.append(nn.Dropout(dropout_rate))  # Randomly drop connections
            prev_size = hidden_size
            
        # Output layer (no activation here - that's in the loss function)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.training_history = {'loss': [], 'accuracy': []}
        
    def forward(self, x):
        """
        Forward pass - how information flows through the network.
        
        This is just function composition:
        output = fn(fn-1(...f2(f1(x))))
        
        Each layer transforms the representation, hopefully
        making it easier for the next layer to do its job.
        """
        return self.network(x)
    
    def count_parameters(self):
        """
        How complex is this network?
        
        I once trained a network with 100M parameters on 1000 samples.
        It memorized everything perfectly and generalized nothing.
        Now I always check: parameters should be << samples.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\nNetwork Complexity Analysis:")
        print("-" * 50)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Memory requirement: ~{total * 4 / 1024:.2f} KB (float32)")
        
        print("\nRule of thumb I follow:")
        if total > 100000:
            print("- This is a large network, needs lots of data")
        elif total > 10000:
            print("- This is a medium network, reasonable for most tasks")
        else:
            print("- This is a small network, good for simple patterns")
        
        return total
    
    def visualize_layers(self):
        """
        Show the architecture in a human-friendly way.
        
        I always visualize architecture before training.
        Helps catch silly mistakes like forgetting layers
        or having dimension mismatches.
        """
        print("\nNetwork Architecture Visualization:")
        print("-" * 50)
        total_params = 0
        
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                params = layer.in_features * layer.out_features + layer.out_features
                total_params += params
                
                print(f"Layer {i}: Linear({layer.in_features} → {layer.out_features})")
                print(f"         Parameters: {params:,}")
                print(f"         Purpose: Transform {layer.in_features}D → {layer.out_features}D space")
                
            elif isinstance(layer, nn.ReLU):
                print(f"Layer {i}: ReLU (Activation)")
                print(f"         Purpose: Add non-linearity (if positive, keep; else zero)")
                
            elif isinstance(layer, nn.Dropout):
                print(f"Layer {i}: Dropout({layer.p})")
                print(f"         Purpose: Randomly drop {layer.p*100}% of connections (prevent overfit)")
        
        print(f"\nTotal network parameters: {total_params:,}")


def create_spiral_dataset(n_samples=1000):
    """
    Create a dataset that's impossible for linear models.
    
    The spiral is my favorite toy dataset. It's simple to visualize
    but requires genuine non-linear decision boundaries. If your
    model can solve the spiral, it can probably handle real complexity.
    
    Personal note: I use this to test every new architecture idea.
    If it can't solve the spiral, it won't solve my real problems.
    """
    np.random.seed(42)
    
    n_classes = 3
    n_features = 2
    X = []
    y = []
    
    for class_id in range(n_classes):
        # Create spiral arms with increasing radius
        theta = np.linspace(0, 4 * np.pi, n_samples // n_classes)
        theta += (2 * np.pi / n_classes) * class_id  # Offset each arm
        
        radius = np.linspace(0.1, 2, n_samples // n_classes)
        
        # Convert to Cartesian coordinates with some noise
        x1 = radius * np.cos(theta) + np.random.normal(0, 0.1, len(theta))
        x2 = radius * np.sin(theta) + np.random.normal(0, 0.1, len(theta))
        
        X.append(np.column_stack([x1, x2]))
        y.append(np.full(len(theta), class_id))
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    return X, y


def train_network(model, train_loader, val_loader, epochs=50, learning_rate=0.01):
    """
    Train the network with patience and understanding.
    
    Training neural networks is an art. Too fast and you overshoot.
    Too slow and you never converge. Too long and you overfit.
    Too short and you underfit. It's all about balance.
    
    What I've learned:
    - Watch validation loss, not training loss
    - If loss explodes, learning rate is too high
    - If loss plateaus early, learning rate might be too low
    - Validation loss going up while training goes down = overfitting
    """
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam usually just works
    
    print("\nTraining Neural Network...")
    print("-" * 50)
    print("Watch the loss decrease - that's the network learning patterns")
    print("If accuracy stops improving, we've hit our limit\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()  # Enable dropout
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            # Zero gradients - PyTorch accumulates them otherwise
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass - the magic of backpropagation
            loss.backward()
            
            # Update weights - take a step based on gradients
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Validation phase - no gradient computation needed
        model.eval()  # Disable dropout
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # Save memory, we're not training here
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Record history
        model.training_history['loss'].append(train_loss / len(train_loader))
        model.training_history['accuracy'].append(100. * correct / total)
        
        # Progress update (every 10 epochs to not spam)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train Acc: {100.*correct/total:.2f}% | "
                  f"Val Acc: {100.*val_correct/val_total:.2f}%")
            
            # My personal check for overfitting
            train_acc = 100. * correct / total
            val_acc = 100. * val_correct / val_total
            if train_acc - val_acc > 15:
                print("         Warning: Significant overfitting detected!")
    
    print("-" * 50)
    print(f"Training complete!")
    print(f"Final training accuracy: {100.*correct/total:.2f}%")
    print(f"Final validation accuracy: {100.*val_correct/val_total:.2f}%")
    
    # Final overfitting check
    train_acc = 100. * correct / total
    val_acc = 100. * val_correct / val_total
    
    if train_acc - val_acc > 10:
        print("\nOverfitting detected! The model memorized training data.")
        print("Suggestions: Add more dropout, reduce model size, or get more data")
    else:
        print("\nGood generalization! The model learned transferable patterns.")


def visualize_decision_boundary(model, X, y, title="Neural Network Decision Boundary"):
    """
    Visualize what the network learned.
    
    This visualization changed how I think about neural networks.
    You can literally see the decision boundary morph and twist
    to separate the classes. It's like watching the network's
    understanding of the problem space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE["fog"])
    
    # Prepare grid for decision boundary
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get predictions for every point in the grid
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        Z = model(grid_tensor).argmax(dim=1).numpy()
    Z = Z.reshape(xx.shape)
    
    # Plot 1: Decision boundary
    ax1 = axes[0]
    colors = [PALETTE["mint"], PALETTE["blossom"], PALETTE["lilac"]]
    
    # Contour plot for decision regions
    contour = ax1.contourf(xx, yy, Z, alpha=0.3, 
                           colors=colors, levels=2)
    
    # Plot actual data points
    for class_id in range(3):
        mask = y == class_id
        ax1.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[class_id], s=20, alpha=0.7,
                   edgecolors='white', linewidths=0.5,
                   label=f'Class {class_id}')
    
    ax1.set_title("Decision Boundary\n(The network's understanding of the problem)", 
                  fontsize=12, color=PALETTE["lavender"])
    ax1.set_xlabel("Feature 1", fontsize=10)
    ax1.set_ylabel("Feature 2", fontsize=10)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor(PALETTE["fog"])
    
    # Plot 2: Training history
    ax2 = axes[1]
    epochs = range(1, len(model.training_history['loss']) + 1)
    
    # Loss on primary y-axis
    ax2.plot(epochs, model.training_history['loss'], 
             color=PALETTE["lavender"], linewidth=2, label='Loss')
    ax2.fill_between(epochs, model.training_history['loss'],
                     alpha=0.3, color=PALETTE["lilac"])
    
    # Accuracy on secondary y-axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, model.training_history['accuracy'],
                 color=PALETTE["mint"], linewidth=2, label='Accuracy')
    
    ax2.set_title("Learning Progress\n(Loss down, accuracy up = learning)", 
                  fontsize=12, color=PALETTE["lavender"])
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10, color=PALETTE["lavender"])
    ax2_twin.set_ylabel("Accuracy (%)", fontsize=10, color=PALETTE["mint"])
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor(PALETTE["fog"])
    
    plt.suptitle(title, fontsize=14, color=PALETTE["dusk"], y=1.02)
    plt.tight_layout()
    plt.show()


def activation_exploration():
    """
    Explore different activation functions.
    
    This visualization was my 'aha' moment with neural networks.
    The activation function is what allows networks to learn
    non-linear patterns. Without it, deep networks are pointless.
    
    Personal story: I once forgot to add activation functions
    and wondered why my 10-layer network performed like logistic
    regression. Turns out, it WAS logistic regression!
    """
    x = np.linspace(-3, 3, 100)
    
    activations = {
        'Linear': x,
        'Sigmoid': 1 / (1 + np.exp(-x)),
        'ReLU': np.maximum(0, x),
        'Tanh': np.tanh(x),
        'Leaky ReLU': np.where(x > 0, x, 0.01 * x)
    }
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.patch.set_facecolor(PALETTE["fog"])
    
    colors = [PALETTE["lavender"], PALETTE["mint"], PALETTE["blossom"], 
              PALETTE["lilac"], PALETTE["rose"]]
    
    for idx, (name, values) in enumerate(activations.items()):
        ax = axes[idx]
        ax.plot(x, values, color=colors[idx], linewidth=2)
        ax.fill_between(x, 0, values, alpha=0.3, color=colors[idx])
        ax.grid(True, alpha=0.3)
        ax.set_title(name, fontsize=10, color=PALETTE["dusk"])
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1.5, 3)
        ax.set_facecolor(PALETTE["fog"])
        
        # Mark zero - important reference point
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        
        # Add insight for each
        insights = {
            'Linear': 'No transform',
            'Sigmoid': 'Smooth, [0,1]',
            'ReLU': 'Simple, effective',
            'Tanh': 'Zero-centered',
            'Leaky ReLU': 'Avoids dead neurons'
        }
        ax.text(0.5, 0.05, insights[name], 
               transform=ax.transAxes, fontsize=8,
               ha='center', color=PALETTE["dusk"])
    
    plt.suptitle("Activation Functions: The Secret to Non-linearity", 
                 fontsize=14, color=PALETTE["lavender"], y=1.05)
    plt.tight_layout()
    plt.show()


def main():
    """
    Bringing it all together: From neurons to networks to understanding.
    
    This progression mirrors my own understanding:
    1. Single neuron (the atom)
    2. Activation functions (the non-linearity)
    3. Networks (composition of atoms)
    4. Training (finding the right weights)
    5. Visualization (seeing what was learned)
    
    Each step builds on the previous, creating something greater.
    """
    print("="*70)
    print("NEURAL NETWORKS: From Simple Units to Complex Intelligence")
    print("="*70)
    print("\nPersonal journey: It took me 6 months to truly understand")
    print("backpropagation. Then everything clicked. Neural networks")
    print("aren't magic - they're calculus and linear algebra having a party.")
    
    # Part 1: Understanding a single neuron
    print("\nPart 1: A Single Neuron")
    print("-" * 40)
    print("Let's start with one neuron - the building block...")
    neuron = SingleNeuron(n_inputs=2, activation='sigmoid')
    sample_input = np.array([1.5, -0.5])
    output = neuron.forward(sample_input)
    
    print(f"Input: {sample_input}")
    print(f"Output: {output:.4f}")
    print("\nWhat happened internally:")
    neuron.explain()
    
    # Part 2: Activation functions exploration
    print("\nPart 2: Activation Functions - The Non-linearity")
    print("-" * 40)
    print("Without these, neural networks are just linear models...")
    print("Visualizing different activation functions...")
    activation_exploration()
    
    # Part 3: Create complex dataset
    print("\nPart 3: The Spiral Challenge")
    print("-" * 40)
    print("Creating a dataset that would break linear models...")
    X, y = create_spiral_dataset(n_samples=1500)
    
    # Normalize features - neural networks are sensitive to scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Dataset: {len(X)} samples, 3 spiral classes")
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    print("\nThis spiral pattern is impossible for linear models.")
    print("But neural networks? They eat this for breakfast.")
    
    # Prepare PyTorch data
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Part 4: Build and train network
    print("\nPart 4: Building the Network")
    print("-" * 40)
    
    model = GentleNeuralNetwork(
        input_size=2,
        hidden_sizes=[64, 32, 16],  # 3 hidden layers, decreasing size
        output_size=3,  # 3 classes
        dropout_rate=0.2
    )
    
    print("Network architecture:")
    model.visualize_layers()
    model.count_parameters()
    
    # Train the network
    print("\nStarting training...")
    print("(This is where the magic happens - watch the loss decrease)")
    train_network(model, train_loader, val_loader, epochs=50, learning_rate=0.01)
    
    # Part 5: Visualize what was learned
    print("\nPart 5: Visualizing What the Network Learned")
    print("-" * 40)
    print("The moment of truth - did it learn the spiral pattern?")
    visualize_decision_boundary(model, X_test, y_test)
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS FROM MY NEURAL NETWORK JOURNEY:")
    print("-" * 70)
    print("1. Neural networks are universal function approximators")
    print("   (Given enough neurons, they can learn any pattern)")
    print("2. Depth allows learning hierarchical representations")
    print("   (Each layer builds on the previous)")
    print("3. Non-linearity is essential")
    print("   (Without it, it's just expensive linear regression)")
    print("4. Initialization matters")
    print("   (Bad init = vanishing/exploding gradients)")
    print("5. Regularization prevents memorization")
    print("   (Dropout, weight decay, early stopping)")
    print("6. The 'black box' problem is real")
    print("   (Good luck explaining that 10-layer network to your manager)")
    print("7. More parameters != better performance")
    print("   (Overparameterization can hurt generalization)")
    print("8. They need lots of data")
    print("   (Rule of thumb: 10x parameters as samples, minimum)")
    print("9. But when they work, they're incredible")
    print("   (Nothing beats a well-trained neural network on complex patterns)")
    print("="*70)
    print("\nNext step: Bringing it all together in production...")
    print("See: 04_connecting_it_all_production.py")


if __name__ == "__main__":
    main()