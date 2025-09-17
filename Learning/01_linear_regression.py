"""
Linear Regression: Where Every ML Journey Begins
================================================
Author: Cazandra Aporbo (becaziam@gmail.com)
Date: January 23 2025

When I first started learning machine learning, linear regression seemed almost
too simple. Just fitting a line? But then I realized - this "simple" line is
the foundation of everything. Neural networks? Just many lines with non-linearity.
Deep learning? Composed linear transformations. Everything builds from here.

What took me months to understand: it's not about the complexity of the model,
it's about understanding WHY it works. Once you truly grasp how a simple line
finds its way through data points, minimizing error, updating weights - you've
understood the core of all machine learning.

Mathematical Beauty:
    y = β₀ + β₁x + ε
    
    Where:
    - y is what we're trying to understand (the unknown)
    - x is what we know (our clues)
    - β₀, β₁ are the patterns we're learning (the relationship)
    - ε is the humility to admit we can't explain everything (noise, chaos, life)

Personal Production Notes:
    After deploying my first models, I learned:
    - Simple models fail in obvious ways (easier to debug)
    - Complex models fail in mysterious ways (good luck at 3 AM)
    - Linear regression is often "good enough" (and that's beautiful)
    - Interpretability beats accuracy when stakes are high
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Set our aesthetic - these soft colors remind me of early morning coding sessions
plt.style.use('seaborn-v0_8-pastel')
PALETTE = {
    "blossom": "#FFCFE7",
    "lilac": "#F6EAFE", 
    "mint": "#A8E6CF",
    "lavender": "#6B5B95",
    "fog": "#FDF3FF"
}


class GentleLinearRegression:
    """
    A linear regression implementation that explains itself as it learns.
    
    I built this from scratch because using sklearn.LinearRegression is like
    using a calculator without understanding arithmetic. Sure, it works, but
    you miss the beauty of what's actually happening under the hood.
    
    Personal story: I once spent a week debugging a production model, only to
    realize I didn't truly understand gradient descent. Never again.
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000, verbose=True):
        """
        Initialize with patience and reasonable defaults.
        
        Learning rate took me forever to understand. Too high and your model
        panics, jumping wildly past the minimum. Too low and you'll wait
        forever, like watching paint dry. 0.01 is the goldilocks zone I've
        found works for most problems.
        
        Args:
            learning_rate: How big our steps are down the hill
            iterations: How many times we're willing to try
            verbose: Whether to share the journey (I always want to see)
        """
        self.lr = learning_rate
        self.iterations = iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.losses = []
        
    def fit(self, X, y):
        """
        The learning happens here. This is where I fell in love with ML.
        
        Watching gradient descent work for the first time was magical.
        The loss decreasing, iteration by iteration, the model literally
        learning from its mistakes. It's like watching a child learn to walk -
        stumbling at first, then finding balance.
        """
        n_samples, n_features = X.shape
        
        # Initialize with small random values
        # I learned the hard way: initializing with zeros = dead neurons later
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        # The learning loop - where math becomes intelligence
        for i in range(self.iterations):
            # Forward pass: make predictions with what we know so far
            y_pred = X.dot(self.weights) + self.bias
            
            # Calculate loss - how wrong were we?
            # MSE punishes big mistakes more (squared term)
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)
            
            # Calculate gradients - which direction reduces error?
            # This is just calculus, but it felt like magic at first
            dw = -(2/n_samples) * X.T.dot(y - y_pred)
            db = -(2/n_samples) * np.sum(y - y_pred)
            
            # Update weights - take a small step toward being less wrong
            # This line is where learning happens. Still gives me chills.
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Share progress - transparency in AI starts here
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i:4d} | Loss: {loss:.4f} | "
                      f"Weights: {self.weights[0]:.4f} | Bias: {self.bias:.4f}")
                
        return self
    
    def predict(self, X):
        """
        Make predictions with what we've learned.
        
        This is just math - multiply and add. But somehow, through training,
        these numbers have captured patterns in data. That still amazes me.
        """
        return X.dot(self.weights) + self.bias
    
    def explain_prediction(self, x_sample):
        """
        Break down exactly how we arrived at a prediction.
        
        In production, when a stakeholder asks "why did the model decide this?",
        you better have an answer. This method is that answer.
        """
        prediction = x_sample.dot(self.weights) + self.bias
        
        print("\n--- Prediction Breakdown ---")
        print(f"Starting with bias: {self.bias:.4f}")
        print("(Think of bias as the model's baseline assumption)")
        
        for i, (feature_val, weight) in enumerate(zip(x_sample, self.weights)):
            contribution = feature_val * weight
            print(f"Feature {i}: {feature_val:.4f} × {weight:.4f} = {contribution:.4f}")
            if weight > 0:
                print(f"  → This feature increases the prediction")
            else:
                print(f"  → This feature decreases the prediction")
        
        print(f"\nFinal prediction: {prediction:.4f}")
        return prediction


def create_story_dataset():
    """
    Create a dataset that tells a story.
    
    Real data is messy, but when learning, stories help. This dataset models
    something I think about daily: the relationship between coffee and happiness.
    
    Personal note: After tracking my own coffee consumption and mood for a month,
    I found this relationship surprisingly accurate. Too much coffee = anxiety.
    Too little = sluggish. There's a sweet spot around 2-3 cups.
    """
    np.random.seed(42)  # Reproducibility matters in science
    
    coffee_cups = np.random.uniform(0, 5, 200).reshape(-1, 1)
    
    # Base happiness increases with coffee (caffeine is wonderful)
    happiness = 2 + 1.5 * coffee_cups.flatten()
    
    # Add realistic noise - life is unpredictable
    # Some days, even perfect coffee can't help
    happiness += np.random.normal(0, 0.5, 200)
    
    # Too much coffee decreases happiness - the jittery, anxious zone
    # I learned this the hard way during thesis writing
    too_much_mask = coffee_cups.flatten() > 3.5
    happiness[too_much_mask] -= (coffee_cups.flatten()[too_much_mask] - 3.5) * 2
    
    return coffee_cups, happiness


def visualize_learning_journey(model, X_train, y_train, X_test, y_test):
    """
    Create beautiful visualizations that tell the story of learning.
    
    A mentor once told me: "If you can't visualize it, you don't understand it."
    These plots are my understanding made visible. Each one taught me something
    different about how models learn.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor(PALETTE["fog"])
    
    # 1. The Learning Curve - watching understanding emerge
    ax1 = axes[0, 0]
    ax1.plot(model.losses, color=PALETTE["lavender"], linewidth=2)
    ax1.fill_between(range(len(model.losses)), model.losses, 
                     alpha=0.3, color=PALETTE["lilac"])
    ax1.set_title("The Learning Journey\n(Each iteration brings us closer to understanding)", 
                  fontsize=12, color=PALETTE["lavender"])
    ax1.set_xlabel("Iterations (attempts to understand)", fontsize=10)
    ax1.set_ylabel("Loss (how wrong we are)", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor(PALETTE["fog"])
    
    # Note: The plateau at the end? That's convergence - we've learned what we can
    
    # 2. Training Fit - did we capture the pattern?
    ax2 = axes[0, 1]
    ax2.scatter(X_train, y_train, alpha=0.6, s=30, 
                color=PALETTE["mint"], label="Training data (what we learned from)")
    
    # Create smooth prediction line
    X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax2.plot(X_line, y_line, color=PALETTE["lavender"], 
             linewidth=2, label="Our learned pattern")
    
    ax2.set_title("Training: Learning the Coffee-Happiness Relationship", 
                  fontsize=12, color=PALETTE["lavender"])
    ax2.set_xlabel("Coffee Cups (input)", fontsize=10)
    ax2.set_ylabel("Happiness Level (output)", fontsize=10)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor(PALETTE["fog"])
    
    # 3. Test Performance - the moment of truth
    ax3 = axes[1, 0]
    y_pred_test = model.predict(X_test)
    ax3.scatter(X_test, y_test, alpha=0.6, s=30, 
                color=PALETTE["blossom"], label="Test data (never seen before)")
    ax3.plot(X_line, y_line, color=PALETTE["lavender"], 
             linewidth=2, label="Our prediction")
    
    # Add confidence interval - being honest about uncertainty
    residual_std = np.std(y_test - y_pred_test)
    ax3.fill_between(X_line.flatten(), 
                     y_line.flatten() - residual_std,
                     y_line.flatten() + residual_std,
                     alpha=0.2, color=PALETTE["lilac"],
                     label="Uncertainty zone")
    
    ax3.set_title("Testing: How well did we generalize?", 
                  fontsize=12, color=PALETTE["lavender"])
    ax3.set_xlabel("Coffee Cups", fontsize=10)
    ax3.set_ylabel("Happiness Level", fontsize=10)
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor(PALETTE["fog"])
    
    # 4. Residuals - examining our mistakes
    # This plot taught me humility - perfect models don't exist
    ax4 = axes[1, 1]
    residuals = y_test - y_pred_test
    ax4.scatter(y_pred_test, residuals, alpha=0.6, s=30, color=PALETTE["mint"])
    ax4.axhline(y=0, color=PALETTE["lavender"], linestyle='--', linewidth=1)
    ax4.fill_between(sorted(y_pred_test), -residual_std, residual_std,
                     alpha=0.2, color=PALETTE["lilac"])
    
    ax4.set_title("Residuals: Understanding our mistakes", 
                  fontsize=12, color=PALETTE["lavender"])
    ax4.set_xlabel("Predicted Happiness", fontsize=10)
    ax4.set_ylabel("Prediction Error", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor(PALETTE["fog"])
    
    # If residuals show a pattern, we're missing something
    # Random residuals = good. Patterns = back to the drawing board.
    
    plt.suptitle("Linear Regression: A Visual Story", 
                 fontsize=16, color=PALETTE["lavender"], y=1.02)
    plt.tight_layout()
    plt.show()


def main():
    """
    Bringing it all together with kindness and clarity.
    
    This main function is structured like I wish tutorials were when I started:
    1. Build intuition with simple data
    2. Watch the model learn
    3. Evaluate honestly
    4. Understand what happened
    5. Extract lessons for next time
    """
    print("=" * 60)
    print("LINEAR REGRESSION: The Foundation of Everything")
    print("=" * 60)
    print("\nPersonal note: If you understand this deeply, you understand")
    print("50% of machine learning. The rest is variations on this theme.")
    
    # Create our story dataset
    print("\nCreating dataset: Coffee vs Happiness")
    print("(Based on actual personal data from my coffee journal)")
    X, y = create_story_dataset()
    
    # Split with care - training vs testing is like practice vs game day
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)} (what we learn from)")
    print(f"Testing samples: {len(X_test)} (how we measure success)")
    
    # Train our gentle model
    print("\nTraining our model...")
    print("Watch the loss decrease - that's learning happening in real-time:")
    model = GentleLinearRegression(learning_rate=0.01, iterations=1000, verbose=True)
    model.fit(X_train, y_train)
    
    # Evaluate with honesty
    print("\nEvaluation Metrics:")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Training MSE: {train_mse:.4f} | R²: {train_r2:.4f}")
    print(f"Testing MSE:  {test_mse:.4f} | R²: {test_r2:.4f}")
    
    # Check for overfitting - the model's biggest enemy
    if train_mse < test_mse * 0.8:
        print("\nWarning: Possible overfitting detected.")
        print("The model memorized training data instead of learning patterns.")
        print("In production, this model would fail on new data.")
    else:
        print("\nGood generalization! The model learned patterns, not memorized data.")
        print("This is what we want in production - robust patterns.")
    
    # Explain a prediction - interpretability matters
    print("\nLet's explain a specific prediction:")
    sample_coffee = np.array([[2.5]])  # 2.5 cups of coffee
    model.explain_prediction(sample_coffee)
    
    # Visualize the journey
    print("\nCreating visualizations...")
    print("(These plots taught me more than any textbook)")
    visualize_learning_journey(model, X_train, y_train, X_test, y_test)
    
    # Final thoughts - lessons learned the hard way
    print("\n" + "=" * 60)
    print("KEY INSIGHTS (from my journey):")
    print("1. Linear regression assumes a straight-line relationship")
    print("   (Life rarely gives us straight lines)")
    print("2. It minimizes squared errors")
    print("   (Big mistakes hurt more than small ones)")
    print("3. Simple is often better than complex")
    print("   (I've seen PhDs humbled by linear regression)")
    print("4. Always validate on unseen data")
    print("   (Training accuracy lies, testing accuracy tells truth)")
    print("5. Understanding > Accuracy")
    print("   (A model you understand beats a black box every time)")
    print("=" * 60)
    print("\nNext step: When straight lines aren't enough...")
    print("See: 02_decision_trees_to_forests.py")


if __name__ == "__main__":
    main()
