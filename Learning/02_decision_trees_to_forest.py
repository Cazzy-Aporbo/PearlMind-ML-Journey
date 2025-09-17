"""
From Single Trees to Random Forests: When Linear Isn't Enough
=============================================================
Author: Cazandra Aporbo (becaziam@gmail.com)
Date: Feb 10, 2025

After months of trying to force linear models to work on non-linear problems
(and failing spectacularly), I discovered decision trees. The moment I understood
them, I felt like I'd been trying to paint with one color and suddenly discovered
the entire palette.

My journey with trees:
- Started with single decision trees (overfit everything)
- Discovered random forests (wisdom of crowds)
- Mastered gradient boosting (learning from mistakes)
- Learned when NOT to use them (interpretability matters)

The humbling realization: Trees think like humans do. "If this, then that."
No complex math, no assumptions about distributions, just simple questions
leading to answers. Yet from this simplicity emerges remarkable power.

Why Trees Changed Everything for Me:
    - They handle non-linearity naturally (life isn't linear)
    - No scaling needed (trees don't care if age is 0-100 or salary is 0-1M)
    - Feature interactions for free (they find them automatically)
    - Can actually explain decisions (try that with a neural network)

Hard-Learned Production Lessons:
    - Single trees overfit like crazy (learned this in my first production model)
    - Random forests are embarrassingly parallel (use all those cores)
    - Feature importance can lie (especially with correlated features)
    - Memory usage grows fast (each tree stores the training data structure)
    - But for tabular data? Often unbeatable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Our aesthetic continues - these colors remind me of decision boundaries
PALETTE = {
    "blossom": "#FFCFE7",
    "lilac": "#F6EAFE", 
    "mint": "#A8E6CF",
    "lavender": "#6B5B95",
    "fog": "#FDF3FF",
    "dusk": "#6E6E80"
}


class InterpretableDecisionTree:
    """
    A decision tree that explains its thinking at every split.
    
    I built this wrapper because sklearn's trees are powerful but opaque.
    In production, when a decision affects someone's loan application or
    medical diagnosis, "the algorithm said so" isn't good enough.
    
    Personal story: A tree-based model I deployed once rejected a loan application.
    When asked why, I couldn't explain it clearly. Never again. Now every split
    has a story.
    """
    
    def __init__(self, max_depth=3):
        """
        Start simple. Depth 3 = 8 possible outcomes.
        
        I learned that depth 3-5 trees are often enough. Beyond that,
        you're probably overfitting. A depth-10 tree once achieved 99.9%
        training accuracy and 60% test accuracy. Lesson learned.
        
        Think of depth like this:
        - Depth 1: One question (are you tall?)
        - Depth 2: Follow-up question (are you tall AND heavy?)
        - Depth 3: Getting specific (tall, heavy, AND young?)
        - Depth 10: Memorizing individuals (tall, heavy, young, born on Tuesday...)
        """
        self.max_depth = max_depth
        self.tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        self.feature_names = None
        self.split_history = []
        
    def fit(self, X, y, feature_names=None):
        """
        Learn, but remember why we made each decision.
        
        This is what I wish sklearn did by default - keep a human-readable
        record of every split decision. In production, this has saved me
        countless times when debugging unexpected predictions.
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.tree.fit(X, y)
        self._extract_split_logic()
        return self
        
    def _extract_split_logic(self):
        """
        Extract human-readable split decisions.
        
        This method turns "node 0: feature[2] <= 3.5" into
        "Is Education <= 3.5 years?" - much more interpretable.
        """
        tree = self.tree.tree_
        feature = tree.feature
        threshold = tree.threshold
        
        self.split_history = []
        for node_id in range(tree.node_count):
            if feature[node_id] != -2:  # Not a leaf
                self.split_history.append({
                    'node': node_id,
                    'feature': self.feature_names[feature[node_id]],
                    'threshold': threshold[node_id],
                    'samples': tree.n_node_samples[node_id]
                })
    
    def predict(self, X):
        """Predict with the learned tree."""
        return self.tree.predict(X)
    
    def explain_path(self, sample):
        """
        Trace the decision path for a single sample.
        
        This is gold for production debugging. Customer complains about
        a prediction? Run this method, get the exact decision path.
        "Your application was rejected because: Age > 25? No. Income > 50k? No."
        
        I've used this to find data quality issues (Age = 999) and
        business logic problems (rejecting everyone from certain zip codes).
        """
        feature = self.tree.tree_.feature
        threshold = self.tree.tree_.threshold
        
        node_indicator = self.tree.decision_path(sample.reshape(1, -1))
        leaf = self.tree.apply(sample.reshape(1, -1))[0]
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        
        print("\nDecision Path Explanation:")
        print("-" * 40)
        print("Let me walk you through how I made this decision...\n")
        
        for node_id in node_index:
            if feature[node_id] != -2:  # Not a leaf
                feature_name = self.feature_names[feature[node_id]]
                feature_value = sample[feature[node_id]]
                threshold_value = threshold[node_id]
                
                if feature_value <= threshold_value:
                    decision = "<="
                    direction = "left"
                else:
                    decision = ">"
                    direction = "right"
                    
                print(f"Question {node_id}: Is {feature_name} {decision} {threshold_value:.2f}?")
                print(f"  Your value: {feature_value:.2f}")
                print(f"  Decision: Going {direction}\n")
        
        prediction = self.tree.tree_.value[leaf][0][0]
        print(f"Final prediction: {prediction:.2f}")
        print("\nThis path was taken by {:.0f} training samples".format(
            self.tree.tree_.n_node_samples[leaf]))
        return prediction


class ForestOfWisdom:
    """
    Random Forest: Because diverse opinions lead to better decisions.
    
    The breakthrough moment: realizing that many mediocre models can
    combine into one excellent model. It's democracy for algorithms.
    
    Personal insight: Random forests are like asking 100 experts, where
    each expert only sees part of the picture. Somehow, their collective
    wisdom exceeds any individual expert. This principle changed how I
    approach not just ML, but team decisions too.
    """
    
    def __init__(self, n_trees=100):
        """
        100 trees is my sweet spot - enough diversity without 
        excessive computation. I've tested everything from 10 to 1000 trees.
        Diminishing returns hit hard after 100-200.
        """
        self.n_trees = n_trees
        self.forest = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=5,  # Shallow trees to prevent overfitting
            random_state=42,
            n_jobs=-1  # Use all CPU cores - trees are independent
        )
        self.feature_names = None
        
    def fit(self, X, y, feature_names=None):
        """
        Train a forest, each tree learning from different perspectives.
        
        The magic: Each tree sees a different random sample of data AND
        features. This forced diversity prevents groupthink (overfitting).
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.forest.fit(X, y)
        return self
        
    def get_feature_importance_story(self):
        """
        Which features did the forest find most useful?
        
        Warning from experience: These importances can be misleading!
        1. Correlated features split importance
        2. High-cardinality features appear more important
        3. It measures reduction in impurity, not actual predictive value
        
        I once removed a "low importance" feature and accuracy dropped 10%.
        Always validate with actual removal, not just importance scores.
        """
        importances = self.forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature Importance (what the forest learned to pay attention to):")
        print("-" * 50)
        print("\nThink of this as 'which questions were most useful?'\n")
        
        for i in range(min(10, len(indices))):
            idx = indices[i]
            print(f"{i+1:2d}. {self.feature_names[idx]:20s}: {importances[idx]:.4f}")
            
            # Add context based on importance level
            if importances[idx] > 0.3:
                print(f"    → Critical feature - drives many decisions")
            elif importances[idx] > 0.1:
                print(f"    → Important feature - significant influence")
            elif importances[idx] > 0.05:
                print(f"    → Moderate feature - some influence")
            else:
                print(f"    → Minor feature - rarely used")
        
        # Add warning about interpretation
        print("\nImportant caveats I learned the hard way:")
        print("- Importance doesn't mean causation")
        print("- Correlated features share importance")
        print("- Random features can show non-zero importance")
        print("- Always validate by removing features and re-testing")
        
        return importances


def create_complex_dataset():
    """
    Create a dataset that linear regression would struggle with.
    
    This is based on a real problem I faced: predicting salaries.
    Linear models assumed experience always increases salary linearly.
    Reality: After 20 years, experience matters less. Education multiplies
    the effect of experience. Age has a sweet spot. Coffee... well, that's
    my personal addition.
    """
    np.random.seed(42)
    n_samples = 500
    
    # Features inspired by real HR data
    age = np.random.uniform(22, 65, n_samples)
    experience = np.random.uniform(0, 40, n_samples)
    education = np.random.uniform(12, 22, n_samples)  # Years of education
    coffee = np.random.uniform(0, 10, n_samples)  # My signature feature
    
    # Salary with complex, real-world interactions
    salary = (
        30000 +  # Base salary everyone gets
        
        # Linear effects (the obvious patterns)
        1000 * experience +  
        500 * education +  
        100 * age +  
        
        # Non-linear interactions (the hidden patterns that matter)
        50 * experience * (education > 16) +  # Degree amplifies experience
        -20 * (age - 45) ** 2 * (age > 45) +  # Age penalty after 45 (unfair but real)
        200 * coffee * (coffee < 4) +  # Optimal coffee zone
        -100 * coffee * (coffee > 6) +  # Too much coffee hurts
        
        # Random noise (life isn't deterministic)
        np.random.normal(0, 5000, n_samples)
    )
    
    X = np.column_stack([age, experience, education, coffee])
    feature_names = ['Age', 'Experience', 'Education', 'Coffee_Cups']
    
    return X, salary, feature_names


def compare_model_evolution(X_train, y_train, X_test, y_test, feature_names):
    """
    Show the evolution from simple to complex models.
    
    This comparison changed my perspective on model selection.
    More complex isn't always better. It's about finding the right
    tool for the specific problem.
    """
    results = {}
    
    print("\n" + "="*60)
    print("MODEL EVOLUTION: My Journey from Simple to Sophisticated")
    print("="*60)
    
    # 1. Single Decision Tree - My first non-linear model
    print("\n1. SINGLE DECISION TREE (depth=3)")
    print("-" * 40)
    print("My first attempt - interpretable but limited...")
    simple_tree = InterpretableDecisionTree(max_depth=3)
    simple_tree.fit(X_train, y_train, feature_names)
    
    y_pred = simple_tree.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R²: {r2:.4f}")
    print("\nWhat I learned:")
    print("- Super interpretable (can draw on whiteboard)")
    print("- Captures non-linearity (finally!)")
    print("- But too simple for complex patterns")
    
    results['Simple Tree'] = {'mse': mse, 'r2': r2, 'predictions': y_pred}
    
    # 2. Deeper Tree - When I thought deeper was always better
    print("\n2. DEEP DECISION TREE (depth=10)")
    print("-" * 40)
    print("My overconfident phase - 'more depth = more better', right?")
    deep_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
    deep_tree.fit(X_train, y_train)
    
    y_pred = deep_tree.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R²: {r2:.4f}")
    print("\nHarsh reality check:")
    print("- Memorized training data perfectly")
    print("- Failed miserably on test data")
    print("- Lesson: Complexity != Performance")
    
    results['Deep Tree'] = {'mse': mse, 'r2': r2, 'predictions': y_pred}
    
    # 3. Random Forest - The game changer
    print("\n3. RANDOM FOREST (100 trees)")
    print("-" * 40)
    print("The breakthrough - wisdom of crowds...")
    forest = ForestOfWisdom(n_trees=100)
    forest.fit(X_train, y_train, feature_names)
    
    y_pred = forest.forest.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R²: {r2:.4f}")
    print("\nWhy this works so well:")
    print("- Each tree's errors are different (uncorrelated)")
    print("- Averaging reduces variance without increasing bias much")
    print("- Parallel processing makes it fast")
    print("- My go-to model for competitions")
    
    results['Random Forest'] = {'mse': mse, 'r2': r2, 'predictions': y_pred}
    
    # Show feature importance
    forest.get_feature_importance_story()
    
    # 4. Gradient Boosting - The perfectionist
    print("\n4. GRADIENT BOOSTING (learning from mistakes)")
    print("-" * 40)
    print("The sophisticated approach - each tree fixes previous errors...")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    
    y_pred = gb.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R²: {r2:.4f}")
    print("\nThe double-edged sword:")
    print("- Often wins competitions (XGBoost/LightGBM)")
    print("- Sequential nature = slower training")
    print("- Prone to overfitting if not careful")
    print("- My choice when accuracy is everything")
    
    results['Gradient Boosting'] = {'mse': mse, 'r2': r2, 'predictions': y_pred}
    
    return results, simple_tree, forest


def visualize_model_comparison(results, X_test, y_test, feature_names):
    """
    Beautiful visualizations comparing our models.
    
    These plots tell the story of model evolution. Each time I look at them,
    I'm reminded that there's no free lunch in ML - every model has trade-offs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor(PALETTE["fog"])
    
    models = list(results.keys())
    colors = [PALETTE["mint"], PALETTE["lilac"], PALETTE["blossom"], PALETTE["lavender"]]
    
    # Prediction vs Actual for each model
    for idx, (model_name, color) in enumerate(zip(models, colors)):
        ax = axes[idx // 2, idx % 2]
        y_pred = results[model_name]['predictions']
        
        ax.scatter(y_test, y_pred, alpha=0.5, s=20, color=color)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'k--', alpha=0.5, label='Perfect prediction')
        
        # Add R² to plot
        r2 = results[model_name]['r2']
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=PALETTE["fog"], alpha=0.8))
        
        # Add personal insights
        insights = {
            'Simple Tree': 'Clean but limited',
            'Deep Tree': 'Overfit disaster',
            'Random Forest': 'Reliable workhorse',
            'Gradient Boosting': 'Competition winner'
        }
        
        ax.set_title(f'{model_name}\n({insights[model_name]})', 
                    fontsize=14, color=PALETTE["dusk"])
        ax.set_xlabel('Actual Salary', fontsize=10)
        ax.set_ylabel('Predicted Salary', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(PALETTE["fog"])
        ax.legend()
    
    plt.suptitle("Model Evolution: Each Step Taught Me Something New", 
                 fontsize=16, color=PALETTE["lavender"], y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Model comparison bar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE["fog"])
    
    model_names = list(results.keys())
    r2_scores = [results[m]['r2'] for m in model_names]
    
    x = np.arange(len(model_names))
    bars = ax.bar(x, r2_scores, color=[PALETTE["mint"], PALETTE["lilac"], 
                                       PALETTE["blossom"], PALETTE["lavender"]], 
                 alpha=0.8)
    
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Model Performance Comparison\n(Higher = Better fit, but watch for overfitting!)', 
                 fontsize=14, color=PALETTE["lavender"])
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor(PALETTE["fog"])
    
    # Add value labels and insights
    for bar, score, name in zip(bars, r2_scores, model_names):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    The journey from a single decision to a forest of wisdom.
    
    This progression mirrors my own learning path. Started with simple trees,
    got drunk on complexity with deep trees, found balance with forests,
    and reached sophistication with boosting. Each step was necessary.
    """
    print("="*70)
    print("FROM TREES TO FORESTS: My Non-Linear Awakening")
    print("="*70)
    print("\nPersonal story: I spent 3 months trying to make linear models")
    print("work on customer churn data. Trees solved it in 30 minutes.")
    print("Sometimes the problem isn't you - it's the tool.")
    
    # Create complex dataset
    print("\nCreating complex dataset with interactions...")
    print("(Based on real salary data, with my coffee addition)")
    X, y, feature_names = create_complex_dataset()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset: {len(X)} samples, {len(feature_names)} features")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Target: Salary (with complex interactions)")
    print("\nThe interactions here would break linear regression.")
    print("But trees? They eat this for breakfast.")
    
    # Compare models
    results, simple_tree, forest = compare_model_evolution(
        X_train, y_train, X_test, y_test, feature_names
    )
    
    # Explain a single prediction
    print("\n" + "="*60)
    print("EXPLAINABILITY DEMONSTRATION")
    print("="*60)
    print("This is why I love trees - I can explain every decision.")
    
    sample_idx = 0
    sample = X_test[sample_idx]
    actual = y_test[sample_idx]
    
    print(f"\nSample person:")
    for fname, fval in zip(feature_names, sample):
        print(f"  {fname}: {fval:.2f}")
    print(f"  Actual salary: ${actual:,.2f}")
    
    print("\nHow the simple tree made its decision:")
    prediction = simple_tree.explain_path(sample)
    
    # Visualize comparisons
    print("\nGenerating visualizations...")
    print("(These plots show why ensemble methods dominate competitions)")
    visualize_model_comparison(results, X_test, y_test, feature_names)
    
    # Lessons learned
    print("\n" + "="*60)
    print("LESSONS FROM THE FOREST (learned through trial and error):")
    print("1. Single trees are glass cannons - powerful but fragile")
    print("2. Deep trees memorize, shallow trees generalize")
    print("3. Random forests are my Swiss Army knife - always solid")
    print("4. Gradient boosting wins Kaggle (but needs babysitting)")
    print("5. Feature importance lies sometimes (validate by removal)")
    print("6. Trees find interactions linear models miss")
    print("7. No scaling needed = one less thing to mess up")
    print("8. But trees can't extrapolate (predictions are bounded by training data)")
    print("="*60)
    print("\nNext step: When even trees aren't enough...")
    print("See: 03_neural_networks_awakening.py")


if __name__ == "__main__":
    main()