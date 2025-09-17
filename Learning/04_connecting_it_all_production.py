"""
Connecting It All: A Production ML System
=========================================
Author: Cazandra Aporbo (becaziam@gmail.com)
Date: Feb 19, 2025

After years of building models in isolation, I had an epiphany:
Production ML isn't about having the best model. It's about having
the right model at the right time. This file represents everything
I wish I'd known when I deployed my first production system.

The painful lessons that led here:
- My first production model: 99% accurate, 10 second latency (unusable)
- My second: Neural network for a linear problem (overengineered)
- My third: No fallback, crashed on edge cases (3 AM wake-up call)
- My fourth: Perfect model, no monitoring (silent degradation)

Now I know: Production ML is 20% modeling, 80% engineering.

What This System Represents:
    An intelligent router that sends queries to the right model.
    Simple problems get simple models (fast, interpretable).
    Complex problems get complex models (when justified).
    Everything has a fallback (because Murphy's Law is real).
    Everything is monitored (what you don't measure, you can't fix).

Hard-Won Production Wisdom:
    - Latency matters more than accuracy (usually)
    - Simple models fail in predictable ways (good)
    - Complex models fail in mysterious ways (bad)
    - Always have a fallback (always)
    - Log everything (you'll thank yourself later)
    - Monitor drift (models decay like fruit)
    - Explainability isn't optional (regulators will ask)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
import hashlib
from datetime import datetime

# Our models from the journey
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Aesthetics for our final visualization
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


class ModelComplexity(Enum):
    """
    How complex is this query?
    
    This enum represents months of learning when to use which model.
    The temptation is always to use the most complex model.
    The reality is that 80% of problems are SIMPLE.
    """
    SIMPLE = "simple"       # Linear model territory
    MODERATE = "moderate"   # Trees shine here
    COMPLEX = "complex"     # Neural network justified
    CRITICAL = "critical"   # Ensemble everything, accuracy matters most


@dataclass
class PredictionRequest:
    """
    A request that needs intelligent routing.
    
    In production, every request tells a story. This class
    captures not just the data, but the context. When debugging
    at 3 AM, you'll appreciate having request_id and timestamp.
    """
    request_id: str
    features: np.ndarray
    complexity: Optional[ModelComplexity] = None
    timestamp: datetime = None
    metadata: Dict = None
    
    def __post_init__(self):
        """
        Post-init is where I add production necessities.
        Timestamps for debugging, IDs for tracking.
        """
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.request_id is None:
            # Generate unique ID from features - helps track duplicates
            self.request_id = hashlib.md5(
                str(self.features).encode()
            ).hexdigest()[:8]


@dataclass
class PredictionResponse:
    """
    Response with full traceability.
    
    Early in my career, I returned just predictions.
    Then came the questions: "Why this prediction?",
    "How confident?", "Which model?", "How long did it take?"
    Now I return everything.
    """
    request_id: str
    prediction: Any
    confidence: float
    model_used: str
    latency_ms: float
    explanation: Dict
    fallback_triggered: bool = False


class ModelRouter:
    """
    The brain of our system - routes requests to appropriate models.
    
    This class embodies years of production lessons:
    - Not every problem needs deep learning
    - Latency budgets are real
    - Models fail (plan for it)
    - Simple often beats complex
    
    Think of this as an emergency room triage nurse - 
    quickly assessing problems and routing to the right specialist.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_history = []
        self.routing_stats = {
            'simple': 0,
            'moderate': 0,
            'complex': 0,
            'critical': 0
        }
        
    def register_model(self, name: str, model: Any, 
                       complexity: ModelComplexity,
                       scaler: Optional[StandardScaler] = None):
        """
        Register a model with its complexity level.
        
        I learned to track everything about each model:
        - How many times called (usage patterns)
        - Total latency (performance degradation)
        - Error count (reliability)
        
        This metadata has saved me countless times in production.
        """
        self.models[name] = {
            'model': model,
            'complexity': complexity,
            'scaler': scaler,
            'calls': 0,
            'total_latency': 0,
            'errors': 0
        }
        print(f"Registered {name} for {complexity.value} queries")
        
    def assess_complexity(self, request: PredictionRequest) -> ModelComplexity:
        """
        Determine query complexity based on features.
        
        This is simplified for demonstration. In production, I use:
        - A dedicated complexity classifier (trained on historical data)
        - Feature statistics (variance, cardinality, sparsity)
        - Business rules (certain features trigger complex models)
        - Time constraints (if latency budget is tight, force simple)
        
        The goal: Use the simplest model that will work.
        """
        features = request.features
        
        # Simple heuristics for demonstration
        n_features = len(features)
        feature_variance = np.var(features)
        feature_range = np.ptp(features)  # Peak to peak (max - min)
        
        # My complexity scoring formula (refined over many iterations)
        complexity_score = (
            n_features * 0.3 +        # More features = more complex
            feature_variance * 10 +    # High variance = complex patterns
            feature_range * 0.2        # Wide range = diverse scenarios
        )
        
        # Thresholds learned from production data
        if complexity_score < 5:
            return ModelComplexity.SIMPLE
        elif complexity_score < 15:
            return ModelComplexity.MODERATE
        elif complexity_score < 30:
            return ModelComplexity.COMPLEX
        else:
            return ModelComplexity.CRITICAL
            
    def route_request(self, request: PredictionRequest) -> PredictionResponse:
        """
        Route request to appropriate model with fallback logic.
        
        This method is the heart of the system. Every decision here
        comes from a production incident:
        - Try primary model first (optimistic)
        - Have fallback ready (pessimistic)
        - Track everything (realistic)
        """
        start_time = time.time()
        
        # Assess complexity if not provided
        if request.complexity is None:
            request.complexity = self.assess_complexity(request)
        
        # Update routing statistics - essential for optimization
        self.routing_stats[request.complexity.value] += 1
        
        # Select models
        primary_model = self._select_model(request.complexity)
        fallback_model = self._select_fallback(request.complexity)
        
        try:
            # Try primary model
            response = self._execute_prediction(
                request, primary_model, start_time
            )
            
        except Exception as e:
            # Fallback on error - this has saved me many times
            print(f"Warning: Primary model failed: {e}")
            print(f"Falling back to {fallback_model}")
            
            response = self._execute_prediction(
                request, fallback_model, start_time
            )
            response.fallback_triggered = True
            
            # In production, I'd alert here
            # send_alert(f"Model {primary_model} failed, used fallback")
            
        # Record performance for analysis
        self.performance_history.append({
            'timestamp': request.timestamp,
            'complexity': request.complexity.value,
            'model': response.model_used,
            'latency': response.latency_ms,
            'confidence': response.confidence
        })
        
        return response
        
    def _select_model(self, complexity: ModelComplexity) -> str:
        """
        Select best model for complexity level.
        
        Selection strategy I've refined:
        1. Find models matching complexity
        2. Choose one with best latency/accuracy trade-off
        3. Consider recent error rates
        4. Factor in current load
        
        In production, this might use multi-armed bandit algorithms.
        """
        candidates = [
            name for name, info in self.models.items()
            if info['complexity'] == complexity
        ]
        
        if not candidates:
            # No exact match, degrade gracefully
            # This is better than crashing
            if complexity == ModelComplexity.SIMPLE:
                # Try moderate models for simple problems
                candidates = [name for name, info in self.models.items()
                             if info['complexity'] == ModelComplexity.MODERATE]
            else:
                # Try complex models as last resort
                candidates = [name for name, info in self.models.items()
                             if info['complexity'] == ModelComplexity.COMPLEX]
        
        # Select based on performance (least average latency)
        # In production, I'd weight this by accuracy too
        if candidates:
            best = min(candidates, key=lambda x: 
                      self.models[x]['total_latency'] / max(1, self.models[x]['calls']))
            return best
        
        # Ultimate fallback - just pick something
        return list(self.models.keys())[0]
        
    def _select_fallback(self, complexity: ModelComplexity) -> str:
        """
        Select simpler fallback model.
        
        Philosophy: When things go wrong, simplify.
        Complex model failed? Try trees.
        Trees failed? Try linear.
        Linear failed? Return baseline prediction.
        
        This cascade has prevented total failures many times.
        """
        if complexity == ModelComplexity.COMPLEX:
            return self._select_model(ModelComplexity.MODERATE)
        elif complexity == ModelComplexity.MODERATE:
            return self._select_model(ModelComplexity.SIMPLE)
        else:
            # Simple is already the simplest
            return self._select_model(ModelComplexity.SIMPLE)
            
    def _execute_prediction(self, request: PredictionRequest, 
                           model_name: str, start_time: float) -> PredictionResponse:
        """
        Execute prediction with a specific model.
        
        This method handles the actual prediction plus all the
        production necessities: scaling, timing, error handling,
        explanation generation. Each line represents a lesson learned.
        """
        model_info = self.models[model_name]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Scale features if needed - forgetting this crashed production once
        features = request.features.reshape(1, -1)
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            # Classification with probability
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities)
        else:
            # Regression or no probability
            prediction = model.predict(features)[0]
            confidence = 0.0  # No natural confidence measure
        
        # Calculate latency - critical for SLA monitoring
        latency_ms = (time.time() - start_time) * 1000
        
        # Update model statistics
        model_info['calls'] += 1
        model_info['total_latency'] += latency_ms
        
        # Generate explanation - regulators love this
        explanation = self._generate_explanation(
            model_name, features, prediction, confidence
        )
        
        return PredictionResponse(
            request_id=request.request_id,
            prediction=prediction,
            confidence=confidence,
            model_used=model_name,
            latency_ms=latency_ms,
            explanation=explanation,
            fallback_triggered=False
        )
        
    def _generate_explanation(self, model_name: str, features: np.ndarray,
                             prediction: Any, confidence: float) -> Dict:
        """
        Generate human-readable explanation.
        
        Learned the hard way: Every prediction needs an explanation.
        Customers ask why. Regulators demand it. Debugging requires it.
        This method provides that "why".
        """
        model_info = self.models[model_name]
        
        explanation = {
            'model_type': type(model_info['model']).__name__,
            'complexity_level': model_info['complexity'].value,
            'features_used': features.shape[1],
            'confidence': f"{confidence:.2%}" if confidence > 0 else "N/A",
            'reasoning': self._get_reasoning(model_name, features)
        }
        
        return explanation
        
    def _get_reasoning(self, model_name: str, features: np.ndarray) -> str:
        """
        Generate model-specific reasoning.
        
        Each model type explains itself differently.
        Linear models: coefficients
        Trees: feature importance
        Neural networks: ...good luck
        
        This is where interpretability matters.
        """
        model = self.models[model_name]['model']
        
        if isinstance(model, LinearRegression):
            return "Linear combination of features with learned weights"
        elif isinstance(model, RandomForestClassifier):
            return f"Ensemble of {model.n_estimators} decision trees voted"
        elif isinstance(model, MLPClassifier):
            layers = len(model.hidden_layer_sizes)
            return f"Neural network with {layers} hidden layers processed patterns"
        else:
            return "Model-specific logic applied to input features"
            
    def get_performance_report(self) -> pd.DataFrame:
        """
        Generate performance report.
        
        This report has been my best friend in production.
        It answers: Which models are actually being used?
        What's the latency distribution? Where are the bottlenecks?
        
        I run this daily and alert on anomalies.
        """
        if not self.performance_history:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.performance_history)
        
        print("\nPERFORMANCE REPORT")
        print("=" * 60)
        print("(This is what I check every morning with coffee)")
        
        # Routing distribution - are we using models as expected?
        print("\nRouting Distribution:")
        print("(Shows which complexity levels we're actually seeing)")
        for complexity, count in self.routing_stats.items():
            total = sum(self.routing_stats.values())
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {complexity:10s}: {count:4d} requests ({percentage:.1f}%)")
            
            # My rules of thumb
            if complexity == 'simple' and percentage < 50:
                print(f"    Note: Less simple than expected - investigate")
            elif complexity == 'complex' and percentage > 30:
                print(f"    Note: Many complex queries - check if justified")
        
        # Model usage - which models are working hardest?
        print("\nModel Usage Statistics:")
        print("(Shows which models are carrying the load)")
        for name, info in self.models.items():
            if info['calls'] > 0:
                avg_latency = info['total_latency'] / info['calls']
                print(f"  {name:15s}: {info['calls']:4d} calls, "
                      f"{avg_latency:.2f}ms avg latency")
                
                # Performance warnings
                if avg_latency > 100:
                    print(f"    Warning: High latency - consider optimization")
                if info['errors'] > info['calls'] * 0.01:
                    print(f"    Warning: Error rate > 1% - investigate")
        
        # Latency by complexity - meeting SLAs?
        print("\nLatency by Complexity:")
        print("(Critical for SLA compliance)")
        for complexity in ['simple', 'moderate', 'complex', 'critical']:
            subset = df[df['complexity'] == complexity]
            if len(subset) > 0:
                p50 = subset['latency'].median()
                p99 = subset['latency'].quantile(0.99)
                print(f"  {complexity:10s}: P50={p50:.2f}ms, P99={p99:.2f}ms")
                
                # SLA checks (my typical limits)
                if complexity == 'simple' and p99 > 50:
                    print(f"    Warning: Simple queries too slow!")
                elif complexity == 'complex' and p99 > 500:
                    print(f"    Warning: Complex queries exceeding budget!")
        
        return df


class ModelOrchestrator:
    """
    Orchestrates the entire ML system.
    
    This is where individual models become a symphony.
    Think of it as a conductor - coordinating different
    instruments (models) to create harmony (predictions).
    
    This class represents my philosophy: ML in production
    is about systems, not individual models.
    """
    
    def __init__(self):
        self.router = ModelRouter()
        self.monitoring = {
            'requests_processed': 0,
            'average_latency': 0,
            'errors': 0,
            'fallbacks': 0
        }
        
    def build_model_zoo(self):
        """
        Build our collection of models.
        
        Each model here represents a stage in my learning journey:
        - Linear: Where I started
        - Random Forest: Where I found success
        - Neural Network: Where I went too far
        - The combination: Where I found balance
        """
        print("\nBuilding Model Zoo...")
        print("-" * 40)
        print("Creating models from our journey...")
        
        # Simple: Linear Model - Old reliable
        print("\n1. Linear Model (Simple)")
        linear_model = LogisticRegression(random_state=42)
        # Train on dummy data for demo
        X_simple = np.random.randn(100, 5)
        y_simple = (X_simple.sum(axis=1) > 0).astype(int)
        linear_model.fit(X_simple, y_simple)
        
        self.router.register_model(
            "linear_baseline",
            linear_model,
            ModelComplexity.SIMPLE
        )
        print("   Role: Handle simple, linear patterns")
        print("   Strength: Fast, interpretable")
        print("   Weakness: Can't handle complexity")
        
        # Moderate: Random Forest - The workhorse
        print("\n2. Random Forest (Moderate)")
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        X_moderate = np.random.randn(200, 5)
        y_moderate = ((X_moderate[:, 0] * X_moderate[:, 1]) > 0).astype(int)
        rf_model.fit(X_moderate, y_moderate)
        
        self.router.register_model(
            "random_forest",
            rf_model,
            ModelComplexity.MODERATE
        )
        print("   Role: Handle non-linear patterns")
        print("   Strength: Robust, handles interactions")
        print("   Weakness: Can be memory intensive")
        
        # Complex: Neural Network - The heavy artillery
        print("\n3. Neural Network (Complex)")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            random_state=42,
            max_iter=500
        )
        X_complex = np.random.randn(300, 5)
        y_complex = (np.sin(X_complex[:, 0]) * np.cos(X_complex[:, 1]) > 0).astype(int)
        nn_model.fit(X_complex, y_complex)
        
        scaler = StandardScaler()
        scaler.fit(X_complex)
        
        self.router.register_model(
            "neural_network",
            nn_model,
            ModelComplexity.COMPLEX,
            scaler
        )
        print("   Role: Handle complex, abstract patterns")
        print("   Strength: Can learn anything (theoretically)")
        print("   Weakness: Black box, needs lots of data")
        
        print("\nModel zoo ready for production!")
        
    def process_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """
        Process a batch of requests intelligently.
        
        Batch processing is crucial in production:
        - Amortizes overhead
        - Enables optimizations
        - Improves throughput
        
        But remember: Batch size is a trade-off between
        latency (smaller batches) and throughput (larger batches).
        """
        responses = []
        
        print(f"\nProcessing batch of {len(requests)} requests...")
        print("(In production, this would be async with queues)")
        
        for request in requests:
            try:
                response = self.router.route_request(request)
                responses.append(response)
                
                # Update monitoring
                self.monitoring['requests_processed'] += 1
                if response.fallback_triggered:
                    self.monitoring['fallbacks'] += 1
                    
            except Exception as e:
                print(f"Failed to process {request.request_id}: {e}")
                self.monitoring['errors'] += 1
                # In production: log to error tracking service
                
        # Update average latency
        if responses:
            avg_latency = np.mean([r.latency_ms for r in responses])
            self.monitoring['average_latency'] = avg_latency
            
        return responses
        
    def visualize_system_performance(self):
        """
        Create beautiful visualizations of system performance.
        
        These visualizations have been my window into production health.
        I have dashboards with these plots updating in real-time.
        When something looks wrong here, something IS wrong.
        """
        df = self.router.get_performance_report()
        
        if df.empty:
            print("No data to visualize yet!")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(PALETTE["fog"])
        
        # 1. Routing distribution pie chart
        ax1 = axes[0, 0]
        routing_data = list(self.router.routing_stats.values())
        routing_labels = list(self.router.routing_stats.keys())
        colors = [PALETTE["mint"], PALETTE["lilac"], 
                 PALETTE["blossom"], PALETTE["lavender"]]
        
        # Only plot if we have data
        if sum(routing_data) > 0:
            ax1.pie(routing_data, labels=routing_labels, colors=colors,
                    autopct='%1.1f%%', startangle=90)
            ax1.set_title("Request Complexity Distribution\n(What types of problems are we seeing?)", 
                         fontsize=12, color=PALETTE["dusk"])
        else:
            ax1.text(0.5, 0.5, 'No data yet', ha='center', va='center')
            ax1.set_title("Request Complexity Distribution", 
                         fontsize=12, color=PALETTE["dusk"])
        
        # 2. Latency distribution histogram
        ax2 = axes[0, 1]
        for complexity, color in zip(['simple', 'moderate', 'complex'], 
                                     [PALETTE["mint"], PALETTE["lilac"], PALETTE["blossom"]]):
            subset = df[df['complexity'] == complexity]
            if len(subset) > 0:
                ax2.hist(subset['latency'], alpha=0.6, label=complexity,
                        color=color, bins=20)
        
        ax2.set_xlabel("Latency (ms)", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.set_title("Latency Distribution by Complexity\n(Are we meeting performance targets?)", 
                     fontsize=12, color=PALETTE["dusk"])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor(PALETTE["fog"])
        
        # 3. Model usage bar chart
        ax3 = axes[1, 0]
        model_usage = {}
        for name, info in self.router.models.items():
            model_usage[name] = info['calls']
        
        if model_usage:
            ax3.bar(model_usage.keys(), model_usage.values(),
                   color=PALETTE["lavender"], alpha=0.7)
            ax3.set_xlabel("Model", fontsize=10)
            ax3.set_ylabel("Usage Count", fontsize=10)
            ax3.set_xticklabels(model_usage.keys(), rotation=45, ha='right')
        ax3.set_title("Model Usage Statistics\n(Which models are doing the heavy lifting?)", 
                     fontsize=12, color=PALETTE["dusk"])
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_facecolor(PALETTE["fog"])
        
        # 4. System health metrics
        ax4 = axes[1, 1]
        
        # Calculate success rate
        total = self.monitoring['requests_processed']
        success_rate = (1 - self.monitoring['errors']/max(1, total)) * 100
        
        metrics_text = f"""System Health Dashboard
        
Total Requests: {self.monitoring['requests_processed']}
Average Latency: {self.monitoring['average_latency']:.2f}ms
Errors: {self.monitoring['errors']}
Fallbacks Used: {self.monitoring['fallbacks']}
Success Rate: {success_rate:.1f}%

Status: {"Healthy" if success_rate > 95 else "Needs Attention"}"""
        
        ax4.text(0.5, 0.5, metrics_text, 
                transform=ax4.transAxes,
                fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor=PALETTE["lilac"], alpha=0.3))
        ax4.set_title("System Health Metrics\n(Overall system status)", 
                     fontsize=12, color=PALETTE["dusk"])
        ax4.axis('off')
        
        plt.suptitle("Production ML System Performance Dashboard", 
                    fontsize=16, color=PALETTE["lavender"], y=1.02)
        plt.tight_layout()
        plt.show()


def simulate_production_traffic():
    """
    Simulate realistic production traffic with varying complexity.
    
    This simulation is based on actual production patterns I've seen:
    - Most queries are simple (Pareto principle applies)
    - Complex queries are rare but important
    - Edge cases will find you (Murphy's Law)
    - Performance varies throughout the day
    """
    print("\nSimulating Production Traffic...")
    print("=" * 60)
    print("Creating realistic request distribution...")
    print("(Based on actual production patterns I've observed)")
    
    # Create orchestrator
    orchestrator = ModelOrchestrator()
    orchestrator.build_model_zoo()
    
    # Generate diverse requests mimicking real traffic
    requests = []
    
    # Simple requests (40% - bulk of traffic)
    print("\nGenerating simple requests (linear patterns)...")
    for i in range(40):
        features = np.random.randn(5) * 0.5  # Low variance
        requests.append(PredictionRequest(
            request_id=f"simple_{i}",
            features=features
        ))
    
    # Moderate requests (35% - typical business logic)
    print("Generating moderate requests (tree-worthy patterns)...")
    for i in range(35):
        features = np.random.randn(5) * 1.5  # Medium variance
        requests.append(PredictionRequest(
            request_id=f"moderate_{i}",
            features=features
        ))
    
    # Complex requests (20% - the challenging ones)
    print("Generating complex requests (neural network territory)...")
    for i in range(20):
        features = np.random.randn(5) * 3  # High variance
        requests.append(PredictionRequest(
            request_id=f"complex_{i}",
            features=features
        ))
    
    # Critical requests (5% - when accuracy really matters)
    print("Generating critical requests (pull out all stops)...")
    for i in range(5):
        features = np.random.randn(5) * 5  # Very high variance
        requests.append(PredictionRequest(
            request_id=f"critical_{i}",
            features=features,
            complexity=ModelComplexity.CRITICAL
        ))
    
    # Process requests
    print("\nProcessing requests through the system...")
    responses = orchestrator.process_batch(requests)
    
    # Show sample responses
    print("\nSample Responses (what the system returned):")
    print("-" * 40)
    for response in responses[:3]:
        print(f"\nRequest: {response.request_id}")
        print(f"  Model Used: {response.model_used}")
        print(f"  Prediction: {response.prediction}")
        print(f"  Confidence: {response.confidence:.2%}")
        print(f"  Latency: {response.latency_ms:.2f}ms")
        print(f"  Reasoning: {response.explanation['reasoning']}")
        
        # My mental model for evaluating responses
        if response.latency_ms < 10:
            print(f"  Performance: Excellent (under 10ms)")
        elif response.latency_ms < 50:
            print(f"  Performance: Good (under 50ms)")
        elif response.latency_ms < 100:
            print(f"  Performance: Acceptable (under 100ms)")
        else:
            print(f"  Performance: Needs optimization (over 100ms)")
    
    # Visualize performance
    print("\nGenerating Performance Dashboard...")
    print("(This is what I check first thing every morning)")
    orchestrator.visualize_system_performance()
    
    return orchestrator


def main():
    """
    The culmination of our journey: A production ML system.
    
    This main function represents everything I've learned about
    production ML. It's not about the models (though they matter).
    It's about the system - routing, fallbacks, monitoring, and
    most importantly, solving real problems reliably.
    
    If you take away one thing: Production ML is engineering first,
    data science second. The best model that doesn't work in production
    is worse than a simple model that does.
    """
    print("="*70)
    print("CONNECTING IT ALL: Production ML System")
    print("="*70)
    print("\nWelcome to the final lesson of our journey.")
    print("\nThis system represents years of hard-won wisdom:")
    print("- Use the right tool for the job (not always the fanciest)")
    print("- Plan for failure (it will happen)")
    print("- Monitor everything (what you don't measure, you can't fix)")
    print("- Simple beats complex (when it works)")
    print("- Latency matters (users won't wait)")
    print("- Explainability isn't optional (trust requires transparency)")
    
    # Run production simulation
    orchestrator = simulate_production_traffic()
    
    # Production insights
    print("\n" + "="*70)
    print("PRODUCTION INSIGHTS (learned the hard way):")
    print("-" * 70)
    print("1. Not every problem needs deep learning")
    print("   I've seen linear regression outperform neural networks")
    print("\n2. Latency matters as much as accuracy")
    print("   99% accurate with 10s latency = unusable")
    print("\n3. Simple models are easier to debug at 3 AM")
    print("   And yes, you will be debugging at 3 AM")
    print("\n4. Ensemble different architectures for robustness")
    print("   Diversity in models = resilience in production")
    print("\n5. Always have fallback options")
    print("   Primary model will fail. Be ready.")
    print("\n6. Monitor everything")
    print("   If you didn't log it, it didn't happen")
    print("\n7. Explainability becomes critical in production")
    print("   'The model said so' doesn't fly with regulators")
    print("\n8. The best model is the one that works reliably")
    print("   Fancy doesn't mean better")
    print("\n9. System design > model complexity")
    print("   A well-designed system with simple models beats")
    print("   a poor system with complex models")
    print("\n10. Production ML is 20% modeling, 80% engineering")
    print("    The model is the easy part")
    print("="*70)
    
    print("\nCongratulations! You've completed the ML journey:")
    print("   From fundamentals → production systems")
    print("   From single models → orchestrated intelligence")
    print("   From theory → practice")
    print("\nWhat's next?")
    print("   - Build your own production system")
    print("   - Learn from your failures (you will have them)")
    print("   - Share your knowledge (like I'm doing here)")
    print("   - Never stop learning (the field moves fast)")
    print("\nWelcome to production ML engineering!")
    print("\nFinal thought: The journey never really ends.")
    print("Every model in production teaches you something new.")
    print("Stay curious, stay humble, and happy modeling!")
    print("\n- Cazandra Aporbo (becaziam@gmail.com)")


if __name__ == "__main__":
    main()