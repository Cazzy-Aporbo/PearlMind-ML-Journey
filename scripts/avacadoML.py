import numpy as np
import random
import json
import os
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AvocadoAI:
    """AI models that predict avocado quality and growth"""
    def __init__(self, model_type="quality_classifier"):
        self.model_type = model_type
        self.model = None
        self.accuracy = 0.0
        self.level = 1
        self.experience = 0
        self.predictions_made = 0
        self.correct_predictions = 0
        
    def train_quality_predictor(self, training_data):
        """Train AI to predict avocado ripeness and quality"""
        X = []
        y = []
        
        for avocado in training_data:
            features = [
                avocado['firmness'],
                avocado['color_score'], 
                avocado['size'],
                avocado['days_since_harvest'],
                avocado['temperature_exposure']
            ]
            X.append(features)
            y.append(avocado['quality_label'])  # 0=bad, 1=good, 2=perfect
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) > 10:  # Need minimum data to train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            if self.level <= 3:
                self.model = LogisticRegression(random_state=42, max_iter=1000)
            elif self.level <= 6:
                self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                self.model = SVC(probability=True, random_state=42)
            
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, predictions)
            
            exp_gained = int(self.accuracy * 100)
            self.experience += exp_gained
            
            if self.experience >= self.level * 300:
                self.level_up()
            
            return exp_gained, self.accuracy
        
        return 0, 0.0
    
    def predict_avocado_quality(self, avocado_features):
        """Predict if an avocado is ready for harvest/sale"""
        if self.model is None:
            return random.randint(0, 2)  # Random guess if untrained
        
        prediction = self.model.predict([avocado_features])[0]
        confidence = max(self.model.predict_proba([avocado_features])[0])
        
        self.predictions_made += 1
        return prediction, confidence
    
    def level_up(self):
        self.level += 1
        self.experience = 0
        return f"AI Model leveled up to {self.level}! Improved prediction accuracy!"

class AvocadoFarm:
    """Virtual avocado farm with ML-driven optimization"""
    def __init__(self):
        self.avocados = []
        self.harvested_avocados = []
        self.farm_level = 1
        self.farm_size = 10
        self.auto_harvest_unlocked = False
        self.weather_predictor = None
        
    def plant_avocado(self):
        """Plant a new avocado with random genetics"""
        avocado = {
            'id': len(self.avocados),
            'planted_time': datetime.now(),
            'growth_rate': random.uniform(0.5, 2.0),
            'max_quality_potential': random.uniform(0.6, 1.0),
            'firmness': random.uniform(1.0, 10.0),
            'color_score': random.uniform(1.0, 10.0),
            'size': random.uniform(50, 200),  # grams
            'days_since_harvest': 0,
            'temperature_exposure': random.uniform(15, 35),
            'is_ready': False,
            'quality_label': None
        }
        
        if len(self.avocados) < self.farm_size:
            self.avocados.append(avocado)
            return True
        return False
    
    def update_avocados(self):
        """Simulate avocado growth and ripening"""
        current_time = datetime.now()
        
        for avocado in self.avocados:
            # Calculate growth progress
            time_diff = (current_time - avocado['planted_time']).total_seconds() / 3600  # hours
            growth_progress = time_diff * avocado['growth_rate'] / 24  # days equivalent
            
            # Update avocado properties based on growth
            avocado['days_since_harvest'] = growth_progress
            
            # Determine ripeness and quality
            if growth_progress >= 3:  # Ready after 3 "days" of growth
                avocado['is_ready'] = True
                
                # Quality degradation over time
                if growth_progress > 7:
                    avocado['quality_label'] = 0  # Overripe/bad
                elif growth_progress > 5:
                    avocado['quality_label'] = 1  # Good
                else:
                    avocado['quality_label'] = 2  # Perfect
    
    def get_ready_avocados(self):
        """Get list of avocados ready for harvest decision"""
        return [a for a in self.avocados if a['is_ready']]

class AvocadoMLGame:
    def __init__(self):
        self.player_name = ""
        self.level = 1
        self.experience = 0
        self.gold = 100
        self.research_points = 0
        self.farm = AvocadoFarm()
        self.ai_models = {
            'quality_predictor': AvocadoAI('quality_classifier'),
            'growth_optimizer': AvocadoAI('growth_regressor'),
            'market_predictor': AvocadoAI('market_classifier')
        }
        self.training_data = []
        self.save_file = "avocado_ml_save.json"
        self.last_update = datetime.now()
        
        # Market simulation
        self.market_demand = random.uniform(0.5, 1.5)
        self.market_prices = {'bad': 1, 'good': 5, 'perfect': 15}
        
        # Achievements
        self.achievements = []
        self.stats = {
            'avocados_harvested': 0,
            'perfect_avocados': 0,
            'ai_predictions': 0,
            'correct_predictions': 0,
            'total_earnings': 0,
            'models_trained': 0
        }
        
        # Addictive mechanics
        self.daily_bonus_available = True
        self.streak_days = 0
        self.last_played_date = None
        
    def start_game(self):
        print("=" * 70)
        print("    AVOCADO ML: THE NEURAL HARVEST")
        print("=" * 70)
        print("\nWelcome to the world's first AI-powered avocado farming simulator!")
        print("Train machine learning models to optimize your avocado empire!")
        print("\nUnique Features:")
        print
