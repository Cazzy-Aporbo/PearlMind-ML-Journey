import numpy as np
import random
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import warnings
warnings.filterwarnings('ignore')

class DataCreature:
    """AI creatures that players train and battle with"""
    def __init__(self, name, creature_type="classifier"):
        self.name = name
        self.creature_type = creature_type  # classifier, regressor, clusterer
        self.model = None
        self.level = 1
        self.experience = 0
        self.accuracy = 0.0
        self.training_history = []
        self.abilities = []
        self.health = 100
        self.energy = 100
        
    def train_on_data(self, X_train, y_train, X_test, y_test):
        """Train the creature's model and gain experience"""
        if self.creature_type == "classifier":
            if self.level <= 3:
                self.model = LogisticRegression(random_state=42)
            elif self.level <= 6:
                self.model = DecisionTreeClassifier(random_state=42)
            else:
                self.model = RandomForestClassifier(random_state=42)
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, predictions)
        
        # Gain experience based on performance
        exp_gained = int(self.accuracy * 100)
        self.experience += exp_gained
        
        if self.experience >= self.level * 200:
            self.level_up()
        
        self.training_history.append({
            'accuracy': self.accuracy,
            'experience_gained': exp_gained,
            'timestamp': datetime.now().isoformat()
        })
        
        return exp_gained
    
    def level_up(self):
        self.level += 1
        self.health += 20
        self.energy += 10
        self.experience = 0
        
        # Unlock new abilities
        abilities = [
            "Feature Selection", "Hyperparameter Tuning", "Cross Validation",
            "Ensemble Methods", "Deep Learning", "Transfer Learning"
        ]
        
        if self.level <= len(abilities):
            new_ability = abilities[self.level - 2]
            if new_ability not in self.abilities:
                self.abilities.append(new_ability)
        
        return f"Level up! {self.name} is now level {self.level} and learned {new_ability}!"

class DataDungeon:
    """Procedurally generated datasets as dungeon levels"""
    def __init__(self, difficulty=1):
        self.difficulty = difficulty
        self.dataset_type = random.choice(["classification", "regression", "clustering"])
        self.noise_level = min(0.1 + (difficulty * 0.05), 0.5)
        self.n_samples = 100 + (difficulty * 50)
        self.n_features = 2 + (difficulty // 3)
        
    def generate_dataset(self):
        """Generate a dataset based on dungeon parameters"""
        if self.dataset_type == "classification":
            X, y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_redundant=0,
                n_informative=self.n_features,
                n_clusters_per_class=1,
                flip_y=self.noise_level,
                random_state=42
            )
            
            # Create story context
            contexts = [
                "Ancient scrolls need classification by magical school",
                "Identify corrupted vs pure mana crystals",
                "Classify friendly vs hostile creatures in the forest",
                "Determine which potions are healing vs poison"
            ]
            
        elif self.dataset_type == "regression":
            X, y = make_regression(
                n_samples=self.n_samples,
                n_features=self.n_features,
                noise=self.noise_level * 10,
                random_state=42
            )
            
            contexts = [
                "Predict the power level of magical artifacts",
                "Estimate dragon age based on scale patterns",
                "Forecast mana regeneration rates",
                "Predict spell casting success probability"
            ]
        
        story_context = random.choice(contexts)
        
        return {
            'X': X,
            'y': y,
            'type': self.dataset_type,
            'story': story_context,
            'difficulty': self.difficulty,
            'features': self.n_features,
            'samples': self.n_samples
        }

class MLForgeGame:
    def __init__(self):
        self.player_name = ""
        self.level = 1
        self.experience = 0
        self.gold = 100
        self.creatures = []
        self.current_dungeon = None
        self.completed_dungeons = 0
        self.research_points = 0
        self.unlocked_algorithms = ["Linear Regression", "Logistic Regression"]
        self.save_file = "mlforge_save.json"
        
        # ML Concept Progress Tracking
        self.concept_mastery = {
            "Supervised Learning": 0,
            "Unsupervised Learning": 0,
            "Feature Engineering": 0,
            "Model Evaluation": 0,
            "Hyperparameter Tuning": 0,
            "Ensemble Methods": 0,
            "Deep Learning": 0,
            "MLOps": 0
        }
        
    def start_game(self):
        print("=" * 70)
        print("    WELCOME TO MLFORGE: THE DATA SCIENTIST'S ODYSSEY")
        print("=" * 70)
        print("\nIn a world where data holds magical power, you are a Data Mage")
        print("Train AI creatures, explore data dungeons, and master machine learning!")
        print("\nUnlike other ML games that focus on concepts, MLForge lets you:")
        print("- Code real ML models that become your battle companions")
        print("- Explore procedurally generated datasets as dungeon levels")
        print("- Battle other players' trained models in the Arena")
        print("- Build and deploy actual ML pipelines as magical spells")
        
        if os.path.exists(self.save_file):
            choice = input("\nFound existing save file. Load game? (y/n): ").lower()
            if choice == 'y':
                self.load_game()
            else:
                self.create_new_player()
        else:
            self.create_new_player()
        
        self.main_menu()
    
    def create_new_player(self):
        self.player_name = input("\nEnter your Data Mage name: ").strip()
        if not self.player_name:
            self.player_name = "Anonymous Mage"
        
        print(f"\nWelcome, {self.player_name}!")
        print("You receive your first AI creature: a basic Classifier Sprite!")
        
        # Create starter creature
        starter = DataCreature("Classifier Sprite", "classifier")
        self.creatures.append(starter)
        
        self.tutorial()
        self.save_game()
    
    def tutorial(self):
        print("\n" + "=" * 50)
        print("    TUTORIAL: YOUR FIRST DATA DUNGEON")
        print("=" * 50)
        
        print("\nA simple classification dungeon has appeared!")
