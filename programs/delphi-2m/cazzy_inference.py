import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy import stats
import shap
import warnings
warnings.filterwarnings('ignore')

class CazzyInference:
    """Advanced inference pipeline for Cazzy Aporbo model"""
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        
        # Recreate model
        from cazzy_aporbo_model import create_model
        self.model = create_model(checkpoint['config']['model_config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Load ICD-10 code mappings
        self.icd10_names = self._load_icd10_names()
        
        # Initialize explainer
        self.explainer = None
        
    def _load_icd10_names(self) -> Dict[int, str]:
        """Load ICD-10 code to disease name mappings"""
        # Simplified mapping - in practice, load from a comprehensive database
        return {
            0: "No Event",
            1: "Certain infectious diseases",
            10: "Intestinal infections",
            20: "Tuberculosis",
            50: "Viral infections",
            100: "Neoplasms",
            150: "Malignant neoplasms",
            200: "Blood diseases",
            250: "Endocrine disorders",
            280: "Mental disorders",
            300: "Nervous system diseases",
            350: "Eye diseases",
            400: "Ear diseases",
            450: "Circulatory diseases",
            500: "Respiratory diseases",
            550: "Digestive diseases",
            600: "Skin diseases",
            650: "Musculoskeletal diseases",
            700: "Genitourinary diseases",
            750: "Pregnancy complications",
            800: "Perinatal conditions",
            850: "Congenital malformations",
            900: "Symptoms and signs",
            950: "Injury and poisoning",
            1000: "External causes",
            1399: "Death"
        }
    
    def predict_individual(self, 
                          patient_history: Dict,
                          horizon_years: int = 10,
                          return_uncertainty: bool = True) -> Dict:
        """Generate predictions for an individual patient"""
        
        # Prepare input
        tokens = torch.tensor(patient_history['disease_codes'], dtype=torch.long).unsqueeze(0)
        ages = torch.tensor(patient_history['ages_days'], dtype=torch.float32).unsqueeze(0)
        
        # Optional inputs
        biomarkers = None
        if 'biomarkers' in patient_history:
            biomarkers = torch.tensor(patient_history['biomarkers'], dtype=torch.float32).unsqueeze(0)
            
        genetics = None
        if 'genetics' in patient_history:
            genetics = torch.tensor(patient_history['genetics'], dtype=torch.float32).unsqueeze(0)
        
        # Move to device
        tokens = tokens.to(self.device)
        ages = ages.to(self.device)
        if biomarkers is not None:
            biomarkers = biomarkers.to(self.device)
        if genetics is not None:
            genetics = genetics.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                tokens, ages, biomarkers, genetics, 
                return_uncertainty=return_uncertainty
            )
        
        # Process outputs
        current_age_years = ages[0, -1].item() / 365.25
        
        # Get top disease risks
        disease_probs = torch.softmax(outputs['disease_logits'][0, -1], dim=-1)
        top_k = 10
        top_probs, top_indices = torch.topk(disease_probs, top_k)
        
        top_diseases = []
        for idx, prob in zip(top_indices.cpu().numpy(), top_probs.cpu().numpy()):
            if idx in self.icd10_names:
                disease_name = self.icd10_names[idx]
            else:
                disease_name = f"ICD-10 Code {idx}"
            
            # Time to event
            time_days = outputs['time_to_event'][0, -1, 0].item()
            time_years = time_days / 365.25
            
            top_diseases.append({
                'code': int(idx),
                'name': disease_name,
                'probability': float(prob),
                'expected_time_years': float(time_years),
                'expected_age': float(current_age_years + time_years)
            })
        
        # Risk stratification
        risk_levels = outputs['risk_stratification'][0, -1].cpu().numpy()
        risk_categories = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        current_risk = risk_categories[np.argmax(risk_levels)]
        
        # Survival curves
        survival_curves = outputs['survival_curves'][0, -1].cpu().numpy()
        
        # Uncertainty quantification
        uncertainty = None
        if return_uncertainty and 'uncertainty' in outputs:
            uncertainty = outputs['uncertainty'][0, -1].mean().item()
        
        # Generate future trajectory samples
        trajectory_samples = self._sample_future_trajectories(
            tokens[0], ages[0], num_samples=100, horizon_years=horizon_years
        )
        
        return {
            'current_age': current_age_years,
            'top_disease_risks': top_diseases,
            'risk_level': current_risk,
            'risk_distribution': risk_levels.tolist(),
            'survival_curve': survival_curves.tolist(),
            'uncertainty': uncertainty,
            'trajectory_samples': trajectory_samples
        }
    
    def _sample_future_trajectories(self, 
                                   tokens: torch.Tensor,
                                   ages: torch.Tensor,
                                   num_samples: int = 100,
                                   horizon_years: int = 10) -> List[Dict]:
        """Sample multiple future trajectories"""
        
        trajectories = []
        current_age = ages[-1].item()
        max_age = current_age + (horizon_years * 365.25)
        
        for _ in range(num_samples):
            traj_tokens, traj_times = self.model.sample_trajectory(
                tokens.clone(),
                ages.clone(),
                max_age=max_age / 365.25,
                temperature=1.0
            )
            
            trajectories.append({
                'disease_codes': traj_tokens,
                'ages_days': traj_times
            })
        
        return trajectories
    
    def population_analysis(self, 
                           population_data: List[Dict],
                           stratify_by: Optional[str] = None) -> Dict:
        """Analyze disease risk across a population"""
        
        predictions = []
        
        for patient in population_data:
            pred = self.predict_individual(patient, return_uncertainty=False)
            pred['patient_id'] = patient.get('id', len(predictions))
            
            if stratify_by and stratify_by in patient:
                pred['stratum'] = patient[stratify_by]
                
            predictions.append(pred)
        
        # Aggregate statistics
        all_risks = {}
        for pred in predictions:
            for disease in pred['top_disease_risks']:
                code = disease['code']
                if code not in all_risks:
                    all_risks[code] = []
                all_risks[code].append(disease['probability'])
        
        # Calculate population-level disease prevalences
        disease_prevalences = {}
        for code, probs in all_risks.items():
            disease_prevalences[code] = {
                'mean_risk': np.mean(probs),
                'std_risk': np.std(probs),
                'median_risk': np.median(probs),
                'high_risk_fraction': np.mean(np.array(probs) > 0.5)
            }
        
        # Stratified analysis
        stratified_results = {}
        if stratify_by:
            strata = set(p.get('stratum') for p in predictions if 'stratum' in p)
            for stratum in strata:
                stratum_preds = [p for p in predictions if p.get('stratum') == stratum]
                stratified_results[stratum] = self._calculate_stratum_stats(stratum_preds)
        
        return {
            'population_size': len(population_data),
            'disease_prevalences': disease_prevalences,
            'risk_distribution': self._get_risk_distribution(predictions),
            'stratified_analysis': stratified_results,
            'predictions': predictions
        }
    
    def _calculate_stratum_stats(self, predictions: List[Dict]) -> Dict:
        """Calculate statistics for a population stratum"""
        
        risk_levels = [p['risk_level'] for p in predictions]
        risk_counts = pd.Series(risk_levels).value_counts().to_dict()
        
        # Average survival curves
        survival_curves = np.array([p['survival_curve'] for p in predictions])
        mean_survival = survival_curves.mean(axis=0)
        
        return {
            'n': len(predictions),
            'risk_distribution': risk_counts,
            'mean_survival_curve': mean_survival.tolist()
        }
    
    def _get_risk_distribution(self, predictions: List[Dict]) -> Dict:
        """Get overall risk distribution"""
        
        risk_levels = [p['risk_level'] for p in predictions]
        risk_counts = pd.Series(risk_levels).value_counts(normalize=True).to_dict()
        
        return risk_counts
    
    def explain_prediction(self, 
                          patient_history: Dict,
                          target_disease: Optional[int] = None) -> Dict:
        """Generate explanations for predictions using SHAP-like analysis"""
        
        tokens = torch.tensor(patient_history['disease_codes'], dtype=torch.long)
        ages = torch.tensor(patient_history['ages_days'], dtype=torch.float32)
        
        # Create baseline (no disease history)
        baseline_tokens = torch.zeros_like(tokens)
        baseline_ages = ages.clone()
        
        # Get predictions for actual and baseline
        with torch.no_grad():
            actual_outputs = self.model(
                tokens.unsqueeze(0).to(self.device),
                ages.unsqueeze(0).to(self.device)
            )
            
            baseline_outputs = self.model(
                baseline_tokens.unsqueeze(0).to(self.device),
                baseline_ages.unsqueeze(0).to(self.device)
            )
        
        # Calculate feature importance
        actual_probs = torch.softmax(actual_outputs['disease_logits'][0, -1], dim=-1)
        baseline_probs = torch.softmax(baseline_outputs['disease_logits'][0, -1], dim=-1)
        
        # Attribution for each past disease
        attributions = {}
        for i, code in enumerate(patient_history['disease_codes'][:-1]):
            # Create masked version
            masked_tokens = tokens.clone()
            masked_tokens[i] = 0
            
            with torch.no_grad():
                masked_outputs = self.model(
                    masked_tokens.unsqueeze(0).to(self.device),
                    ages.unsqueeze(0).to(self.device)
                )
            
            masked_probs = torch.softmax(masked_outputs['disease_logits'][0, -1], dim=-1)
            
            # Calculate attribution
            if target_disease is not None:
                attribution = (actual_probs[target_disease] - masked_probs[target_disease]).item()
            else:
                attribution = torch.norm(actual_probs - masked_probs).item()
            
            disease_name = self.icd10_names.get(code, f"ICD-10 Code {code}")
            age_at_diagnosis = ages[i].item() / 365.25
            
            attributions[i] = {
                'disease': disease_name,
                'code': int(code),
                'age_at_diagnosis': float(age_at_diagnosis),
                'attribution_score': float(attribution)
            }
        
        # Sort by attribution
        sorted_attributions = sorted(
            attributions.items(), 
            key=lambda x: abs(x[1]['attribution_score']), 
            reverse=True
        )
        
        return {
            'attributions': [attr for _, attr in sorted_attributions],
            'baseline_risk': baseline_probs[target_disease].item() if target_disease else None,
            'actual_risk': actual_probs[target_disease].item() if target_disease else None
        }
    
    def survival_analysis(self, 
                         cohort_data: List[Dict],
                         follow_up_years: int = 10) -> Dict:
        """Perform survival analysis on a cohort"""
        
        survival_data = []
        
        for patient in cohort_data:
            pred = self.predict_individual(patient, horizon_years=follow_up_years)
            
            # Extract survival information
            survival_curve = pred['survival_curve']
            
            # Check for events in actual data if available
            has_event = False
            event_time = follow_up_years
            
            if 'future_events' in patient:
                future_events = patient['future_events']
                if future_events:
                    has_event = True
                    event_time = (future_events[0]['age_days'] - patient['ages_days'][-1]) / 365.25
                    event_time = min(event_time, follow_up_years)
            
            survival_data.append({
                'patient_id': patient.get('id', len(survival_data)),
                'survival_curve': survival_curve,
                'event': has_event,
                'time': event_time,
                'risk_level': pred['risk_level']
            })
        
        # Kaplan-Meier analysis by risk group
        km_results = self._kaplan_meier_by_risk(survival_data)
        
        # Cox proportional hazards
        cox_results = self._cox_regression(survival_data, cohort_data)
        
        return {
            'n_patients': len(cohort_data),
            'follow_up_years': follow_up_years,
            'kaplan_meier': km_results,
            'cox_regression': cox_results,
            'survival_data': survival_data
        }
    
    def _kaplan_meier_by_risk(self, survival_data: List[Dict]) -> Dict:
        """Perform Kaplan-Meier analysis stratified by risk level"""
        
        km_results = {}
        
        for risk_level in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
            subset = [d for d in survival_data if d['risk_level'] == risk_level]
            
            if len(subset) > 0:
                kmf = KaplanMeierFitter()
                times = [d['time'] for d in subset]
                events = [d['event'] for d in subset]
                
                kmf.fit(times, events)
                
                km_results[risk_level] = {
                    'n': len(subset),
                    'median_survival': kmf.median_survival_time_,
                    'survival_function': kmf.survival_function_.to_dict()
                }
        
        return km_results
    
    def _cox_regression(self, survival_data: List[Dict], cohort_data: List[Dict]) -> Dict:
        """Perform Cox regression analysis"""
        
        # Prepare data for Cox regression
        df_data = []
        
        for i, (surv, patient) in enumerate(zip(survival_data, cohort_data)):
            row = {
                'time': surv['time'],
                'event': int(surv['event']),
                'age': patient['ages_days'][-1] / 365.25,
                'n_diseases': len(patient['disease_codes']),
                'risk_score': ['Very Low', 'Low', 'Medium', 'High', 'Very High'].index(surv['risk_level'])
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(df, duration_col='time', event_col='event')
        
        return {
            'coefficients': cph.params_.to_dict(),
            'hazard_ratios': np.exp(cph.params_).to_dict(),
            'concordance_index': cph.concordance_index_,
            'log_likelihood': cph.log_likelihood_
        }
    
    def visualize_predictions(self, 
                            prediction_results: Dict,
                            save_path: Optional[str] = None):
        """Create comprehensive visualizations of predictions"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Top disease risks
        ax = axes[0, 0]
        diseases = prediction_results['top_disease_risks'][:5]
        names = [d['name'][:20] for d in diseases]
        probs = [d['probability'] for d in diseases]
        
        ax.barh(names, probs)
        ax.set_xlabel('Probability')
        ax.set_title('Top 5 Disease Risks')
        ax.set_xlim([0, 1])
        
        # 2. Risk stratification
        ax = axes[0, 1]
        risk_dist = prediction_results['risk_distribution']
        categories = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        
        ax.bar(categories, risk_dist)
        ax.set_ylabel('Probability')
        ax.set_title('Risk Stratification')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Survival curve
        ax = axes[0, 2]
        survival = prediction_results['survival_curve']
        years = list(range(1, len(survival) + 1))
        
        ax.plot(years, survival, 'b-', linewidth=2)
        ax.fill_between(years, survival, alpha=0.3)
        ax.set_xlabel('Years')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Predicted Survival Curve')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # 4. Expected timeline
        ax = axes[1, 0]
        current_age = prediction_results['current_age']
        timeline_data = []
        
        for d in diseases[:5]:
            timeline_data.append({
                'disease': d['name'][:15],
                'age': d['expected_age']
            })
        
        timeline_data.sort(key=lambda x: x['age'])
        
        for i, item in enumerate(timeline_data):
            ax.barh(i, item['age'] - current_age, left=current_age, height=0.5)
            ax.text(item['age'], i, item['disease'], va='center', fontsize=8)
        
        ax.set_xlabel('Age (years)')
        ax.set_title('Expected Disease Timeline')
        ax.set_ylim([-0.5, len(timeline_data) - 0.5])
        
        # 5. Trajectory samples (if available)
        ax = axes[1, 1]
        if 'trajectory_samples' in prediction_results:
            trajectories = prediction_results['trajectory_samples'][:20]
            
            for traj in trajectories:
                ages = np.array(traj['ages_days']) / 365.25
                events = list(range(len(ages)))
                ax.plot(ages, events, 'b-', alpha=0.1)
            
            ax.set_xlabel('Age (years)')
            ax.set_ylabel('Number of Events')
            ax.set_title('Sampled Future Trajectories')
        
        # 6. Uncertainty visualization
        ax = axes[1, 2]
        if prediction_results['uncertainty'] is not None:
            uncertainty = prediction_results['uncertainty']
            
            # Create uncertainty bands for top diseases
            diseases_with_uncertainty = []
            for d in diseases[:5]:
                prob = d['probability']
                lower = max(0, prob - uncertainty)
                upper = min(1, prob + uncertainty)
                diseases_with_uncertainty.append({
                    'name': d['name'][:15],
                    'prob': prob,
                    'lower': lower,
                    'upper': upper
                })
            
            x_pos = list(range(len(diseases_with_uncertainty)))
            probs = [d['prob'] for d in diseases_with_uncertainty]
            errors = [[d['prob'] - d['lower'] for d in diseases_with_uncertainty],
                     [d['upper'] - d['prob'] for d in diseases_with_uncertainty]]
            
            ax.errorbar(x_pos, probs, yerr=errors, fmt='o', capsize=5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([d['name'] for d in diseases_with_uncertainty], rotation=45, ha='right')
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Uncertainty')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def benchmark_performance(self, test_data: List[Dict]) -> Dict:
        """Comprehensive performance benchmarking"""
        
        results = {
            'disease_prediction': {},
            'time_prediction': {},
            'risk_stratification': {},
            'survival_analysis': {}
        }
        
        all_disease_preds = []
        all_disease_true = []
        all_time_preds = []
        all_time_true = []
        all_risk_preds = []
        all_risk_true = []
        
        for patient in test_data:
            if 'future_events' not in patient or not patient['future_events']:
                continue
            
            # Get predictions
            pred = self.predict_individual(patient)
            
            # Next disease prediction
            next_event = patient['future_events'][0]
            true_disease = next_event['disease_code']
            true_time = (next_event['age_days'] - patient['ages_days'][-1]) / 365.25
            
            # Find predicted probability for true disease
            disease_probs = {d['code']: d['probability'] for d in pred['top_disease_risks']}
            pred_prob = disease_probs.get(true_disease, 0)
            
            all_disease_preds.append(pred_prob)
            all_disease_true.append(true_disease)
            
            # Time prediction
            predicted_time = pred['top_disease_risks'][0]['expected_time_years']
            all_time_preds.append(predicted_time)
            all_time_true.append(true_time)
            
            # Risk stratification
            risk_map = {'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
            pred_risk = risk_map[pred['risk_level']]
            
            # True risk based on number of future events
            num_future_events = len(patient['future_events'])
            if num_future_events < 2:
                true_risk = 0
            elif num_future_events < 5:
                true_risk = 1
            elif num_future_events < 10:
                true_risk = 2
            elif num_future_events < 15:
                true_risk = 3
            else:
                true_risk = 4
            
            all_risk_preds.append(pred_risk)
            all_risk_true.append(true_risk)
        
        # Calculate metrics
        
        # Disease prediction metrics
        if all_disease_preds:
            results['disease_prediction'] = {
                'mean_probability_true_disease': np.mean(all_disease_preds),
                'median_probability_true_disease': np.median(all_disease_preds),
                'top1_accuracy': np.mean([p > 0.5 for p in all_disease_preds]),
                'top5_accuracy': np.mean([p > 0.1 for p in all_disease_preds])
            }
        
        # Time prediction metrics
        if all_time_preds:
            time_errors = np.array(all_time_preds) - np.array(all_time_true)
            results['time_prediction'] = {
                'mean_absolute_error': np.mean(np.abs(time_errors)),
                'median_absolute_error': np.median(np.abs(time_errors)),
                'rmse': np.sqrt(np.mean(time_errors**2)),
                'correlation': np.corrcoef(all_time_preds, all_time_true)[0, 1]
            }
        
        # Risk stratification metrics
        if all_risk_preds:
            results['risk_stratification'] = {
                'accuracy': np.mean(np.array(all_risk_preds) == np.array(all_risk_true)),
                'mean_absolute_error': np.mean(np.abs(np.array(all_risk_preds) - np.array(all_risk_true))),
                'correlation': np.corrcoef(all_risk_preds, all_risk_true)[0, 1]
            }
        
        return results

def generate_synthetic_cohort(n_patients: int = 1000) -> List[Dict]:
    """Generate synthetic patient cohort for testing"""
    
    np.random.seed(42)
    cohort = []
    
    for i in range(n_patients):
        # Generate patient history
        num_past_events = np.random.randint(3, 20)
        
        disease_codes = np.random.randint(1, 1000, size=num_past_events).tolist()
        
        age_increments = np.random.exponential(500, size=num_past_events)
        ages_days = np.cumsum(age_increments + 365 * 20).tolist()  # Start at age 20
        
        # Generate some future events for validation
        num_future_events = np.random.randint(1, 10)
        future_codes = np.random.randint(1, 1000, size=num_future_events).tolist()
        future_increments = np.random.exponential(500, size=num_future_events)
        future_ages = ages_days[-1] + np.cumsum(future_increments).tolist()
        
        future_events = [
            {'disease_code': code, 'age_days': age}
            for code, age in zip(future_codes, future_ages)
        ]
        
        # Optional: add biomarkers and genetics
        biomarkers = [np.random.randn(64).tolist() for _ in range(num_past_events)]
        genetics = np.random.randn(128).tolist()
        
        patient = {
            'id': f'patient_{i}',
            'disease_codes': disease_codes,
            'ages_days': ages_days,
            'biomarkers': biomarkers,
            'genetics': genetics,
            'future_events': future_events,
            'sex': np.random.choice(['M', 'F']),
            'ethnicity': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'])
        }
        
        cohort.append(patient)
    
    return cohort

def main():
    """Main demonstration script"""
    
    print("Cazzy Aporbo Model - Advanced Health Trajectory Prediction")
    print("="*60)
    
    # Load model
    print("\nLoading trained model...")
    inference = CazzyInference('checkpoints/best_model.pt')
    
    # Generate synthetic test cohort
    print("\nGenerating synthetic test cohort...")
    cohort = generate_synthetic_cohort(n_patients=100)
    
    # Individual prediction example
    print("\n1. Individual Patient Prediction")
    print("-"*40)
    
    test_patient = cohort[0]
    prediction = inference.predict_individual(test_patient, horizon_years=10)
    
    print(f"Patient Age: {prediction['current_age']:.1f} years")
    print(f"Risk Level: {prediction['risk_level']}")
    print(f"Prediction Uncertainty: {prediction['uncertainty']:.3f}" if prediction['uncertainty'] else "")
    
    print("\nTop 5 Disease Risks:")
    for i, disease in enumerate(prediction['top_disease_risks'][:5], 1):
        print(f"  {i}. {disease['name']}: {disease['probability']:.3f} "
              f"(expected in {disease['expected_time_years']:.1f} years)")
    
    # Population analysis
    print("\n2. Population Analysis")
    print("-"*40)
    
    population_results = inference.population_analysis(
        cohort[:50], 
        stratify_by='sex'
    )
    
    print(f"Population Size: {population_results['population_size']}")
    print(f"Risk Distribution: {population_results['risk_distribution']}")
    
    if population_results['stratified_analysis']:
        print("\nStratified by Sex:")
        for stratum, stats in population_results['stratified_analysis'].items():
            print(f"  {stratum}: n={stats['n']}, "
                  f"risk distribution={stats['risk_distribution']}")
    
    # Explainability
    print("\n3. Prediction Explanation")
    print("-"*40)
    
    explanation = inference.explain_prediction(test_patient, target_disease=500)
    
    print("Most influential past diagnoses:")
    for i, attr in enumerate(explanation['attributions'][:5], 1):
        print(f"  {i}. {attr['disease']} (age {attr['age_at_diagnosis']:.1f}): "
              f"attribution={attr['attribution_score']:.4f}")
    
    # Survival analysis
    print("\n4. Survival Analysis")
    print("-"*40)
    
    survival_results = inference.survival_analysis(cohort[:50], follow_up_years=10)
    
    print(f"Cohort Size: {survival_results['n_patients']}")
    print(f"Follow-up: {survival_results['follow_up_years']} years")
    
    if survival_results['cox_regression']:
        print("\nCox Regression Results:")
        print(f"  Concordance Index: {survival_results['cox_regression']['concordance_index']:.3f}")
        print("  Hazard Ratios:")
        for var, hr in survival_results['cox_regression']['hazard_ratios'].items():
            print(f"    {var}: {hr:.3f}")
    
    # Performance benchmarking
    print("\n5. Model Performance Benchmarking")
    print("-"*40)
    
    benchmark_results = inference.benchmark_performance(cohort[50:])
    
    if benchmark_results['disease_prediction']:
        print("Disease Prediction:")
        for metric, value in benchmark_results['disease_prediction'].items():
            print(f"  {metric}: {value:.3f}")
    
    if benchmark_results['time_prediction']:
        print("\nTime Prediction:")
        for metric, value in benchmark_results['time_prediction'].items():
            print(f"  {metric}: {value:.3f}")
    
    # Visualization
    print("\n6. Generating Visualizations...")
    inference.visualize_predictions(prediction, save_path='prediction_visualization.png')
    
    print("\nAnalysis complete! Visualization saved to 'prediction_visualization.png'")

if __name__ == "__main__":
    main()
