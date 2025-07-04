import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import time
import math
import warnings
warnings.filterwarnings('ignore')

class PUMAOptimizer:
    def __init__(self, X, y, population_size=30, generations=50, cv_folds=3):
        """
        Initialize PUMA Optimizer
        
        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples]
        population_size : int, default=30
        generations : int, default=50
        cv_folds : int, default=3
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.population_size = max(10, population_size)  # Minimum 10 for proper selection
        self.generations = generations
        self.cv_folds = cv_folds
        
        # Initialize tracking variables
        self.best_individual = None
        self.best_score = -np.inf
        self.history = []
        self.feature_names = []
        
        # Validation
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of samples")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        
        # Scale data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # RF parameter ranges
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 500},
            'max_depth': {'type': 'int', 'min': 5, 'max': 50},
            'min_samples_split': {'type': 'int', 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 10},
            'max_features': {'type': 'categorical', 'values': ['sqrt', 'log2', None]}
        }
        
        # PUMA parameters
        self.dimension = len(self.param_ranges)
        self.p = 0.5  # Initial probability
        self.pf = [0.5, 0.5, 0.3]  # Weight factors for scoring
        
        print(f"PUMA Optimizer initialized:")
        print(f"  Population size: {self.population_size}")
        print(f"  Generations: {self.generations}")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Test samples: {len(self.X_test)}")
        
    def create_individual(self):
        """Create a random individual (parameter set)"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                individual[param] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'float':
                individual[param] = random.uniform(range_info['min'], range_info['max'])
            elif range_info['type'] == 'categorical':
                individual[param] = random.choice(range_info['values'])
        return individual
    
    def evaluate_individual(self, individual):
        """Evaluate fitness of an individual (higher is better)"""
        try:
            rf = RandomForestClassifier(
                n_estimators=individual['n_estimators'],
                max_depth=individual['max_depth'],
                min_samples_split=individual['min_samples_split'],
                min_samples_leaf=individual['min_samples_leaf'],
                max_features=individual['max_features'],
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            # Use cross-validation for robust evaluation
            cv_scores = cross_val_score(
                rf, self.X_train_scaled, self.y_train, 
                cv=self.cv_folds, scoring='f1', n_jobs=-1
            )
            return float(np.mean(cv_scores))
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return -1.0  # Return very low score for invalid individuals
    
    def apply_bounds(self, individual):
        """Apply parameter bounds and fix invalid values"""
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                individual[param] = int(np.clip(individual[param], 
                                              range_info['min'], range_info['max']))
            elif range_info['type'] == 'float':
                individual[param] = np.clip(individual[param], 
                                          range_info['min'], range_info['max'])
            elif range_info['type'] == 'categorical':
                # If value is invalid, choose random valid value
                if individual[param] not in range_info['values']:
                    individual[param] = random.choice(range_info['values'])
        return individual
    
    def create_population(self):
        """Create initial population"""
        population = []
        fitness_values = []
        
        print("Creating initial population...")
        for i in range(self.population_size):
            individual = self.create_individual()
            fitness = self.evaluate_individual(individual)
            population.append(individual)
            fitness_values.append(fitness)
            
            if (i + 1) % 10 == 0:
                print(f"  Created {i + 1}/{self.population_size} individuals")
        
        return population, fitness_values
    
    def sort_population(self, population, fitness_values):
        """Sort population by fitness (descending - higher is better)"""
        sorted_pairs = sorted(zip(population, fitness_values), 
                            key=lambda x: x[1], reverse=True)
        population_sorted, fitness_sorted = zip(*sorted_pairs)
        return list(population_sorted), list(fitness_sorted)
    
    def exploration_phase(self, population, fitness_values):
        """PUMA Exploration Phase - Implementation matching MATLAB version"""
        # Sort population by fitness (lower cost is better in MATLAB, but higher fitness is better here)
        population, fitness_values = self.sort_population(population, fitness_values)
        
        # Initialize parameters (Eq 28 and 29)
        pCR = 0.20
        PCR = 1 - pCR
        p = PCR / self.population_size
        
        new_population = []
        new_fitness = []
        
        for i in range(self.population_size):
            current = population[i]
            
            # Select 6 different solutions randomly (excluding current)
            available_indices = list(range(self.population_size))
            available_indices.remove(i)
            selected_indices = random.sample(available_indices, 6)
            a, b, c, d, e, f = selected_indices
            
            # Create new solution
            new_individual = {}
            
            # For each parameter
            for param, range_info in self.param_ranges.items():
                if range_info['type'] in ['int', 'float']:
                    # Eq 26: G = 2*rand-1
                    G = 2 * random.random() - 1
                    
                    # Eq 25
                    if random.random() < 0.5:
                        # Random solution within bounds
                        if range_info['type'] == 'int':
                            new_val = random.randint(range_info['min'], range_info['max'])
                        else:
                            new_val = random.uniform(range_info['min'], range_info['max'])
                    else:
                        # Complex update formula from MATLAB implementation
                        x_a = population[a][param]
                        x_b = population[b][param]
                        x_c = population[c][param]
                        x_d = population[d][param]
                        x_e = population[e][param]
                        x_f = population[f][param]
                        
                        # y = x_a + G*(x_a - x_b) + G*(((x_a - x_b)-(x_c - x_d))+((x_c - x_d)-(x_e - x_f)))
                        term1 = x_a + G * (x_a - x_b)
                        term2 = G * (((x_a - x_b) - (x_c - x_d)) + ((x_c - x_d) - (x_e - x_f)))
                        new_val = term1 + term2
                        
                        # Apply bounds
                        if range_info['type'] == 'int':
                            new_val = int(round(np.clip(new_val, range_info['min'], range_info['max'])))
                        else:
                            new_val = np.clip(new_val, range_info['min'], range_info['max'])
                            
                    # Apply crossover with probability pCR
                    if random.random() <= pCR:
                        new_individual[param] = new_val
                    else:
                        new_individual[param] = current[param]
                        
                elif range_info['type'] == 'categorical':
                    # For categorical parameters, use crossover with probability pCR
                    if random.random() <= pCR:
                        # Select value from one of the selected solutions
                        values = [population[idx][param] for idx in selected_indices]
                        new_individual[param] = random.choice(values)
                    else:
                        new_individual[param] = current[param]
            
            # Evaluate new solution
            new_fitness_val = self.evaluate_individual(new_individual)
            
            # Update solution if better (note: in MATLAB lower is better, here higher is better)
            if new_fitness_val > fitness_values[i]:
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
                pCR = min(1.0, pCR + p)  # Eq 30
        
        return new_population, new_fitness
    
    def exploitation_phase(self, population, fitness_values):
        """PUMA Exploitation Phase - Implementation matching MATLAB version"""
        new_population = []
        new_fitness = []
        
        # Sort population to get best solution
        sorted_pop, sorted_fitness = self.sort_population(population, fitness_values)
        best_solution = sorted_pop[0]  # Best solution (highest fitness)
        
        # Parameters for exploitation (matching MATLAB)
        Q = 0.67
        Beta = 2.0
        current_iter = len(self.history) + 1
        max_iter = self.generations
        
        for i in range(self.population_size):
            current = population[i]
            new_individual = {}
            
            # Generate new solution using exploitation strategy
            for param, range_info in self.param_ranges.items():
                if range_info['type'] in ['int', 'float']:
                    current_val = current[param]
                    best_val = best_solution[param]
                    
                    # MATLAB implementation
                    beta1 = 2 * random.random()
                    beta2 = np.random.randn()  # Single value for this parameter
                    
                    # Calculate F1 and F2 (Eq 35, 36)
                    F1 = random.random() * math.exp(2 - current_iter * (2/max_iter))
                    w = random.random()
                    v = random.random()
                    F2 = w * (v**2) * math.cos(2 * random.random() * w)
                    
                    # Calculate mean best (mbest)
                    mean_val = np.mean([ind[param] for ind in population])
                    
                    # R_1 (Eq 34)
                    R_1 = 2 * random.random() - 1
                    
                    # S1, S2 calculation
                    S1 = (2 * random.random() - 1 + random.gauss(0, 1))
                    S2 = F1 * R_1 * current_val + F2 * (1 - R_1) * best_val
                    
                    VEC = S2 / S1 if S1 != 0 else S2
                    
                    if random.random() <= 0.5:
                        Xatack = VEC
                        if random.random() > Q:
                            # Select random solution
                            random_sol = random.choice(population)
                            random_val = random_sol[param]
                            new_val = best_val + beta1 * math.exp(beta2) * (random_val - current_val)
                        else:
                            new_val = beta1 * Xatack - best_val
                    else:
                        # Alternative strategy (Eq 33)
                        random_sol = random.choice(population)
                        random_val = random_sol[param]
                        sign = 1 if random.random() > 0.5 else -1
                        new_val = (mean_val * random_val - (sign * current_val)) / (1 + (Beta * random.random()))
                    
                    new_individual[param] = new_val
                    
                elif range_info['type'] == 'categorical':
                    # For categorical, bias towards best solution
                    if random.random() < 0.7:  # 70% chance to use best
                        new_individual[param] = best_solution[param]
                    else:
                        new_individual[param] = current[param]
            
            # Apply bounds
            new_individual = self.apply_bounds(new_individual)
            
            # Evaluate new solution
            new_fitness_val = self.evaluate_individual(new_individual)
            
            # Update solution if better
            if new_fitness_val > fitness_values[i]:
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
        
        return new_population, new_fitness
    
    def calculate_phase_scores(self, costs_explore, costs_exploit, times_explore, times_exploit):
        """Calculate scores for exploration and exploitation phases"""
        if len(costs_explore) < 3 or len(costs_exploit) < 3:
            return 0.5, 0.5  # Default equal probability
        
        # Calculate cost differences (improvement rates)
        seq_cost_explore = [
            abs(costs_explore[i] - costs_explore[i-1]) if i > 0 else costs_explore[i]
            for i in range(min(3, len(costs_explore)))
        ]
        
        seq_cost_exploit = [
            abs(costs_exploit[i] - costs_exploit[i-1]) if i > 0 else costs_exploit[i]
            for i in range(min(3, len(costs_exploit)))
        ]
        
        # Calculate time-normalized scores
        avg_cost_explore = np.mean(seq_cost_explore)
        avg_cost_exploit = np.mean(seq_cost_exploit)
        avg_time_explore = np.mean(times_explore[-3:])
        avg_time_exploit = np.mean(times_exploit[-3:])
        
        # Calculate final scores
        f1_explore = self.pf[0] * (avg_cost_explore / max(avg_time_explore, 0.001))
        f1_exploit = self.pf[0] * (avg_cost_exploit / max(avg_time_exploit, 0.001))
        
        score_explore = f1_explore
        score_exploit = f1_exploit
        
        return score_explore, score_exploit
    
    def optimize(self):
        """Main PUMA optimization algorithm - Fixed Implementation"""
        try:
            print("\n" + "="*60)
            print("STARTING PUMA OPTIMIZATION")
            print("="*60)
            
            # Initialize population
            population, fitness_values = self.create_population()
            
            # Find initial best
            best_idx = np.argmax(fitness_values)
            self.best_individual = population[best_idx].copy()
            self.best_score = fitness_values[best_idx]
            
            print(f"\nInitial best score: {self.best_score:.4f}")
            print(f"Initial average score: {np.mean(fitness_values):.4f}")
            
            # Tracking for phase selection
            costs_explore = []
            costs_exploit = []
            times_explore = []
            times_exploit = []
            
            # Phase 1: Unexperienced phase (first 3 iterations)
            print("\n" + "-"*40)
            print("PHASE 1: UNEXPERIENCED (Learning Phase)")
            print("-"*40)
            
            for iteration in range(min(3, self.generations)):
                print(f"\nIteration {iteration + 1}/3")
                
                # Run exploration phase
                start_time = time.time()
                explore_pop, explore_fit = self.exploration_phase(population, fitness_values)
                explore_time = time.time() - start_time
                
                best_explore_score = max(explore_fit)
                costs_explore.append(best_explore_score)
                times_explore.append(explore_time)
                
                # Run exploitation phase
                start_time = time.time()
                exploit_pop, exploit_fit = self.exploitation_phase(population, fitness_values)
                exploit_time = time.time() - start_time
                
                best_exploit_score = max(exploit_fit)
                costs_exploit.append(best_exploit_score)
                times_exploit.append(exploit_time)
                
                # Select best approach for this iteration
                if best_explore_score > best_exploit_score:
                    population, fitness_values = explore_pop, explore_fit
                    used_phase = "Exploration"
                else:
                    population, fitness_values = exploit_pop, exploit_fit
                    used_phase = "Exploitation"
                
                # Update global best
                current_best_idx = np.argmax(fitness_values)
                if fitness_values[current_best_idx] > self.best_score:
                    self.best_score = fitness_values[current_best_idx]
                    self.best_individual = population[current_best_idx].copy()
                    print(f"*** NEW BEST FOUND! Score: {self.best_score:.4f} ***")
                
                print(f"Used: {used_phase}")
                print(f"Best: {self.best_score:.4f}, Avg: {np.mean(fitness_values):.4f}")
                
                # Store history
                self.history.append({
                    'iteration': iteration + 1,
                    'phase': used_phase,
                    'best_score': self.best_score,
                    'avg_score': float(np.mean(fitness_values)),
                    'best_params': self.best_individual.copy()
                })
            
            # Phase 2: Experienced phase (remaining iterations)
            if self.generations > 3:
                print("\n" + "-"*40)
                print("PHASE 2: EXPERIENCED (Adaptive Phase)")
                print("-"*40)
                
                for iteration in range(3, self.generations):
                    print(f"\nIteration {iteration + 1}/{self.generations}")
                    
                    # Calculate phase scores
                    score_explore, score_exploit = self.calculate_phase_scores(
                        costs_explore, costs_exploit, times_explore, times_exploit
                    )
                    
                    # Select phase based on scores
                    if score_explore > score_exploit:
                        start_time = time.time()
                        population, fitness_values = self.exploration_phase(population, fitness_values)
                        used_phase = "Exploration"
                        phase_time = time.time() - start_time
                        
                        best_score = max(fitness_values)
                        costs_explore.append(best_score)
                        times_explore.append(phase_time)
                    else:
                        start_time = time.time()
                        population, fitness_values = self.exploitation_phase(population, fitness_values)
                        used_phase = "Exploitation"
                        phase_time = time.time() - start_time
                        
                        best_score = max(fitness_values)
                        costs_exploit.append(best_score)
                        times_exploit.append(phase_time)
                    
                    # Update global best
                    current_best_idx = np.argmax(fitness_values)
                    if fitness_values[current_best_idx] > self.best_score:
                        self.best_score = fitness_values[current_best_idx]
                        self.best_individual = population[current_best_idx].copy()
                        print(f"*** NEW BEST FOUND! Score: {self.best_score:.4f} ***")
                    
                    print(f"Used: {used_phase} (Score: {score_explore:.3f} vs {score_exploit:.3f})")
                    print(f"Best: {self.best_score:.4f}, Avg: {np.mean(fitness_values):.4f}")
                    
                    # Store history
                    self.history.append({
                        'iteration': iteration + 1,
                        'phase': used_phase,
                        'best_score': self.best_score,
                        'avg_score': float(np.mean(fitness_values)),
                        'best_params': self.best_individual.copy()
                    })
                    
                    # Early stopping check
                    if len(self.history) > 10:
                        recent_scores = [h['best_score'] for h in self.history[-10:]]
                        if max(recent_scores) - min(recent_scores) < 0.001:
                            print("\nEarly stopping: No improvement in last 10 iterations")
                            break
            
            print("\n" + "="*60)
            print("PUMA OPTIMIZATION COMPLETED!")
            print("="*60)
            
            if self.best_individual is not None:
                print(f"\nFinal best score: {self.best_score:.4f}")
                print("\nOptimal parameters:")
                for param, value in self.best_individual.items():
                    print(f"  {param}: {value}")
            
            return self.best_individual, self.best_score
            
        except Exception as e:
            print(f"Error in optimization: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, -np.inf
    
    def evaluate_final_model(self):
        """Evaluate the final optimized model on test set"""
        if self.best_individual is None:
            print("No optimized model available!")
            return None
        
        print("\n" + "="*50)
        print("FINAL MODEL EVALUATION")
        print("="*50)
        
        # Train model with best parameters
        best_rf = RandomForestClassifier(
            n_estimators=self.best_individual['n_estimators'],
            max_depth=self.best_individual['max_depth'],
            min_samples_split=self.best_individual['min_samples_split'],
            min_samples_leaf=self.best_individual['min_samples_leaf'],
            max_features=self.best_individual['max_features'],
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        best_rf.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_pred = best_rf.predict(self.X_test_scaled)
        y_prob = best_rf.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        test_f1 = f1_score(self.y_test, y_pred)
        test_auc = roc_auc_score(self.y_test, y_prob)
        test_acc = accuracy_score(self.y_test, y_pred)
        
        print(f"\nTest Set Results:")
        print(f"  F1-Score: {test_f1:.4f}")
        print(f"  AUC-ROC:  {test_auc:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        
        # Feature importance
        feature_names = getattr(self, 'feature_names', None)
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
        
        importances = best_rf.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 5 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance[:5]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        return {
            'model': best_rf,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': self.best_individual,
            'feature_importance': feature_importance
        }
    
    def plot_convergence(self):
        """Plot optimization convergence"""
        if not self.history:
            print("No optimization history available!")
            return
        
        iterations = [h['iteration'] for h in self.history]
        best_scores = [h['best_score'] for h in self.history]
        avg_scores = [h['avg_score'] for h in self.history]
        
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(iterations, best_scores, 'b-', linewidth=2, label='Best Score')
            plt.plot(iterations, avg_scores, 'r--', linewidth=1, label='Average Score')
            plt.xlabel('Iteration')
            plt.ylabel('F1-Score')
            plt.title('PUMA Optimization Convergence')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            phases = [h['phase'] for h in self.history]
            phase_colors = ['blue' if p == 'Exploration' else 'red' for p in phases]
            plt.bar(iterations, [0.1] * len(iterations), color=phase_colors, alpha=0.7)
            plt.xlabel('Iteration')
            plt.ylabel('Phase')
            plt.title('Phase Selection Over Time')
            plt.yticks([0, 0.1], ['', 'Exploration=Blue, Exploitation=Red'])
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Install it to see convergence plots.")
            print("Data available in self.history for manual plotting.")


def main():
    """Main function to run PUMA optimization"""
    print("="*60)
    print("PUMA OPTIMIZER FOR FLOOD PREDICTION")
    print("="*60)
    
    # Update this path to your data file
    file_path = "flood_data.xlsx"  # Update with your actual file path
    
    try:
        # Read data
        df = pd.read_excel(file_path)
        
        # Define feature columns (adjust according to your Excel file)
        feature_columns = [
            'Rainfall', 'Elevation', 'Slope', 'Aspect', 'Flow_direction',
            'Flow_accumulation', 'TWI', 'Distance_to_river', 'Drainage_capacity',
            'LandCover', 'Imperviousness', 'Surface_temperature'
        ]
        
        # Define label column (adjust according to your Excel file)
        label_column = 'label_column'  # 1 = flood, 0 = no flood
        
        # Verify columns exist
        available_columns = list(df.columns)
        missing_features = [col for col in feature_columns if col not in available_columns]
        
        if missing_features:
            print(f"\nWARNING: Missing feature columns: {missing_features}")
            print(f"Available columns: {available_columns}")
            
            # Use available columns
            feature_columns = [col for col in feature_columns if col in available_columns]
            if not feature_columns:
                print("No valid feature columns found!")
                return
        
        if label_column not in available_columns:
            print(f"\nWARNING: Label column '{label_column}' not found!")
            print(f"Available columns: {available_columns}")
            return
        
        # Prepare data
        print(f"\nUsing {len(feature_columns)} features:")
        for i, col in enumerate(feature_columns):
            print(f"  {i+1}. {col}")
        
        X = df[feature_columns].values
        y = df[label_column].values
        
        # Handle missing values
        if np.isnan(X).any():
            print("\nHandling missing values...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            print("Missing values imputed with median values")
        
        # Data info
        print(f"\nDataset Info:")
        print(f"  Features shape: {X.shape}")
        print(f"  Label distribution:")
        
        unique_labels, counts = np.unique(y.astype(int), return_counts=True)
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(y)) * 100
            print(f"    Class {label}: {count} samples ({percentage:.1f}%)")
        
        # Initialize optimizer
        print(f"\nInitializing PUMA Optimizer...")
        optimizer = PUMAOptimizer(
            X, y, 
            population_size=20,  # Smaller for faster testing
            generations=30,      # Reduced for demo
            cv_folds=3
        )
        optimizer.feature_names = feature_columns
        
        # Run optimization
        start_time = time.time()
        best_params, best_score = optimizer.optimize()
        total_time = time.time() - start_time
        
        print(f"\nOptimization completed in {total_time:.2f} seconds")
        print(f"Average time per iteration: {total_time/optimizer.generations:.2f} seconds")
        
        # Evaluate final model
        if best_params is not None:
            final_results = optimizer.evaluate_final_model()
            
            # Show optimization summary
            print(f"\n" + "="*60)
            print("OPTIMIZATION SUMMARY")
            print("="*60)
            print(f"Best Cross-Validation F1-Score: {best_score:.4f}")
            print(f"Final Test F1-Score: {final_results['test_f1']:.4f}")
            print(f"Final Test AUC-ROC: {final_results['test_auc']:.4f}")
            print(f"Final Test Accuracy: {final_results['test_accuracy']:.4f}")
            
            # Try to plot convergence
            optimizer.plot_convergence()
            
        else:
            print("\nOptimization failed!")
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found!")
        print("Please update the file_path variable with the correct path to your Excel file.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()