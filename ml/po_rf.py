import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, r2_score, mean_squared_error, 
    mean_absolute_error
)
import random
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Add constant random seed for reproducibility
RANDOM_SEED = 42

class PUMAOptimizer:
    def __init__(self, X, y, population_size=25, generations=50):
        self.X = np.array(X)
        self.y = np.array(y)
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = -np.inf
        self.best_scores_history = []  # Track best scores for plotting
        self.metrics_history = []  # Track all metrics for each generation
        self.best_metrics = None  # Track best metrics
        self.pCR = 0.5  # Initial crossover rate
        self.p = 0.1    # pCR adjustment rate
        
        # Split and scale data with fixed random seed
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=RANDOM_SEED
        )
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # RF parameter ranges
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 500},
            'max_depth': {'type': 'int', 'min': 5, 'max': 50},
            'min_samples_split': {'type': 'int', 'min': 2, 'max': 200},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 200}
        }
        
        # Get numerical parameters for consistent vector operations
        self.numerical_params = list(self.param_ranges.keys())
        self.num_numerical = len(self.numerical_params)
    
    def create_individual(self):
        """Create a random individual (parameter set)"""
        individual = {}
        used_combinations = set()  # Track used combinations to avoid duplicates
        
        while True:
            temp_individual = {}
            for param, range_info in self.param_ranges.items():
                range_size = range_info['max'] - range_info['min']
                # Add small random noise to avoid clustering around certain values
                noise = random.gauss(0, range_size * 0.1)  # 10% of range as standard deviation
                rand_val = random.random() * range_size + range_info['min'] + noise
                # Ensure value stays within bounds after adding noise
                rand_val = max(range_info['min'], min(range_info['max'], rand_val))
                temp_individual[param] = int(round(rand_val))
            
            # Create a tuple of parameters to check for duplicates
            param_tuple = tuple(temp_individual.values())
            if param_tuple not in used_combinations:
                individual = temp_individual
                used_combinations.add(param_tuple)
                break
            
        return individual
    
    def evaluate_individual(self, individual):
        """Evaluate fitness of an individual using Random Forest Regressor and multiple metrics"""
        try:
            # Import RandomForestRegressor
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Create Random Forest Regressor model
            model = RandomForestRegressor(
                n_estimators=individual['n_estimators'],
                max_depth=individual['max_depth'],
                min_samples_split=individual['min_samples_split'],
                min_samples_leaf=individual['min_samples_leaf'],
                n_jobs=-1,
                random_state=RANDOM_SEED
            )

            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Get predictions (flood probability from 0 to 1)
            y_pred = model.predict(self.X_test_scaled)
            y_pred = np.array(y_pred)  # Ensure numpy array
            
            # Clip predictions to ensure they're between 0 and 1
            y_pred = np.clip(y_pred, 0, 1)
            
            # Ensure y_test is numpy array
            y_test = np.array(self.y_test)
            
            # Calculate regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate R² score
            r2 = r2_score(y_test, y_pred)
            
            # Normalized Root Mean Square Error (NRMSE)
            nrmse = rmse / np.maximum(np.max(y_test) - np.min(y_test), 1e-8)
            
            # Calculate Adjusted R² (takes into account the number of predictors)
            n = len(y_test)  # number of samples
            p = self.X.shape[1]  # number of predictors
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            
            # Calculate Relative Root Mean Square Error (RRMSE)
            rrmse = rmse / np.mean(y_test) if np.mean(y_test) != 0 else rmse
            
            # Calculate Nash-Sutcliffe Efficiency (NSE)
            # NSE ranges from -inf to 1, where 1 is perfect prediction
            nse = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
            
            # Store all metrics
            current_metrics = {
                'r2': r2,
                'adjusted_r2': adjusted_r2,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'rrmse': rrmse,
                'nrmse': nrmse,
                'nse': nse
            }
            
            # Initialize best_metrics if not set
            if self.best_metrics is None:
                self.best_metrics = current_metrics.copy()
                self.metrics_history.append({
                    'generation': len(self.metrics_history),
                    'params': individual,
                    'metrics': current_metrics
                })
            else:
                # Update based on composite score
                current_composite = self.calculate_composite_score(current_metrics)
                best_composite = self.calculate_composite_score(self.best_metrics)
                
                if current_composite > best_composite:
                    self.best_metrics = current_metrics.copy()
                    self.metrics_history.append({
                        'generation': len(self.metrics_history),
                        'params': individual,
                        'metrics': current_metrics
                    })
                else:
                    # Add entry with best metrics to maintain history length
                    self.metrics_history.append({
                        'generation': len(self.metrics_history),
                        'params': individual,
                        'metrics': self.best_metrics
                    })
            
            # Calculate composite score for regression
            composite_score = self.calculate_composite_score(current_metrics)
            return float(composite_score)
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            return -np.inf
            
    def calculate_composite_score(self, metrics):
        """
        Calculate a composite score from multiple regression metrics.
        Weights are adjusted to prioritize different aspects of model performance.
        """
        # Convert error metrics to scores (higher is better)
        rmse_score = 1 / (1 + metrics['rmse'])  # Bounded between 0 and 1
        nrmse_score = 1 / (1 + metrics['nrmse'])
        rrmse_score = 1 / (1 + metrics['rrmse'])
        
        # R² and NSE are already between -inf and 1, with 1 being perfect
        # Adjust them to be between 0 and 1
        r2_score = (metrics['r2'] + 1) / 2 if metrics['r2'] <= 1 else 1
        adjusted_r2_score = (metrics['adjusted_r2'] + 1) / 2 if metrics['adjusted_r2'] <= 1 else 1
        nse_score = (metrics['nse'] + 1) / 2 if metrics['nse'] <= 1 else 1
        
        # Weighted combination of scores
        # Weights sum to 1, redistributed after removing MAPE
        weights = {
            'r2': 0.30,        # R² score (increased from 0.25)
            'adj_r2': 0.20,    # Adjusted R² (increased from 0.15)
            'rmse': 0.15,      # Root Mean Square Error
            'nrmse': 0.12,     # Normalized RMSE (increased from 0.10)
            'rrmse': 0.12,     # Relative RMSE (increased from 0.10)
            'nse': 0.11        # Nash-Sutcliffe Efficiency (increased from 0.10)
        }
        
        composite = (
            weights['r2'] * r2_score +
            weights['adj_r2'] * adjusted_r2_score +
            weights['rmse'] * rmse_score +
            weights['nrmse'] * nrmse_score +
            weights['rrmse'] * rrmse_score +
            weights['nse'] * nse_score
        )
        
        return composite
    
    def exploration_phase(self, population, fitness_values):
        """PUMA Exploration Phase"""
        new_population = []
        new_fitness = []
        used_combinations = set()  # Track parameter combinations across population
        
        for i in range(self.population_size):
            current = population[i]
            
            # Select 6 different solutions randomly
            available_indices = list(range(self.population_size))
            available_indices.remove(i)
            selected_indices = random.sample(available_indices, 6)
            a, b, c, d, e, f = [population[idx] for idx in selected_indices]
            
            # Create new solution
            attempts = 0
            max_attempts = 10  # Limit attempts to avoid infinite loop
            while attempts < max_attempts:
                new_individual = current.copy()  # Start with current solution
                
                # Ensure at least one parameter changes by selecting random parameter
                j0 = random.choice(list(self.param_ranges.keys()))
                
                for param, range_info in self.param_ranges.items():
                    # Always change j0 parameter or based on pCR probability
                    if param == j0 or random.random() <= self.pCR:
                        if random.random() < 0.5:
                            # Generate random value with added noise for better diversity
                            range_size = range_info['max'] - range_info['min']
                            noise = random.gauss(0, range_size * 0.1)
                            rand_val = random.random() * range_size + range_info['min'] + noise
                            rand_val = max(range_info['min'], min(range_info['max'], rand_val))
                            new_individual[param] = int(round(rand_val))
                        else:
                            G = 2 * random.random() - 1
                            term1 = a[param] + G * (a[param] - b[param])
                            term2 = G * (((a[param] - b[param]) - (c[param] - d[param])) + 
                                       ((c[param] - d[param]) - (e[param] - f[param])))
                            new_val = int(round(np.clip(term1 + term2, range_info['min'], range_info['max'])))
                            new_individual[param] = new_val
                
                # Check if this combination is unique
                param_tuple = tuple(new_individual.values())
                if param_tuple not in used_combinations:
                    used_combinations.add(param_tuple)
                    break
                attempts += 1
            
            # Evaluate and update
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val > fitness_values[i]:
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
                # Update pCR when no improvement (similar to MATLAB implementation)
                self.pCR = min(0.9, self.pCR + self.p)  # Cap at 0.9 to maintain some exploration
        
        return new_population, new_fitness
    
    def exploitation_phase(self, population, fitness_values):
        """PUMA Exploitation Phase"""
        Q = 0.67  # Exploitation constant
        Beta = 2  # Beta constant
        
        # Convert to list of dictionaries with cost for easier manipulation
        Sol = [{'X': pop.copy(), 'Cost': fit} for pop, fit in zip(population, fitness_values)]
        NewSol = [{'X': {}, 'Cost': -np.inf} for _ in range(self.population_size)]
        
        # Get best solution
        best_idx = np.argmax(fitness_values)
        Best = {'X': population[best_idx].copy(), 'Cost': fitness_values[best_idx]}
        
        # Calculate mean position (mbest)
        mbest = {}
        for param in self.param_ranges.keys():
            mbest[param] = np.mean([s['X'][param] for s in Sol])
        
        for i in range(self.population_size):
            # Generate random vectors
            beta1 = 2 * random.random()
            beta2 = {param: random.gauss(0, 1) for param in self.param_ranges.keys()}
            
            # Generate w and v vectors (Eq 37, 38)
            w = {param: random.gauss(0, 1) for param in self.param_ranges.keys()}
            v = {param: random.gauss(0, 1) for param in self.param_ranges.keys()}
            
            # Calculate F1 and F2 (Eq 35, 36)
            F1 = {param: random.gauss(0, 1) * np.exp(2 - i * (2/self.generations)) 
                  for param in self.param_ranges.keys()}
            F2 = {param: w[param] * (v[param]**2) * np.cos((2 * random.random()) * w[param])
                  for param in self.param_ranges.keys()}
            
            # Calculate R_1 (Eq 34)
            R_1 = 2 * random.random() - 1
            
            # Calculate S1 and S2
            S1 = {param: (2 * random.random() - 1 + random.gauss(0, 1))
                  for param in self.param_ranges.keys()}
            S2 = {param: (F1[param] * R_1 * Sol[i]['X'][param] + 
                         F2[param] * (1 - R_1) * Best['X'][param])
                  for param in self.param_ranges.keys()}
            
            # Calculate VEC
            VEC = {param: S2[param] / S1[param] for param in self.param_ranges.keys()}
            
            if random.random() <= 0.5:
                Xatack = VEC
                if random.random() > Q:
                    # Eq 32 first part
                    random_sol = random.choice(Sol)
                    for param in self.param_ranges.keys():
                        new_val = (Best['X'][param] + 
                                 beta1 * np.exp(beta2[param]) * 
                                 (random_sol['X'][param] - Sol[i]['X'][param]))
                        NewSol[i]['X'][param] = int(round(np.clip(new_val,
                                                self.param_ranges[param]['min'],
                                                self.param_ranges[param]['max'])))
                else:
                    # Eq 32 second part
                    for param in self.param_ranges.keys():
                        new_val = beta1 * Xatack[param] - Best['X'][param]
                        NewSol[i]['X'][param] = int(round(np.clip(new_val,
                                                self.param_ranges[param]['min'],
                                                self.param_ranges[param]['max'])))
            else:
                # Eq 33
                r1 = random.randint(0, self.population_size-1)
                sign = 1 if random.random() > 0.5 else -1
                for param in self.param_ranges.keys():
                    new_val = ((mbest[param] * Sol[r1]['X'][param] - 
                              sign * Sol[i]['X'][param]) / 
                             (1 + (Beta * random.random())))
                    NewSol[i]['X'][param] = int(round(np.clip(new_val,
                                            self.param_ranges[param]['min'],
                                            self.param_ranges[param]['max'])))
            
            # Evaluate new solution
            NewSol[i]['Cost'] = self.evaluate_individual(NewSol[i]['X'])
            
            # Update solution (maximizing fitness)
            if NewSol[i]['Cost'] > Sol[i]['Cost']:
                Sol[i] = NewSol[i].copy()
        
        # Convert back to separate population and fitness arrays
        new_population = [s['X'] for s in Sol]
        new_fitness = [s['Cost'] for s in Sol]
        
        return new_population, new_fitness
    
    def optimize(self):
        """Run the PUMA optimization process"""
        # Initialize parameters
        UnSelected = [1, 1]  # [Exploration, Exploitation]
        F3_Explore = 0.001
        F3_Exploit = 0.001
        Seq_Time_Explore = [1.0, 1.0, 1.0]
        Seq_Time_Exploit = [1.0, 1.0, 1.0]
        Seq_Cost_Explore = [1.0, 1.0, 1.0]
        Seq_Cost_Exploit = [1.0, 1.0, 1.0]
        Score_Explore = 0.001
        Score_Exploit = 0.001
        PF = [0.5, 0.5, 0.3]  # Parameters for F1, F2, F3
        PF_F3 = []
        Mega_Explor = 0.99
        Mega_Exploit = 0.99
        Flag_Change = 1
        
        # Reset best metrics for new optimization run
        self.best_metrics = None
        
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness_values = [self.evaluate_individual(ind) for ind in population]
        
        # Find initial best
        best_idx = np.argmax(fitness_values)
        best_individual = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        initial_best_fitness = best_fitness
        current_best_fitness = best_fitness
        
        print("\nOptimization Progress:")
        print("Gen |   R²   | Adj.R² |  RMSE  | RRMSE  | NRMSE  |  NSE   | Score")
        print("-" * 75)
        
        # Track best scores for plotting
        self.best_scores_history = [best_fitness]

        # Unexperienced Phase (First 3 iterations)
        for Iter in range(3):
            # Exploration Phase
            pop_explor, fit_explor = self.exploration_phase(population, fitness_values)
            Costs_Explor = max(fit_explor)
            
            # Exploitation Phase
            pop_exploit, fit_exploit = self.exploitation_phase(population, fitness_values)
            Costs_Exploit = max(fit_exploit)
            
            # Combine and sort solutions
            all_population = population + pop_explor + pop_exploit
            all_fitness = fitness_values + fit_explor + fit_exploit
            
            # Sort by fitness
            sorted_indices = np.argsort(all_fitness)[::-1]  # Descending order for maximization
            population = [all_population[i] for i in sorted_indices[:self.population_size]]
            fitness_values = [all_fitness[i] for i in sorted_indices[:self.population_size]]
            
            # Only update best if fitness improves
            if fitness_values[0] > current_best_fitness:
                best_individual = population[0].copy()
                best_fitness = fitness_values[0]
                current_best_fitness = best_fitness
            
            # Store best score for current iteration (keep previous best if no improvement)
            self.best_scores_history.append(current_best_fitness)
            
            # Print progress with all metrics
            latest_metrics = self.metrics_history[-1]['metrics']
            print(f"{Iter+1:3d} | {latest_metrics['r2']:6.4f} | {latest_metrics['adjusted_r2']:6.4f} | "
                  f"{latest_metrics['rmse']:6.4f} | {latest_metrics['rrmse']:6.4f} | "
                  f"{latest_metrics['nrmse']:6.4f} | {latest_metrics['nse']:6.4f} | {current_best_fitness:6.4f}")
        
        # Calculate initial scores
        Seq_Cost_Explore[0] = abs(initial_best_fitness - Costs_Explor)
        Seq_Cost_Exploit[0] = abs(initial_best_fitness - Costs_Exploit)
        
        # Add non-zero costs to PF_F3
        for cost in Seq_Cost_Explore + Seq_Cost_Exploit:
            if cost != 0:
                PF_F3.append(cost)
        
        # Calculate initial F1 and F2 scores
        F1_Explor = PF[0] * (Seq_Cost_Explore[0] / Seq_Time_Explore[0])
        F1_Exploit = PF[0] * (Seq_Cost_Exploit[0] / Seq_Time_Exploit[0])
        F2_Explor = PF[1] * (sum(Seq_Cost_Explore) / sum(Seq_Time_Explore))
        F2_Exploit = PF[1] * (sum(Seq_Cost_Exploit) / sum(Seq_Time_Exploit))
        
        # Calculate initial scores
        Score_Explore = (PF[0] * F1_Explor) + (PF[1] * F2_Explor)
        Score_Exploit = (PF[0] * F1_Exploit) + (PF[1] * F2_Exploit)
        
        # Experienced Phase
        for Iter in range(3, self.generations):
            if Score_Explore > Score_Exploit:
                # Run Exploration
                SelectFlag = 1
                population, fitness_values = self.exploration_phase(population, fitness_values)
                Count_select = UnSelected.copy()
                UnSelected[1] += 1
                UnSelected[0] = 1
                F3_Explore = PF[2]
                F3_Exploit += PF[2]
                
                # Update sequence costs for exploration
                temp_best_idx = np.argmax(fitness_values)
                temp_best_fitness = fitness_values[temp_best_idx]
                Seq_Cost_Explore[2] = Seq_Cost_Explore[1]
                Seq_Cost_Explore[1] = Seq_Cost_Explore[0]
                Seq_Cost_Explore[0] = abs(current_best_fitness - temp_best_fitness)
                
                if Seq_Cost_Explore[0] != 0:
                    PF_F3.append(Seq_Cost_Explore[0])
                
                if temp_best_fitness > current_best_fitness:
                    best_individual = population[temp_best_idx].copy()
                    best_fitness = temp_best_fitness
                    current_best_fitness = best_fitness
            else:
                # Run Exploitation
                SelectFlag = 2
                population, fitness_values = self.exploitation_phase(population, fitness_values)
                Count_select = UnSelected.copy()
                UnSelected[0] += 1
                UnSelected[1] = 1
                F3_Explore += PF[2]
                F3_Exploit = PF[2]
                
                # Update sequence costs for exploitation
                temp_best_idx = np.argmax(fitness_values)
                temp_best_fitness = fitness_values[temp_best_idx]
                Seq_Cost_Exploit[2] = Seq_Cost_Exploit[1]
                Seq_Cost_Exploit[1] = Seq_Cost_Exploit[0]
                Seq_Cost_Exploit[0] = abs(current_best_fitness - temp_best_fitness)
                
                if Seq_Cost_Exploit[0] != 0:
                    PF_F3.append(Seq_Cost_Exploit[0])
                
                if temp_best_fitness > current_best_fitness:
                    best_individual = population[temp_best_idx].copy()
                    best_fitness = temp_best_fitness
                    current_best_fitness = best_fitness
            
            # Update time sequences if phase changed
            if Flag_Change != SelectFlag:
                Flag_Change = SelectFlag
                Seq_Time_Explore[2] = Seq_Time_Explore[1]
                Seq_Time_Explore[1] = Seq_Time_Explore[0]
                Seq_Time_Explore[0] = Count_select[0]
                Seq_Time_Exploit[2] = Seq_Time_Exploit[1]
                Seq_Time_Exploit[1] = Seq_Time_Exploit[0]
                Seq_Time_Exploit[0] = Count_select[1]
            
            # Update F1 and F2 scores
            F1_Explor = PF[0] * (Seq_Cost_Explore[0] / Seq_Time_Explore[0])
            F1_Exploit = PF[0] * (Seq_Cost_Exploit[0] / Seq_Time_Exploit[0])
            F2_Explor = PF[1] * (sum(Seq_Cost_Explore) / sum(Seq_Time_Explore))
            F2_Exploit = PF[1] * (sum(Seq_Cost_Exploit) / sum(Seq_Time_Exploit))
            
            # Update Mega scores
            if Score_Explore < Score_Exploit:
                Mega_Explor = max((Mega_Explor - 0.01), 0.01)
                Mega_Exploit = 0.99
            elif Score_Explore > Score_Exploit:
                Mega_Explor = 0.99
                Mega_Exploit = max((Mega_Exploit - 0.01), 0.01)
            
            # Calculate lambda values
            lmn_Explore = 1 - Mega_Explor
            lmn_Exploit = 1 - Mega_Exploit
            
            # Update final scores
            Score_Explore = (Mega_Explor * F1_Explor) + (Mega_Explor * F2_Explor) + (lmn_Explore * (min(PF_F3) * F3_Explore))
            Score_Exploit = (Mega_Exploit * F1_Exploit) + (Mega_Exploit * F2_Exploit) + (lmn_Exploit * (min(PF_F3) * F3_Exploit))
            
            # Store best score (keep previous best if no improvement)
            self.best_scores_history.append(current_best_fitness)
            
            # Print progress with all metrics
            latest_metrics = self.metrics_history[-1]['metrics']
            print(f"{Iter+1:3d} | {latest_metrics['r2']:6.4f} | {latest_metrics['adjusted_r2']:6.4f} | "
                  f"{latest_metrics['rmse']:6.4f} | {latest_metrics['rrmse']:6.4f} | "
                  f"{latest_metrics['nrmse']:6.4f} | {latest_metrics['nse']:6.4f} | {current_best_fitness:6.4f}")
        
        # Store final results
        self.best_individual = best_individual
        self.best_score = best_fitness
        
        return self.best_individual, self.best_score

def plot_optimization_progress(scores_history):
    plt.figure(figsize=(10, 6))
    plt.plot(scores_history)
    plt.title('PUMA Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Best F1 Score')
    plt.grid(True)
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices])
    
    # Cải thiện hiển thị nhãn trục x
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45,
               ha='right')
    
    # Thêm padding để tránh cắt nhãn
    plt.tight_layout(pad=2.0)
    
    # Thêm lưới để dễ đọc
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Thêm nhãn trục
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    
    plt.show()

def main():
    try:
        # Read CSV with semicolon separator
        df = pd.read_csv('C:/Users/Admin/Downloads/prj/src/flood_training.csv', sep=';', na_values='<Null>')
        
        # Feature columns for flood prediction
        feature_columns = [
            'Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road',
            'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
            'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall'
        ]
        label_column = 'Nom'
        
        # Convert Yes/No to 1/0 for regression (probability of flood)
        df[label_column] = (df[label_column] == 'Yes').astype(float)
        
        # Replace comma with dot in numeric columns and convert to float
        for col in feature_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
        
        # Prepare data
        X = df[feature_columns].values
        y = np.array(df[label_column].values)
        
        # Handle missing values if any
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Initialize and run PUMA optimizer for RF with increased population size
        print("Starting PUMA optimization...")
        optimizer = PUMAOptimizer(X, y, population_size=25, generations=100)
        best_params, best_score = optimizer.optimize()
        
        # Plot optimization progress
        plt.figure(figsize=(12, 6))
        metrics_to_plot = ['r2', 'mse', 'mae', 'rmse', 'nrmse', 'nse']
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        
        for metric, color in zip(metrics_to_plot, colors):
            # Get metric values for each generation
            metric_values = []
            for gen_idx in range(optimizer.generations):
                gen_metrics = [m['metrics'][metric] for m in optimizer.metrics_history 
                             if m['generation'] == gen_idx]
                if gen_metrics:
                    metric_values.append(gen_metrics[-1])
            
            # Plot with appropriate scaling for different metrics
            if metric in ['mse', 'mae', 'rmse']:
                # Use log scale for error metrics
                plt.semilogy(range(len(metric_values)), metric_values, f'{color}-', label=metric.upper())
            else:
                plt.plot(range(len(metric_values)), metric_values, f'{color}-', label=metric.upper())
        
        plt.title('PUMA Optimization Progress - Regression Metrics')
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Print final results
        print("\n=== Final Results ===")
        print(f"Best Composite Score: {best_score:.4f}")
        print("\nOptimal Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
            
        # Train final model with best parameters
        final_model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            n_jobs=-1,
            random_state=RANDOM_SEED
        )
        
        # Train and evaluate on test set
        final_model.fit(optimizer.X_train_scaled, optimizer.y_train)
        y_pred = final_model.predict(optimizer.X_test_scaled)
        y_pred = np.clip(y_pred, 0, 1)  # Clip predictions between 0 and 1
        
        # Calculate and save final metrics
        y_test = np.array(optimizer.y_test)  # Ensure numpy array
        metrics_data = pd.DataFrame({
            'Metric': ['R²', 'MSE', 'MAE', 'RMSE', 'NRMSE', 'NSE'],
            'Value': [
                r2_score(y_test, y_pred),
                mean_squared_error(y_test, y_pred),
                mean_absolute_error(y_test, y_pred),
                np.sqrt(mean_squared_error(y_test, y_pred)),
                np.sqrt(mean_squared_error(y_test, y_pred)) / np.maximum(np.max(y_test) - np.min(y_test), 1e-8),
                r2_score(y_test, y_pred) # NSE is R², so we use R² for NSE
            ]
        })
        metrics_data.to_csv('po_rf_metrics.csv', index=False)
            
    except FileNotFoundError:
        print("File not found! Please check the dataset path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()