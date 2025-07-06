# Try to import GPU libraries, fallback to CPU if not available
try:
    import cudf
    import cupy as cp
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    import xgboost as xgb
    GPU_AVAILABLE = True
except ImportError:
    from sklearn.preprocessing import StandardScaler as cuStandardScaler
    import xgboost as xgb
    GPU_AVAILABLE = False

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import random
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def check_gpu():
    if not GPU_AVAILABLE:
        print("GPU libraries not available. Using CPU instead.")
        return False
    try:
        import cupy as cp
        print("GPU is available")
        print("CUDA version:", cp.cuda.runtime.runtimeGetVersion())
        print("Number of GPUs:", cp.cuda.runtime.getDeviceCount())
        return True
    except:
        print("GPU is not available. Using CPU instead.")
        return False

class PUMAOptimizer:
    def __init__(self, X, y, population_size=30, generations=100):
        self.X = X
        self.y = y
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = -np.inf
        self.best_scores_history = []  # Track best scores for plotting
        self.convergence_threshold = 1e-6
        self.stagnation_counter = 0
        self.max_stagnation = 10
        self.has_gpu = GPU_AVAILABLE
        
        # Handle null values before converting to numpy
        if GPU_AVAILABLE and isinstance(X, cudf.DataFrame):
            X_filled = X.fillna(X.mean())
            X_np = X_filled.to_numpy()
        else:
            X_filled = X.fillna(X.mean())
            X_np = X_filled.values
        
        if GPU_AVAILABLE and isinstance(y, cudf.Series):
            y_filled = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
            y_np = y_filled.to_numpy()
        else:
            y_filled = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
            y_np = y_filled.values
        
        # Split data using numpy arrays
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.2, stratify=y_np
        )
        
        # Convert back to cuDF if GPU is available
        if self.has_gpu:
            try:
                self.X_train = cudf.DataFrame(X_train)
                self.X_test = cudf.DataFrame(X_test)
                self.y_train = cudf.Series(y_train)
                self.y_test = cudf.Series(y_test)
            except Exception as e:
                print(f"Warning: Could not convert back to cuDF: {e}")
                self.has_gpu = False
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test
        else:
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        
        # Scale data
        self.scaler = cuStandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # XGBoost parameter ranges - optimized for flood prediction with step sizes
        self.param_ranges = {
            'n_estimators': {
                'type': 'int', 
                'min': 100, 
                'max': 1000,
                'step': 50  # Bước nhảy 50 trees
            },
            'max_depth': {
                'type': 'int', 
                'min': 3, 
                'max': 50,
                'step': 1  # Bước nhảy 1 level
            },
            'learning_rate': {
                'type': 'float', 
                'min': 0.01, 
                'max': 0.3,
                'step': 0.01  # Bước nhảy 0.01
            },
            'subsample': {
                'type': 'float', 
                'min': 0.5, 
                'max': 1.0,
                'step': 0.05  # Bước nhảy 0.1
            },
            'colsample_bytree': {
                'type': 'float', 
                'min': 0.5, 
                'max': 1.0,
                'step': 0.05  # Bước nhảy 0.1
            },
            'min_child_weight': {
                'type': 'int', 
                'min': 1, 
                'max': 7,
                'step': 1  # Bước nhảy 1
            },
            'gamma': {
                'type': 'float', 
                'min': 0, 
                'max': 5,
                'step': 0.1  # Bước nhảy 0.1
            },
            'scale_pos_weight': {
                'type': 'float', 
                'min': 1, 
                'max': 10,
                'step': 0.5  # Bước nhảy 0.5
            }
        }
        
        # Get numerical parameters for consistent vector operations
        self.numerical_params = [p for p in self.param_ranges if self.param_ranges[p]['type'] in ['int', 'float']]
        self.num_numerical = len(self.numerical_params)
        
    def create_individual(self):
        """Create a random individual with defined step sizes"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                # Tính số bước có thể có
                num_steps = (range_info['max'] - range_info['min']) // range_info['step']
                # Chọn ngẫu nhiên số bước
                random_steps = random.randint(0, num_steps)
                # Tính giá trị tham số
                individual[param] = range_info['min'] + (random_steps * range_info['step'])
            elif range_info['type'] == 'float':
                # Tính số bước có thể có
                num_steps = int((range_info['max'] - range_info['min']) / range_info['step'])
                # Chọn ngẫu nhiên số bước
                random_steps = random.randint(0, num_steps)
                # Tính giá trị tham số và làm tròn để tránh lỗi floating point
                value = range_info['min'] + (random_steps * range_info['step'])
                individual[param] = round(value, 6)
        return individual

    def evaluate_individual(self, individual):
        """Evaluate fitness of an individual using cross-validation"""
        try:
            # Configure XGBoost for GPU if available
            if self.has_gpu:
                tree_method = 'gpu_hist'
                predictor = 'gpu_predictor'
            else:
                tree_method = 'hist'
                predictor = 'cpu_predictor'
            
            model = xgb.XGBClassifier(
                n_estimators=individual['n_estimators'],
                max_depth=individual['max_depth'],
                learning_rate=individual['learning_rate'],
                subsample=individual['subsample'],
                colsample_bytree=individual['colsample_bytree'],
                min_child_weight=individual['min_child_weight'],
                gamma=individual['gamma'],
                scale_pos_weight=individual['scale_pos_weight'],
                objective='binary:logistic',
                tree_method=tree_method,
                predictor=predictor,
                use_label_encoder=False
            )

            # Evaluate model
            if self.has_gpu:
                if isinstance(self.X_train_scaled, cudf.DataFrame):
                    X_train_np = self.X_train_scaled.to_numpy()
                else:
                    X_train_np = self.X_train_scaled
                
                if isinstance(self.y_train, cudf.Series):
                    y_train_np = self.y_train.to_numpy()
                else:
                    y_train_np = self.y_train
                
                X_train_val, X_val, y_train_val, y_val = train_test_split(
                    X_train_np, y_train_np, test_size=0.2, stratify=y_train_np
                )
                
                # Convert to DMatrix for faster GPU training
                dtrain = xgb.DMatrix(X_train_val, label=y_train_val)
                dval = xgb.DMatrix(X_val, label=y_val)
                
                # Train and predict
                model.fit(X_train_val, y_train_val, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
                
                if isinstance(y_pred, (cudf.Series, cudf.DataFrame)):
                    y_pred = y_pred.to_numpy()
                if isinstance(y_val, (cudf.Series, cudf.DataFrame)):
                    y_val = y_val.to_numpy()
                    
                score = f1_score(y_val, y_pred, average='binary')
            else:
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                          cv=5, scoring='f1')
                score = float(np.mean(cv_scores))
            
            return score
        except Exception as e:
            print(f"Error in evaluate_individual: {e}")
            return -np.inf

    def exploration_phase(self, population, fitness_values):
        """PUMA Exploration Phase"""
        new_population = []
        new_fitness = []
        
        # Get best solution
        best_idx = np.argmax(fitness_values)
        best_solution = population[best_idx]
        
        for i in range(self.population_size):
            current = population[i]
            new_individual = {}
            
            # Random coefficients
            r1 = random.random()
            r2 = random.random()
            r3 = random.random()
            
            # Exploration equations
            for param in self.param_ranges:
                range_info = self.param_ranges[param]
                
                # Calculate step size based on parameter type and range
                if range_info['type'] == 'int':
                    step_size = max(1, int(range_info['step'] * (1 - i/self.population_size)))
                else:
                    step_size = range_info['step'] * (1 - i/self.population_size)
                
                # Exploration movement
                if r1 < 0.33:  # First strategy
                    delta = (range_info['max'] - range_info['min']) * r2
                    new_val = best_solution[param] + delta * (2 * r3 - 1)
                elif r1 < 0.66:  # Second strategy
                    new_val = current[param] + step_size * (2 * r2 - 1)
                else:  # Third strategy
                    random_sol = random.choice(population)
                    new_val = random_sol[param] + step_size * (2 * r2 - 1)
                
                # Apply bounds and step size constraints
                if range_info['type'] == 'int':
                    # Round to nearest step
                    steps = round((new_val - range_info['min']) / range_info['step'])
                    new_val = range_info['min'] + (steps * range_info['step'])
                    # Convert to int and clip
                    new_val = int(round(np.clip(new_val, range_info['min'], range_info['max'])))
                else:
                    # Round to nearest step
                    steps = round((new_val - range_info['min']) / range_info['step'])
                    new_val = range_info['min'] + (steps * range_info['step'])
                    # Clip to bounds
                    new_val = round(np.clip(new_val, range_info['min'], range_info['max']), 6)
                
                new_individual[param] = new_val
            
            # Evaluate and update
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val > fitness_values[i]:
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
        
        return new_population, new_fitness

    def exploitation_phase(self, population, fitness_values):
        """PUMA Exploitation Phase"""
        Q = 0.67  # Exploitation constant
        Beta = 2  # Beta constant
        new_population = []
        new_fitness = []
        
        # Get best solution
        best_idx = np.argmax(fitness_values)
        best_solution = population[best_idx]
        
        # Calculate mean position
        mbest = {}
        for param in self.param_ranges:
            if self.param_ranges[param]['type'] == 'int':
                mbest[param] = int(np.mean([p[param] for p in population]))
            else:
                mbest[param] = np.mean([p[param] for p in population])
        
        for i in range(self.population_size):
            current = population[i]
            new_individual = {}
            
            beta1 = 2 * random.random()
            beta2 = np.random.randn(len(self.param_ranges))
            
            w = np.random.randn(len(self.param_ranges))  # Eq 37
            v = np.random.randn(len(self.param_ranges))  # Eq 38
            
            # Eq 35
            F1 = np.random.randn(len(self.param_ranges)) * np.exp(2 - i * (2/self.generations))
            # Eq 36
            F2 = w * np.power(v, 2) * np.cos((2 * random.random()) * w)
            
            R_1 = 2 * random.random() - 1  # Eq 34
            
            if random.random() <= 0.5:
                # Calculate S1 and S2
                S1 = 2 * random.random() - 1 + np.random.randn(len(self.param_ranges))
                
                # Convert to arrays for vector operations
                current_array = np.array([current[param] for param in self.param_ranges])
                best_array = np.array([best_solution[param] for param in self.param_ranges])
                
                S2 = F1 * R_1 * current_array + F2 * (1 - R_1) * best_array
                VEC = S2 / S1
                
                if random.random() > Q:
                    # Eq 32 first part
                    random_sol = random.choice(population)
                    random_array = np.array([random_sol[param] for param in self.param_ranges])
                    new_pos = best_array + beta1 * (np.exp(beta2)) * (random_array - current_array)
                else:
                    # Eq 32 second part
                    new_pos = beta1 * VEC - best_array
            else:
                # Eq 33
                r1 = random.randint(0, self.population_size-1)
                r1_sol = population[r1]
                r1_array = np.array([r1_sol[param] for param in self.param_ranges])
                mbest_array = np.array([mbest[param] for param in self.param_ranges])
                current_array = np.array([current[param] for param in self.param_ranges])
                
                sign = 1 if random.random() > 0.5 else -1
                new_pos = (mbest_array * r1_array - sign * current_array) / (1 + (Beta * random.random()))
            
            # Convert back to dictionary and clip values
            for j, param in enumerate(self.param_ranges):
                if self.param_ranges[param]['type'] == 'int':
                    new_individual[param] = int(round(np.clip(new_pos[j], 
                                                            self.param_ranges[param]['min'], 
                                                            self.param_ranges[param]['max'])))
                else:
                    new_individual[param] = np.clip(new_pos[j], 
                                                  self.param_ranges[param]['min'], 
                                                  self.param_ranges[param]['max'])
            
            # Evaluate and update
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val > fitness_values[i]:
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
        
        return new_population, new_fitness

    def optimize(self):
        """Main PUMA optimization algorithm"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness_values = [self.evaluate_individual(ind) for ind in population]
        
        # Initialize results tracking
        iteration_results = []
        
        # Initial best
        best_idx = np.argmax(fitness_values)
        self.best_individual = population[best_idx].copy()
        self.best_score = fitness_values[best_idx]
        initial_best_score = self.best_score
        self.best_scores_history.append(self.best_score)
        
        # Parameters for phase selection
        unselected = [1, 1]  # [Exploration, Exploitation]
        seq_time_explore = [1, 1, 1]
        seq_time_exploit = [1, 1, 1]
        seq_cost_explore = [0.1, 0.1, 0.1]
        seq_cost_exploit = [0.1, 0.1, 0.1]
        pf = [0.5, 0.5, 0.3]  # Weights for F1, F2, F3
        mega_explor = 0.99
        mega_exploit = 0.99
        f3_explore = 0
        f3_exploit = 0
        pf_f3 = [0.01]
        flag_change = 1
        
        # Save initial results
        iteration_results.append({
            'iteration': 0,
            'best_score': self.best_score,
            'best_params': self.best_individual.copy()
        })
        
        # Unexperienced Phase (first 3 iterations)
        for iteration in range(3):
            print(f"\nIteration {iteration + 1}/3 (Unexperienced Phase)")
            
            # Exploration
            pop_explore, fit_explore = self.exploration_phase(population, fitness_values)
            cost_explore = max(fit_explore)
            
            # Exploitation
            pop_exploit, fit_exploit = self.exploitation_phase(population, fitness_values)
            cost_exploit = max(fit_exploit)
            
            # Combine and select best solutions
            population = population + pop_explore + pop_exploit
            fitness_values = fitness_values + fit_explore + fit_exploit
            indices = np.argsort(fitness_values)[::-1][:self.population_size]
            population = [population[i] for i in indices]
            fitness_values = [fitness_values[i] for i in indices]
            
            # Update best
            if fitness_values[0] > self.best_score:
                self.best_score = fitness_values[0]
                self.best_individual = population[0].copy()
                self.best_scores_history.append(self.best_score)
            
            print(f"Best F1 Score: {self.best_score:.4f}")
            print(f"Population Mean Score: {np.mean(fitness_values):.4f}")
            
            # Save iteration results
            iteration_results.append({
                'iteration': iteration + 1,
                'best_score': self.best_score,
                'best_params': self.best_individual.copy(),
                'population_mean_score': np.mean(fitness_values),
                'population_min_score': np.min(fitness_values),
                'population_max_score': np.max(fitness_values),
                'phase': 'Unexperienced'
            })
        
        # Initialize sequence costs
        seq_cost_explore[0] = max(0.01, abs(initial_best_score - cost_explore))
        seq_cost_exploit[0] = max(0.01, abs(initial_best_score - cost_exploit))
        
        # Add non-zero costs to PF_F3
        for cost in seq_cost_explore + seq_cost_exploit:
            if cost > 0.01:
                pf_f3.append(cost)
        
        # Calculate initial scores
        f1_explore = pf[0] * (seq_cost_explore[0] / seq_time_explore[0])
        f1_exploit = pf[0] * (seq_cost_exploit[0] / seq_time_exploit[0])
        f2_explore = pf[1] * sum(seq_cost_explore) / sum(seq_time_explore)
        f2_exploit = pf[1] * sum(seq_cost_exploit) / sum(seq_time_exploit)
        score_explore = pf[0] * f1_explore + pf[1] * f2_explore
        score_exploit = pf[0] * f1_exploit + pf[1] * f2_exploit
        
        # Experienced Phase
        for iteration in range(3, self.generations):
            print(f"\nIteration {iteration + 1}/{self.generations}")
            
            if score_explore > score_exploit:
                # Exploration
                population, fitness_values = self.exploration_phase(population, fitness_values)
                count_select = unselected.copy()
                unselected[1] += 1
                unselected[0] = 1
                f3_explore = pf[2]
                f3_exploit += pf[2]
                phase = 'Exploration'
                
                # Update sequence costs
                if fitness_values[0] > self.best_score:
                    cost_diff = abs(self.best_score - fitness_values[0])
                    seq_cost_explore = [max(0.01, cost_diff)] + seq_cost_explore[:2]
                    if cost_diff > 0.01:
                        pf_f3.append(cost_diff)
            else:
                # Exploitation
                population, fitness_values = self.exploitation_phase(population, fitness_values)
                count_select = unselected.copy()
                unselected[0] += 1
                unselected[1] = 1
                f3_explore += pf[2]
                f3_exploit = pf[2]
                phase = 'Exploitation'
                
                # Update sequence costs
                if fitness_values[0] > self.best_score:
                    cost_diff = abs(self.best_score - fitness_values[0])
                    seq_cost_exploit = [max(0.01, cost_diff)] + seq_cost_exploit[:2]
                    if cost_diff > 0.01:
                        pf_f3.append(cost_diff)
            
            # Update best solution
            if fitness_values[0] > self.best_score:
                self.best_score = fitness_values[0]
                self.best_individual = population[0].copy()
                self.best_scores_history.append(self.best_score)
            
            print(f"Current Phase: {phase}")
            print(f"Best F1 Score: {self.best_score:.4f}")
            print(f"Population Mean Score: {np.mean(fitness_values):.4f}")
            
            # Save iteration results
            iteration_results.append({
                'iteration': iteration + 1,
                'best_score': self.best_score,
                'best_params': self.best_individual.copy(),
                'population_mean_score': np.mean(fitness_values),
                'population_min_score': np.min(fitness_values),
                'population_max_score': np.max(fitness_values),
                'phase': phase
            })
            
            # Update time sequences if phase changed
            if flag_change != (1 if score_explore > score_exploit else 2):
                flag_change = 1 if score_explore > score_exploit else 2
                seq_time_explore = [count_select[0]] + seq_time_explore[:2]
                seq_time_exploit = [count_select[1]] + seq_time_exploit[:2]
            
            # Update scores
            if score_explore < score_exploit:
                mega_explor = max(mega_explor - 0.01, 0.01)
                mega_exploit = 0.99
            elif score_explore > score_exploit:
                mega_explor = 0.99
                mega_exploit = max(mega_exploit - 0.01, 0.01)
            
            lmn_explore = 1 - mega_explor
            lmn_exploit = 1 - mega_exploit
            
            f1_explore = pf[0] * (seq_cost_explore[0] / seq_time_explore[0])
            f1_exploit = pf[0] * (seq_cost_exploit[0] / seq_time_exploit[0])
            f2_explore = pf[1] * sum(seq_cost_explore) / sum(seq_time_explore)
            f2_exploit = pf[1] * sum(seq_cost_exploit) / sum(seq_time_exploit)
            
            min_pf_f3 = min(pf_f3) if pf_f3 else 0.01
            score_explore = (mega_explor * f1_explore) + (mega_explor * f2_explore) + (lmn_explore * min_pf_f3 * f3_explore)
            score_exploit = (mega_exploit * f1_exploit) + (mega_exploit * f2_exploit) + (lmn_exploit * min_pf_f3 * f3_exploit)
        
        # Save all iteration results to CSV
        results_df = pd.DataFrame(iteration_results)
        results_df.to_csv('po_xgb_iterations.csv', index=False)
        
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
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Check GPU availability
        has_gpu = check_gpu()
        
        # Read CSV
        df = pd.read_csv('/kaggle/input/data-xgb-po/File training.csv', sep=';', na_values='<Null>')
        
        # Feature columns for flood prediction
        feature_columns = [
            'Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road',
            'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
            'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall'
        ]
        label_column = 'Nom'
        
        # Convert Yes/No to 1/0
        df[label_column] = (df[label_column] == 'Yes').astype(int)
        
        # Replace commas with dots and convert to float
        for col in feature_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
        
        # Handle missing values
        print("Checking missing values...")
        print(df[feature_columns].isnull().sum())
        
        # Fill missing values with mean
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
        
        # Prepare data
        X = df[feature_columns]
        y = df[label_column]
        
        # Convert to cuDF if GPU is available
        if has_gpu:
            try:
                X = cudf.DataFrame(X)
                y = cudf.Series(y)
                print("Successfully converted data to GPU format")
            except Exception as e:
                print(f"Warning: Could not convert to GPU format: {e}")
                print("Falling back to CPU")
                has_gpu = False
        
        # Initialize and run PUMA optimizer
        print("Starting PUMA optimization...")
        optimizer = PUMAOptimizer(
            X, y, 
            population_size=15,
            generations=30
        )
        
        best_params, best_score = optimizer.optimize()
        
        # Plot analysis
        plot_optimization_analysis(optimizer)
        
        # Print final results
        print("\n" + "="*60)
        print("FINAL OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Best F1 Score: {best_score:.6f}")
        print(f"Improvement from initial: {best_score - optimizer.best_scores_history[0]:.6f}")
        print(f"Improvement percentage: {((best_score - optimizer.best_scores_history[0]) / optimizer.best_scores_history[0] * 100):.2f}%")
        print("\nBest parameters:")
        for param, value in best_params.items():
            print(f"  {param:20}: {value}")
        
        # Train final model with best parameters
        if has_gpu:
            tree_method = 'gpu_hist'
            predictor = 'gpu_predictor'
        else:
            tree_method = 'hist'
            predictor = 'cpu_predictor'
            
        final_model = xgb.XGBClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            min_child_weight=best_params['min_child_weight'],
            gamma=best_params['gamma'],
            scale_pos_weight=best_params['scale_pos_weight'],
            objective='binary:logistic',
            tree_method=tree_method,
            predictor=predictor,
            use_label_encoder=False
        )
        
        # Train and evaluate on test set
        final_model.fit(optimizer.X_train_scaled, optimizer.y_train)
        y_pred = final_model.predict(optimizer.X_test_scaled)
        
        # Convert predictions to numpy if using GPU
        if has_gpu:
            if isinstance(y_pred, (cudf.Series, cudf.DataFrame)):
                y_pred = y_pred.to_numpy()
            if isinstance(optimizer.y_test, (cudf.Series, cudf.DataFrame)):
                y_test = optimizer.y_test.to_numpy()
        else:
            y_test = optimizer.y_test
        
        # Calculate and save final metrics
        final_metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='binary'),
            'Recall': recall_score(y_test, y_pred, average='binary'),
            'F1_Score': f1_score(y_test, y_pred, average='binary')
        }
        
        print("\n" + "="*60)
        print("FINAL MODEL PERFORMANCE ON TEST SET")
        print("="*60)
        for metric, value in final_metrics.items():
            print(f"{metric:15}: {value:.6f}")
        
        # Save metrics
        metrics_df = pd.DataFrame([final_metrics])
        metrics_df.to_csv('puma_xgb_final_metrics.csv', index=False)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Flood', 'Flood'],
                   yticklabels=['No Flood', 'Flood'])
        plt.title('Confusion Matrix - PUMA Optimized XGBoost')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('puma_xgb_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title('Feature Importances - PUMA Optimized XGBoost')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [feature_columns[i] for i in indices], 
                  rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('puma_xgb_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save model
        final_model.save_model('puma_xgb_model.json')
        
        print("\n" + "="*60)
        print("ALL RESULTS HAVE BEEN SAVED")
        print("="*60)
        print("• puma_xgb_final_metrics.csv - Final metrics")
        print("• puma_xgb_confusion_matrix.png - Confusion matrix")
        print("• puma_xgb_feature_importance.png - Feature importance")
        print("• puma_xgb_model.json - Trained model")
        
    except FileNotFoundError:
        print("File not found! Please check the dataset path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
