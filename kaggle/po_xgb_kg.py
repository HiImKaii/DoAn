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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import random
import warnings
import matplotlib.pyplot as plt
import time
import traceback
warnings.filterwarnings('ignore')

# Thiết lập hạt giống cố định để đảm bảo tái tạo kết quả
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def check_gpu():
    """Kiểm tra GPU sẵn có và tình trạng hoạt động"""
    if not GPU_AVAILABLE:
        return False
    try:
        import cupy as cp
        # Kiểm tra thêm thông tin GPU
        for i in range(cp.cuda.runtime.getDeviceCount()):
            device_props = cp.cuda.runtime.getDeviceProperties(i)
            if hasattr(device_props, 'name'):
                print(f"GPU {i}: {device_props.name}")
            if hasattr(device_props, 'totalGlobalMem'):
                print(f"GPU {i} Memory: {device_props.totalGlobalMem / 1024**3:.1f} GB")
        return True
    except Exception as e:
        return False

class PUMAOptimizer:
    def __init__(self, X, y, population_size=10, generations=100):
        self.X = X
        self.y = y
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = np.inf  # RMSE: lower is better
        self.best_scores_history = []  # Track best scores for plotting
        self.convergence_threshold = 1e-6
        self.stagnation_counter = 0
        self.max_stagnation = 10
        self.has_gpu = GPU_AVAILABLE
        self.pCR = 0.5  # Crossover probability for exploration phase
        self.p = 0.1    # Increment value for pCR adjustment
        
        # Chuẩn bị dữ liệu
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data(X, y)
        
        # Scale data
        self.scaler = cuStandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Debug scaled data
        X_scaled_check = self.X_train_scaled.values if hasattr(self.X_train_scaled, 'values') else self.X_train_scaled
        
        # Check for NaN/inf in scaled data
        if np.any(np.isnan(X_scaled_check)) or np.any(np.isinf(X_scaled_check)):
            # Clean scaled data
            self.X_train_scaled = np.nan_to_num(X_scaled_check, nan=0.0, 
                                               posinf=3.0, neginf=-3.0)  # Cap at 3 std devs
            if hasattr(self.X_test_scaled, 'values'):
                X_test_scaled_check = self.X_test_scaled.values
            else:
                X_test_scaled_check = self.X_test_scaled
            self.X_test_scaled = np.nan_to_num(X_test_scaled_check, nan=0.0, 
                                              posinf=3.0, neginf=-3.0)
        
        # XGBoost parameter ranges - optimized for regression
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 100, 'max': 1000, 'step': 10},
            'max_depth': {'type': 'int', 'min': 3, 'max': 50, 'step': 1},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.5, 'step': 0.01},
            'subsample': {'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.05},
            'colsample_bytree': {'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.05},
            'min_child_weight': {'type': 'int', 'min': 1, 'max': 50, 'step': 1},
            'gamma': {'type': 'float', 'min': 0, 'max': 5, 'step': 0.1}
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
        """Evaluate fitness of an individual using RMSE, MAE, R2 (lower RMSE is better)"""
        try:
            # Configure XGBoost for GPU if available
            tree_method = 'gpu_hist' if self.has_gpu else 'hist'
            predictor = 'gpu_predictor' if self.has_gpu else 'cpu_predictor'
            
            # Validate parameters
            for param_name, param_value in individual.items():
                if param_name == '_metrics':
                    continue
                if np.isnan(param_value) or np.isinf(param_value):
                    return np.inf
            
            model = xgb.XGBRegressor(
                n_estimators=int(individual['n_estimators']),
                max_depth=int(individual['max_depth']),
                learning_rate=float(individual['learning_rate']),
                subsample=float(individual['subsample']),
                colsample_bytree=float(individual['colsample_bytree']),
                min_child_weight=int(individual['min_child_weight']),
                gamma=float(individual['gamma']),
                tree_method=tree_method,
                predictor=predictor,
                random_state=RANDOM_SEED,
                verbosity=0  # Suppress XGBoost warnings
            )

            # Evaluate model
            if self.has_gpu:
                X_train_np = self._to_numpy(self.X_train_scaled)
                y_train_np = self._to_numpy(self.y_train)
                
                # Check for NaN/inf in data
                if np.any(np.isnan(X_train_np)) or np.any(np.isinf(X_train_np)) or \
                   np.any(np.isnan(y_train_np)) or np.any(np.isinf(y_train_np)):
                    return np.inf
                
                X_train_val, X_val, y_train_val, y_val = train_test_split(
                    X_train_np, y_train_np, test_size=0.2, random_state=RANDOM_SEED
                )
                
                model.fit(X_train_val, y_train_val, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
                
                y_pred = self._to_numpy(y_pred)
                y_val = self._to_numpy(y_val)
            else:
                # Check for NaN/inf in data
                X_check = self.X_train_scaled.values if hasattr(self.X_train_scaled, 'values') else self.X_train_scaled
                y_check = self.y_train.values if hasattr(self.y_train, 'values') else self.y_train
                
                if np.any(np.isnan(X_check)) or np.any(np.isinf(X_check)) or \
                   np.any(np.isnan(y_check)) or np.any(np.isinf(y_check)):
                    return np.inf
                
                # Use a single train-validation split for consistent metrics
                X_train_val, X_val, y_train_val, y_val = train_test_split(
                    self.X_train_scaled, self.y_train, test_size=0.2, random_state=RANDOM_SEED
                )
                
                model.fit(X_train_val, y_train_val)
                y_pred = model.predict(X_val)
            
            # Check predictions for NaN/inf
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                individual['_metrics'] = {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf}
                return np.inf
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # Check if metrics are valid
            if np.isnan(rmse) or np.isinf(rmse) or rmse <= 0:
                individual['_metrics'] = {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf}
                return np.inf
            
            # Store metrics for later use
            individual['_metrics'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            return rmse  # Return RMSE as primary fitness (lower is better)
            
        except Exception as e:
            individual['_metrics'] = {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf}
            return np.inf  # Return high RMSE for failed evaluations

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
            
            # Handle case where population size is too small
            if len(available_indices) < 6:
                # If not enough individuals, repeat some indices with replacement
                selected_indices = random.choices(available_indices, k=6)
            else:
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
                            new_individual[param] = self._apply_bounds_and_type(param, rand_val)
                        else:
                            # PUMA exploration equation with proper vector operations
                            G = 2 * random.random() - 1
                            term1 = a[param] + G * (a[param] - b[param])
                            term2 = G * (((a[param] - b[param]) - (c[param] - d[param])) + 
                                       ((c[param] - d[param]) - (e[param] - f[param])))
                            new_val = term1 + term2
                            new_individual[param] = self._apply_bounds_and_type(param, new_val)
                
                # Check if this combination is unique
                # Only use parameter values, exclude _metrics
                param_values = tuple(v for k, v in new_individual.items() if k != '_metrics')
                if param_values not in used_combinations:
                    used_combinations.add(param_values)
                    break
                attempts += 1
            
            # Evaluate and update (RMSE: lower is better)
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val < fitness_values[i]:
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
                # Update pCR when no improvement
                self.pCR = min(0.9, self.pCR + self.p)
        
        return new_population, new_fitness

    def _apply_bounds_and_type(self, param, new_val):
        """Áp dụng giới hạn và xử lý kiểu dữ liệu cho tham số"""
        range_info = self.param_ranges[param]
        clipped_val = np.clip(new_val, range_info['min'], range_info['max'])
        
        if range_info['type'] == 'int':
            return int(round(clipped_val))
        else:
            return round(clipped_val, 6)

    def exploitation_phase(self, population, fitness_values):
        """PUMA Exploitation Phase"""
        Q = 0.67  # Exploitation constant
        Beta = 2  # Beta constant
        
        # Get best solution (RMSE: lower is better)
        best_idx = np.argmin(fitness_values)
        Best = {'X': population[best_idx].copy(), 'Cost': fitness_values[best_idx]}
        
        # Convert to list of dictionaries with cost for easier manipulation  
        Sol = [{'X': pop.copy(), 'Cost': fit} for pop, fit in zip(population, fitness_values)]
        NewSol = [{'X': {}, 'Cost': np.inf} for _ in range(self.population_size)]
        
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
            
            # Calculate VEC - simplified without division by zero protection for cleaner logic
            VEC = {}
            for param in self.param_ranges.keys():
                if abs(S1[param]) > 1e-10:  # Avoid division by zero
                    VEC[param] = S2[param] / S1[param]
                else:
                    VEC[param] = S2[param] / (1e-10 if S1[param] >= 0 else -1e-10)
            
            if random.random() <= 0.5:
                Xatack = VEC
                if random.random() > Q:
                    # Eq 32 first part
                    random_sol = random.choice(Sol)
                    for param in self.param_ranges.keys():
                        new_val = (Best['X'][param] + 
                                 beta1 * np.exp(beta2[param]) * 
                                 (random_sol['X'][param] - Sol[i]['X'][param]))
                        NewSol[i]['X'][param] = self._apply_bounds_and_type(param, new_val)
                else:
                    # Eq 32 second part
                    for param in self.param_ranges.keys():
                        new_val = beta1 * Xatack[param] - Best['X'][param]
                        NewSol[i]['X'][param] = self._apply_bounds_and_type(param, new_val)
            else:
                # Eq 33
                r1 = random.randint(0, self.population_size-1)
                sign = 1 if random.random() > 0.5 else -1
                for param in self.param_ranges.keys():
                    new_val = ((mbest[param] * Sol[r1]['X'][param] - 
                              sign * Sol[i]['X'][param]) / 
                             (1 + (Beta * random.random())))
                    NewSol[i]['X'][param] = self._apply_bounds_and_type(param, new_val)
            
            # Evaluate new solution
            NewSol[i]['Cost'] = self.evaluate_individual(NewSol[i]['X'])
            
            # Update solution (RMSE: lower is better)
            if NewSol[i]['Cost'] < Sol[i]['Cost']:
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
        
        # Initialize results tracking
        iteration_results = []
        
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness_values = [self.evaluate_individual(ind) for ind in population]
        
        # Find initial best (RMSE: lower is better)
        best_idx = np.argmin(fitness_values)
        best_individual = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        initial_best_fitness = best_fitness
        current_best_fitness = best_fitness
        
        # Store initial results
        self.best_individual = best_individual
        self.best_score = best_fitness
        self.best_scores_history = [best_fitness]
        
        iteration_results.append({
            'iteration': 0,
            'rmse': self.best_score,
            'mae': self.best_individual.get('_metrics', {}).get('mae', 0),
            'r2': self.best_individual.get('_metrics', {}).get('r2', 0),
            **{param: value for param, value in self.best_individual.items() if param != '_metrics'}
        })
        
        print("\nOptimization Progress:")
        print("Gen | Best RMSE | Mean RMSE | Phase")
        print("-" * 40)
        
        # Unexperienced Phase (First 3 iterations)
        for Iter in range(3):
            # Exploration Phase
            pop_explor, fit_explor = self.exploration_phase(population, fitness_values)
            Costs_Explor = min(fit_explor) if fit_explor else np.inf  # Best RMSE from exploration
            
            # Exploitation Phase
            pop_exploit, fit_exploit = self.exploitation_phase(population, fitness_values)
            Costs_Exploit = min(fit_exploit) if fit_exploit else np.inf  # Best RMSE from exploitation
            
            # Combine and sort solutions
            all_population = population + pop_explor + pop_exploit
            all_fitness = fitness_values + fit_explor + fit_exploit
            
            # Sort by fitness (ascending for RMSE minimization)
            sorted_indices = np.argsort(all_fitness)
            population = [all_population[i] for i in sorted_indices[:self.population_size]]
            fitness_values = [all_fitness[i] for i in sorted_indices[:self.population_size]]
            
            # Only update best if fitness improves (RMSE: lower is better)
            if fitness_values[0] < current_best_fitness:
                best_individual = population[0].copy()
                best_fitness = fitness_values[0]
                current_best_fitness = best_fitness
                self.best_individual = best_individual
                self.best_score = best_fitness
            
            # Store best score for current iteration
            self.best_scores_history.append(current_best_fitness)
            
            # Print progress
            mean_fitness = np.mean(fitness_values)
            print(f"{Iter+1:3d} | {current_best_fitness:9.6f} | {mean_fitness:9.6f} | Mixed")
            
            # Save iteration results
            iteration_results.append({
                'iteration': Iter + 1,
                'rmse': current_best_fitness,
                'mae': self.best_individual.get('_metrics', {}).get('mae', 0),
                'r2': self.best_individual.get('_metrics', {}).get('r2', 0),
                **{param: value for param, value in self.best_individual.items() if param != '_metrics'}
            })
        
        # Calculate initial scores (convert RMSE improvements to positive values)
        Seq_Cost_Explore[0] = max(0, initial_best_fitness - Costs_Explor)
        Seq_Cost_Exploit[0] = max(0, initial_best_fitness - Costs_Exploit)
        
        # Add non-zero costs to PF_F3
        for cost in Seq_Cost_Explore + Seq_Cost_Exploit:
            if cost != 0:
                PF_F3.append(cost)
        
        # Initialize PF_F3 if empty
        if not PF_F3:
            PF_F3 = [0.001]
        
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
            previous_best = current_best_fitness
            
            if Score_Explore > Score_Exploit:
                # Run Exploration
                SelectFlag = 1
                new_population, new_fitness = self.exploration_phase(population, fitness_values)
                Count_select = UnSelected.copy()
                UnSelected[1] += 1
                UnSelected[0] = 1
                F3_Explore = PF[2]
                F3_Exploit += PF[2]
                phase_name = "Exploration"
                
                # Update sequence costs for exploration
                temp_best_fitness = min(new_fitness) if new_fitness else np.inf
                Seq_Cost_Explore[2] = Seq_Cost_Explore[1]
                Seq_Cost_Explore[1] = Seq_Cost_Explore[0]
                Seq_Cost_Explore[0] = max(0, current_best_fitness - temp_best_fitness)
                
                if Seq_Cost_Explore[0] != 0:
                    PF_F3.append(Seq_Cost_Explore[0])
                
                # Update population
                population = new_population
                fitness_values = new_fitness
                
            else:
                # Run Exploitation
                SelectFlag = 2
                new_population, new_fitness = self.exploitation_phase(population, fitness_values)
                Count_select = UnSelected.copy()
                UnSelected[0] += 1
                UnSelected[1] = 1
                F3_Explore += PF[2]
                F3_Exploit = PF[2]
                phase_name = "Exploitation"
                
                # Update sequence costs for exploitation
                temp_best_fitness = min(new_fitness) if new_fitness else np.inf
                Seq_Cost_Exploit[2] = Seq_Cost_Exploit[1]
                Seq_Cost_Exploit[1] = Seq_Cost_Exploit[0]
                Seq_Cost_Exploit[0] = max(0, current_best_fitness - temp_best_fitness)
                
                if Seq_Cost_Exploit[0] != 0:
                    PF_F3.append(Seq_Cost_Exploit[0])
                
                # Update population
                population = new_population
                fitness_values = new_fitness
            
            # Update best if improved (RMSE: lower is better)
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < current_best_fitness:
                best_individual = population[best_idx].copy()
                best_fitness = fitness_values[best_idx]
                current_best_fitness = best_fitness
                self.best_individual = best_individual
                self.best_score = best_fitness
            
            # Update time sequences if phase changed
            if Flag_Change != SelectFlag:
                Flag_Change = SelectFlag
                Seq_Time_Explore[2] = Seq_Time_Explore[1]
                Seq_Time_Explore[1] = Seq_Time_Explore[0]
                Seq_Time_Explore[0] = Count_select[0]
                Seq_Time_Exploit[2] = Seq_Time_Exploit[1]
                Seq_Time_Exploit[1] = Seq_Time_Exploit[0]
                Seq_Time_Exploit[0] = Count_select[1]
            
            # Update F1 and F2 scores (avoid division by zero)
            F1_Explor = PF[0] * (Seq_Cost_Explore[0] / max(Seq_Time_Explore[0], 1e-10))
            F1_Exploit = PF[0] * (Seq_Cost_Exploit[0] / max(Seq_Time_Exploit[0], 1e-10))
            F2_Explor = PF[1] * (sum(Seq_Cost_Explore) / max(sum(Seq_Time_Explore), 1e-10))
            F2_Exploit = PF[1] * (sum(Seq_Cost_Exploit) / max(sum(Seq_Time_Exploit), 1e-10))
            
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
            min_pf_f3 = min(PF_F3) if PF_F3 else 0.001
            Score_Explore = (Mega_Explor * F1_Explor) + (Mega_Explor * F2_Explor) + (lmn_Explore * (min_pf_f3 * F3_Explore))
            Score_Exploit = (Mega_Exploit * F1_Exploit) + (Mega_Exploit * F2_Exploit) + (lmn_Exploit * (min_pf_f3 * F3_Exploit))
            
            # Store best score
            self.best_scores_history.append(current_best_fitness)
            
            # Print progress
            mean_fitness = np.mean(fitness_values)
            improvement = previous_best - current_best_fitness
            print(f"{Iter+1:3d} | {current_best_fitness:9.6f} | {mean_fitness:9.6f} | {phase_name}")
            
            # Save iteration results
            iteration_results.append({
                'iteration': Iter + 1,
                'rmse': current_best_fitness,
                'mae': self.best_individual.get('_metrics', {}).get('mae', 0),
                'r2': self.best_individual.get('_metrics', {}).get('r2', 0),
                **{param: value for param, value in self.best_individual.items() if param != '_metrics'}
            })
            
            # Early stopping check
            if improvement < self.convergence_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            
            if self.stagnation_counter >= self.max_stagnation:
                break
        
        # Save all iteration results to CSV and Excel
        results_df = pd.DataFrame(iteration_results)
        
        # Reorder columns: iteration, rmse, mae, r2, then parameters
        param_cols = [col for col in results_df.columns if col not in ['iteration', 'rmse', 'mae', 'r2']]
        column_order = ['iteration', 'rmse', 'mae', 'r2'] + param_cols
        results_df = results_df[column_order]
        
        # Save to CSV and Excel
        results_df.to_csv('po_xgb_iterations.csv', index=False)
        results_df.to_excel('po_xgb_iterations.xlsx', index=False)
        
        return self.best_individual, self.best_score

    def _prepare_data(self, X, y):
        """Xử lý và chuẩn bị dữ liệu cho GPU/CPU (Regression)"""
        # Xử lý giá trị null và chuyển đổi về numpy
        X_filled = X.fillna(X.mean())
        y_filled = y.fillna(y.mean())  # For regression, use mean instead of mode
        
        if GPU_AVAILABLE and isinstance(X, cudf.DataFrame):
            X_np = X_filled.to_numpy()
            y_np = y_filled.to_numpy()
        else:
            X_np = X_filled.values if hasattr(X_filled, 'values') else X_filled
            y_np = y_filled.values if hasattr(y_filled, 'values') else y_filled
        
        # Clean any remaining NaN/inf
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        y_np = np.nan_to_num(y_np, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        
        # Chia dữ liệu (no stratify for regression)
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.2, random_state=RANDOM_SEED
        )
        
        
        # Chuyển về cuDF nếu GPU có sẵn
        if self.has_gpu:
            try:
                return (cudf.DataFrame(X_train), cudf.DataFrame(X_test), 
                       cudf.Series(y_train), cudf.Series(y_test))
            except Exception as e:
                self.has_gpu = False
        
        return X_train, X_test, y_train, y_test

    def _to_numpy(self, data):
        """Chuyển đổi dữ liệu về numpy format"""
        if isinstance(data, (cudf.Series, cudf.DataFrame)):
            return data.to_numpy()
        return data


def fill_missing_with_neighbors(series):
    """
    Fill missing values with the average of immediate neighbors (above and below)
    If neighbors are not available, use column mean as fallback
    """
    series = series.copy()
    missing_indices = series.isnull()
    
    if not missing_indices.any():
        return series  # No missing values
    
    print(f"Filling {missing_indices.sum()} missing values using neighbor averaging...")
    
    for idx in series.index[missing_indices]:
        neighbors = []
        
        # Get value above (previous row)
        if idx > 0:
            above_val = series.iloc[idx - 1] if idx - 1 in series.index else None
            if pd.notna(above_val):
                neighbors.append(above_val)
        
        # Get value below (next row)
        if idx < len(series) - 1:
            below_val = series.iloc[idx + 1] if idx + 1 in series.index else None
            if pd.notna(below_val):
                neighbors.append(below_val)
        
        # Fill with neighbor average or fallback to column mean
        if neighbors:
            series.iloc[idx] = np.mean(neighbors)
        else:
            # Fallback to column mean (excluding NaN values)
            valid_values = series.dropna()
            if len(valid_values) > 0:
                series.iloc[idx] = valid_values.mean()
            else:
                # If all values are NaN, use 0 as last resort
                series.iloc[idx] = 0.0
    
    return series


def plot_optimization_progress(optimizer):
    """Plot optimization progress similar to PSO style."""
    if not hasattr(optimizer, 'best_scores_history') or len(optimizer.best_scores_history) < 2:
        return
    
    # Đọc dữ liệu từ file CSV
    try:
        iteration_results = pd.read_csv('po_xgb_iterations.csv')
        iterations = iteration_results['iteration'].values[1:]  # Skip iteration 0
        rmse_scores = iteration_results['rmse'].values[1:]
        mae_scores = iteration_results['mae'].values[1:]
        r2_scores = iteration_results['r2'].values[1:]
    except:
        # Fallback to best_scores_history
        iterations = list(range(1, len(optimizer.best_scores_history)))
        rmse_scores = optimizer.best_scores_history[1:]
        mae_scores = [x * 0.8 for x in rmse_scores]  # Approximate MAE from RMSE
        r2_scores = [1 - (x / rmse_scores[0])**2 for x in rmse_scores]  # Approximate R²
    
    # Ensure best_rmse is non-increasing (best RMSE should never increase)
    for i in range(1, len(rmse_scores)):
        if rmse_scores[i] > rmse_scores[i-1]:
            rmse_scores[i] = rmse_scores[i-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Best RMSE progression
    axes[0, 0].plot(iterations, rmse_scores, 'b-', label='Best RMSE')
    axes[0, 0].set_title('PUMA Optimization Progress')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Best RMSE')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # Plot 2: R² Progression with Best RMSE
    axes[0, 1].plot(iterations, r2_scores, 'g-', linewidth=2)
    axes[0, 1].set_title('R² Progression with Best RMSE')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].grid(True)
    
    # Plot 3: MAE Progression with Best RMSE
    axes[1, 0].plot(iterations, mae_scores, 'r-', label='MAE', linewidth=2)
    axes[1, 0].set_title('MAE Progression with Best RMSE')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # Plot 4: All metrics together (normalized)
    ax4 = axes[1, 1]
    
    # Normalize values for better visualization
    norm_rmse = [(x - min(rmse_scores)) / (max(rmse_scores) - min(rmse_scores) + 1e-10) if max(rmse_scores) > min(rmse_scores) else x for x in rmse_scores]
    norm_r2 = [(x - min(r2_scores)) / (max(r2_scores) - min(r2_scores) + 1e-10) if max(r2_scores) > min(r2_scores) else x for x in r2_scores]
    norm_mae = [(x - min(mae_scores)) / (max(mae_scores) - min(mae_scores) + 1e-10) if max(mae_scores) > min(mae_scores) else x for x in mae_scores]
    
    ax4.plot(iterations, norm_rmse, 'b-', label='Best RMSE', linewidth=2)
    ax4.plot(iterations, norm_r2, 'g-', label='R²', linewidth=2)
    ax4.plot(iterations, norm_mae, 'r-', label='MAE', linewidth=2)
    
    ax4.set_title('Normalized Metrics Progression')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Normalized Value')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('puma_optimization_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"- Biểu đồ: puma_optimization_results.png")
    plt.show()


def export_detailed_results_to_excel(optimizer, filename='po_xgb_detailed_results.xlsx'):
    """Export detailed results to Excel with multiple sheets - Simplified version"""
    try:
        # Read iteration results
        df = pd.read_csv('po_xgb_iterations.csv')
        
        # Simple export with pandas ExcelWriter
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: All iterations
            df.to_excel(writer, sheet_name='All_Iterations', index=False)
            
            # Sheet 2: Best result only
            best_idx = df['rmse'].idxmin()
            best_df = df.loc[[best_idx]]
            best_df.to_excel(writer, sheet_name='Best_Result', index=False)
            
            # Sheet 3: Summary stats
            metrics = ['rmse', 'mae', 'r2']
            summary = pd.DataFrame({
                'Metric': metrics,
                'Best': [df[m].min() if m != 'r2' else df[m].max() for m in metrics],
                'Mean': [df[m].mean() for m in metrics],
                'Std': [df[m].std() for m in metrics]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)
        
 
    except Exception as e:
        return


def _load_and_preprocess_data():
    """Tải và tiền xử lý dữ liệu cho bài toán hồi quy"""
    df = pd.read_csv('/kaggle/input/data-xgb-po/File training.csv', sep=';', na_values='<Null>')
    
    feature_columns = [
        'Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road',
        'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
        'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall'
    ]
    label_column = 'Nom'
    
    # Check if target column exists and has valid data
    if label_column not in df.columns:
        raise ValueError(f"Target column '{label_column}' not found in data!")
    
    
    # For regression, treat the target as continuous values
    # If the data is categorical (Yes/No), convert to numeric for regression
    if df[label_column].dtype == 'object':
        # Try to convert strings like "1.5", "2.0" etc to numeric
        df[label_column] = pd.to_numeric(df[label_column], errors='coerce')
        
        # If still all NaN after numeric conversion, try label encoding
        if df[label_column].isnull().all():
            print("Target column appears to be categorical, applying label encoding...")
            le = LabelEncoder()
            # Convert back to original and encode
            original_target = pd.read_csv('/kaggle/input/data-xgb-po/File training.csv', sep=';', na_values='<Null>')[label_column]
            # Remove null values for encoding
            valid_mask = ~original_target.isnull()
            if valid_mask.any():
                df.loc[valid_mask, label_column] = le.fit_transform(original_target[valid_mask].astype(str))
            else:
                raise ValueError("Target column has no valid values!")
    
    # Replace commas with dots and convert to float
    for col in feature_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
    
    # Fill features using neighbor averaging
    for col in feature_columns:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = fill_missing_with_neighbors(df[col])
                # Fallback to mean if still missing
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
    
    # Fill target using neighbor averaging
    if df[label_column].isnull().any():
        df[label_column] = fill_missing_with_neighbors(df[label_column])
        # Fallback to mean if still missing
        if df[label_column].isnull().any():
            df[label_column] = df[label_column].fillna(df[label_column].mean())
    
    # Verify target has valid values
    if df[label_column].isnull().all() or df[label_column].std() == 0:
        raise ValueError("Target variable has no variation or all values are missing!")
    
    X = df[feature_columns]
    y = df[label_column]
    
    
    # Check for any remaining NaN/inf values
    nan_features = X.isnull().sum().sum()
    inf_features = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    nan_target = y.isnull().sum()
    inf_target = np.isinf(y).sum()
    

    
    if nan_features > 0 or inf_features > 0 or nan_target > 0 or inf_target > 0:
        # Additional cleaning
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())
    
    # Convert to cuDF if GPU is available
    has_gpu = check_gpu()
    if has_gpu:
        try:
            X = cudf.DataFrame(X)
            y = cudf.Series(y)
        except Exception as e:
            return
    
    return X, y


def _run_optimization(X, y):
    optimizer = PUMAOptimizer(X, y, population_size=15, generations=30)
    best_params, best_score = optimizer.optimize()
    return optimizer, best_params, best_score


def main():
    try:
        # Load and preprocess data
        X, y = _load_and_preprocess_data()
        
        # Run optimization
        optimizer, best_params, best_score = _run_optimization(X, y)

        # Plot optimization progress
        plot_optimization_progress(optimizer)
        
        # Export detailed results to Excel
        export_detailed_results_to_excel(optimizer)
        
        if hasattr(optimizer, 'best_scores_history') and len(optimizer.best_scores_history) > 0:
            initial_score = optimizer.best_scores_history[0]
            improvement = initial_score - best_score  # For RMSE: lower is better
            improvement_pct = (improvement / initial_score * 100) if initial_score > 0 else 0
        for param, value in best_params.items():
            if param != '_metrics':
                print(f"  {param:20}: {value}")
        

        # Save best parameters
        params_df = pd.DataFrame([{k: v for k, v in best_params.items() if k != '_metrics'}])
        params_df.to_csv('po_xgb_best_params.csv', index=False)
        
        # Save final metrics
        if '_metrics' in best_params:
            metrics_df = pd.DataFrame([best_params['_metrics']])
            metrics_df.to_csv('po_xgb_final_metrics.csv', index=False)
        
        
    except FileNotFoundError:
        print("File not found! Please check the dataset path.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
