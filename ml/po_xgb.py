import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

class PUMAOptimizer:
    def __init__(self, X, y, population_size=10, generations=100):
        self.X = np.array(X)
        self.y = np.array(y)
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = np.inf  # RMSE: lower is better
        self.best_scores_history = []  # Track best scores for plotting
        
        # Split and scale data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=None  # Remove stratify for regression
        )
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Các tham số cần tối ưu hóa:
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 500},
            'max_depth': {'type': 'int', 'min': 3, 'max': 15},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.3},
            'subsample': {'type': 'float', 'min': 0.5, 'max': 1.0},
            'colsample_bytree': {'type': 'float', 'min': 0.5, 'max': 1.0},
            'min_child_weight': {'type': 'int', 'min': 1, 'max': 7},
            'gamma': {'type': 'float', 'min': 0, 'max': 5}
        }
    
    #Hàm tạo cá thể
    def create_individual(self):
        """Create a random individual (parameter set)"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                individual[param] = random.randint(range_info['min'], range_info['max'])
            else:
                individual[param] = random.uniform(range_info['min'], range_info['max'])
        return individual

    #Hàm đánh giá cá thể, sử dụng để tính hàm fitness
    def evaluate_individual(self, individual):
        """Evaluate fitness of an individual using RMSE (lower is better)"""
        try:
            model = xgb.XGBRegressor(
                n_estimators=individual['n_estimators'],
                max_depth=individual['max_depth'],
                learning_rate=individual['learning_rate'],
                subsample=individual['subsample'],
                colsample_bytree=individual['colsample_bytree'],
                min_child_weight=individual['min_child_weight'],
                gamma=individual['gamma'],
                random_state=42,
                verbosity=0
            )
            
            # Use train-validation split for consistent evaluation
            X_train_val, X_val, y_train_val, y_val = train_test_split(
                self.X_train_scaled, self.y_train, test_size=0.2, random_state=42
            )
            
            model.fit(X_train_val, y_train_val)
            y_pred = model.predict(X_val)
            
            # Calculate RMSE
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # Store metrics in individual
            individual['_metrics'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            return rmse  # Return RMSE (lower is better)
        except:
            individual['_metrics'] = {'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf}
            return np.inf

    def exploration_phase(self, population, fitness_values):
        """PUMA Exploration Phase for regression"""
        new_population = []
        new_fitness = []
        used_combinations = set()  # Giúp tránh trùng lặp Solutions
        pCR = 0.5  # Crossover probability
        p = 0.1    # Increment value for pCR adjustment
        
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
                    if param == j0 or random.random() <= pCR:
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
            if new_fitness_val < fitness_values[i]:  # Lower RMSE is better
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
                # Update pCR when no improvement
                pCR = min(0.9, pCR + p)
        
        return new_population, new_fitness

    def _apply_bounds_and_type(self, param, new_val):
        """Apply bounds and handle data type for parameter"""
        range_info = self.param_ranges[param]
        clipped_val = np.clip(new_val, range_info['min'], range_info['max'])
        
        if range_info['type'] == 'int':
            return int(round(clipped_val))
        else:
            return round(clipped_val, 6)

    def exploitation_phase(self, population, fitness_values):
        """PUMA Exploitation Phase for regression"""
        Q = 0.67  # Exploitation constant
        Beta = 2  # Beta constant
        new_population = []
        new_fitness = []
        
        # Get best solution (RMSE: lower is better)
        best_idx = np.argmin(fitness_values)
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
                new_individual[param] = self._apply_bounds_and_type(param, new_pos[j])
            
            # Evaluate and update (RMSE: lower is better)
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val < fitness_values[i]:  # Lower RMSE is better
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
        
        return new_population, new_fitness

    def optimize(self):
        """Main PUMA optimization algorithm for regression"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness_values = [self.evaluate_individual(ind) for ind in population]
        
        # Initial best (RMSE: lower is better)
        best_idx = np.argmin(fitness_values)
        self.best_individual = population[best_idx].copy()
        self.best_score = fitness_values[best_idx]
        initial_best_score = self.best_score
        self.best_scores_history.append(self.best_score)
        
        # Initialize iteration results tracking
        iteration_results = []
        
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
        
        # Unexperienced Phase (first 3 iterations)
        for iteration in range(3):
            print(f"Iteration {iteration + 1}/3")
            
            # Exploration
            pop_explore, fit_explore = self.exploration_phase(population, fitness_values)
            cost_explore = min(fit_explore)  # Lower RMSE is better
            
            # Exploitation
            pop_exploit, fit_exploit = self.exploitation_phase(population, fitness_values)
            cost_exploit = min(fit_exploit)  # Lower RMSE is better
            
            # Combine and select best solutions
            population = population + pop_explore + pop_exploit
            fitness_values = fitness_values + fit_explore + fit_exploit
            indices = np.argsort(fitness_values)[:self.population_size]  # Sort ascending (lower RMSE is better)
            population = [population[i] for i in indices]
            fitness_values = [fitness_values[i] for i in indices]
            
            # Update best (lower RMSE is better)
            if fitness_values[0] < self.best_score:
                self.best_score = fitness_values[0]
                self.best_individual = population[0].copy()
                print(f"New best RMSE: {self.best_score:.4f}")
                self.best_scores_history.append(self.best_score)
            
            # Save iteration results
            iteration_results.append({
                'iteration': iteration + 1,
                'best_rmse': self.best_score,
                'best_params': self.best_individual.copy(),
                'population_mean_rmse': np.mean(fitness_values),
                'population_min_rmse': np.min(fitness_values),
                'population_max_rmse': np.max(fitness_values),
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
            print(f"Iteration {iteration + 1}/{self.generations}")
            
            if score_explore > score_exploit:
                # Exploration
                population, fitness_values = self.exploration_phase(population, fitness_values)
                count_select = unselected.copy()
                unselected[1] += 1
                unselected[0] = 1
                f3_explore = pf[2]
                f3_exploit += pf[2]
                
                # Update sequence costs (lower RMSE is better)
                if fitness_values[0] < self.best_score:
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
                
                # Update sequence costs (lower RMSE is better)
                if fitness_values[0] < self.best_score:
                    cost_diff = abs(self.best_score - fitness_values[0])
                    seq_cost_exploit = [max(0.01, cost_diff)] + seq_cost_exploit[:2]
                    if cost_diff > 0.01:
                        pf_f3.append(cost_diff)
            
            # Update best solution (lower RMSE is better)
            if fitness_values[0] < self.best_score:
                self.best_score = fitness_values[0]
                self.best_individual = population[0].copy()
                print(f"New best RMSE: {self.best_score:.4f}")
                self.best_scores_history.append(self.best_score)
            
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
            
            # Save iteration results
            iteration_results.append({
                'iteration': iteration + 1,
                'best_rmse': self.best_score,
                'best_params': self.best_individual.copy(),
                'population_mean_rmse': np.mean(fitness_values),
                'population_min_rmse': np.min(fitness_values),
                'population_max_rmse': np.max(fitness_values),
                'phase': 'Exploration' if score_explore > score_exploit else 'Exploitation'
            })
        
        return self.best_individual, self.best_score

def main():
    file_path = "C:/Users/Admin/Downloads/prj/src/flood_data.xlsx"
    
    try:
        df = pd.read_excel(file_path)
        print(f"Read {len(df)} rows of data")
        
        feature_columns = [
            'Rainfall', 'Elevation', 'Slope', 'Aspect', 'Flow_direction',
            'Flow_accumulation', 'TWI', 'Distance_to_river', 'Drainage_capacity',
            'LandCover', 'Imperviousness', 'Surface_temperature'
        ]
        
        label_column = 'label_column'  # Target variable for regression
        
        # Check for missing columns
        missing_cols = [col for col in feature_columns + [label_column] if col not in df.columns]
        if missing_cols:
            return
        
        # Prepare data
        X = df[feature_columns].values
        y = df[label_column].values
        
        # Handle missing values
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Initialize and run PUMA optimizer for XGBoost Regression
        print("Starting PUMA optimization for regression...")
        optimizer = PUMAOptimizer(X, y, population_size=15, generations=10)
        best_params, best_score = optimizer.optimize()
        
        print("\nOptimization completed!")
        print(f"Best RMSE score: {best_score:.4f}")
        print("\nBest parameters:")
        for param, value in best_params.items():
            if param != '_metrics':  # Skip metrics key
                print(f"  {param}: {value}")
            
        # Export convergence data to CSV
        convergence_data = pd.DataFrame({
            'Iteration': range(1, len(optimizer.best_scores_history) + 1),
            'Best_RMSE_Score': optimizer.best_scores_history
        })
        convergence_data.to_csv('po_xgb_convergence.csv', index=False)
        print("\nConvergence data exported to 'po_xgb_convergence.csv'")
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please ensure your Excel file exists at the specified path")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
