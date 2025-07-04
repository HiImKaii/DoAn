import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import time
import warnings
warnings.filterwarnings('ignore')

class PUMASVMOptimizer:
    def __init__(self, X, y, population_size=20, generations=20):
        self.X = np.array(X)
        self.y = np.array(y)
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = -np.inf
        self.history = []
        self.feature_names = None
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale data (very important for SVM)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # SVM parameter ranges
        self.param_ranges = {
            'C': {'type': 'float', 'min': 0.01, 'max': 100.0},
            'gamma': {'type': 'float', 'min': 0.001, 'max': 10.0},
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': {'type': 'int', 'min': 2, 'max': 5},  # Only for poly kernel
            'coef0': {'type': 'float', 'min': 0.0, 'max': 10.0},  # For poly and sigmoid
            'tol': {'type': 'float', 'min': 1e-5, 'max': 1e-2}
        }
        
        # PUMA-specific parameters
        self.PF = [0.5, 0.5, 0.3]  # Parameters for F1, F2, F3
        self.unselected = [1, 1]  # [Exploration, Exploitation]
        self.F3_explore = 0
        self.F3_exploit = 0
        self.seq_time_explore = [1, 1, 1]
        self.seq_time_exploit = [1, 1, 1]
        self.seq_cost_explore = [0, 0, 0]
        self.seq_cost_exploit = [0, 0, 0]
        self.score_explore = 0
        self.score_exploit = 0
        self.PF_F3 = []
        self.mega_explore = 0.99
        self.mega_exploit = 0.99
    
    def create_individual(self):
        """Create a random individual (parameter set) for SVM"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if isinstance(range_info, dict):  # Continuous range
                if range_info['type'] == 'int':
                    individual[param] = random.randint(range_info['min'], range_info['max'])
                elif range_info['type'] == 'float':
                    if param in ['C', 'gamma']:
                        log_min = np.log10(range_info['min'])
                        log_max = np.log10(range_info['max'])
                        individual[param] = 10 ** random.uniform(log_min, log_max)
                    else:
                        individual[param] = random.uniform(range_info['min'], range_info['max'])
            else:  # Discrete choices
                individual[param] = random.choice(range_info)
        return individual
    
    def evaluate_individual(self, individual):
        """Evaluate individual SVM parameters"""
        try:
            svm_params = {
                'C': individual['C'],
                'kernel': individual['kernel'],
                'tol': individual['tol'],
                'random_state': 42,
                'class_weight': 'balanced',
                'probability': True
            }
            
            if individual['kernel'] == 'rbf':
                svm_params['gamma'] = individual['gamma']
            elif individual['kernel'] == 'poly':
                svm_params['gamma'] = individual['gamma']
                svm_params['degree'] = individual['degree']
                svm_params['coef0'] = individual['coef0']
            elif individual['kernel'] == 'sigmoid':
                svm_params['gamma'] = individual['gamma']
                svm_params['coef0'] = individual['coef0']
            
            svm = SVC(**svm_params)
            cv_scores = cross_val_score(svm, self.X_train_scaled, self.y_train, cv=3, scoring='f1')
            return float(np.mean(cv_scores))
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return -np.inf
    
    def exploitation_phase(self, population, best_solution, iteration, max_iter):
        """PUMA Exploitation Phase"""
        Q = 0.67  # Exploitation constant
        Beta = 2  # Beta constant
        new_population = []
        
        for i in range(len(population)):
            beta1 = 2 * random.random()
            beta2 = np.random.randn(len(self.param_ranges))
            
            w = np.random.randn(len(self.param_ranges))  # Eq 37
            v = np.random.randn(len(self.param_ranges))  # Eq 38
            
            F1 = np.random.randn(len(self.param_ranges)) * np.exp(2 - iteration * (2/max_iter))  # Eq 35
            F2 = w * np.power(v, 2) * np.cos((2 * random.random()) * w)  # Eq 36
            
            # Convert current solution to array
            current_pos = np.array([population[i][param] for param in self.param_ranges.keys()])
            best_pos = np.array([best_solution[param] for param in self.param_ranges.keys()])
            
            R_1 = 2 * random.random() - 1  # Eq 34
            
            if random.random() <= 0.5:
                S1 = 2 * random.random() - 1 + np.random.randn(len(self.param_ranges))
                S2 = F1 * R_1 * current_pos + F2 * (1 - R_1) * best_pos
                VEC = S2 / S1
                
                if random.random() > Q:
                    rand_idx = random.randint(0, len(population)-1)
                    rand_pos = np.array([population[rand_idx][param] for param in self.param_ranges.keys()])
                    new_pos = best_pos + beta1 * (np.exp(beta2)) * (rand_pos - current_pos)
                else:
                    new_pos = beta1 * VEC - best_pos
            else:
                r1 = random.randint(0, len(population)-1)
                r1_pos = np.array([population[r1][param] for param in self.param_ranges.keys()])
                mbest = {param: np.mean([p[param] for p in population]) for param in self.param_ranges.keys()}
                mbest_pos = np.array([mbest[param] for param in self.param_ranges.keys()])
                sign = 1 if random.random() > 0.5 else -1
                new_pos = (mbest_pos * r1_pos - sign * current_pos) / (1 + (Beta * random.random()))
            
            # Convert back to dictionary and check boundaries
            new_solution = {}
            for j, param in enumerate(self.param_ranges.keys()):
                if isinstance(self.param_ranges[param], dict):
                    if self.param_ranges[param]['type'] == 'int':
                        new_solution[param] = int(np.clip(new_pos[j], 
                                                        self.param_ranges[param]['min'], 
                                                        self.param_ranges[param]['max']))
                    else:
                        new_solution[param] = float(np.clip(new_pos[j], 
                                                          self.param_ranges[param]['min'], 
                                                          self.param_ranges[param]['max']))
                else:
                    new_solution[param] = population[i][param]
            
            new_population.append(new_solution)
        
        return new_population

    def exploration_phase(self, population):
        """PUMA Exploration Phase"""
        pCR = 0.20  # Initial crossover rate
        PCR = 1 - pCR  # Eq 28
        p = PCR / len(population)  # Eq 29
        new_population = []
        
        for i in range(len(population)):
            # Select 6 different solutions
            available_indices = list(range(len(population)))
            available_indices.remove(i)
            selected_indices = random.sample(available_indices, 6)
            a, b, c, d, e, f = [population[idx] for idx in selected_indices]
            
            # Convert solutions to arrays
            current_pos = np.array([population[i][param] for param in self.param_ranges.keys()])
            a_pos = np.array([a[param] for param in self.param_ranges.keys()])
            b_pos = np.array([b[param] for param in self.param_ranges.keys()])
            c_pos = np.array([c[param] for param in self.param_ranges.keys()])
            d_pos = np.array([d[param] for param in self.param_ranges.keys()])
            e_pos = np.array([e[param] for param in self.param_ranges.keys()])
            f_pos = np.array([f[param] for param in self.param_ranges.keys()])
            
            G = 2 * random.random() - 1  # Eq 26
            
            if random.random() < 0.5:
                # Random solution in search space (Eq 25)
                y_pos = np.array([
                    random.uniform(
                        self.param_ranges[param]['min'] if isinstance(self.param_ranges[param], dict) else 0,
                        self.param_ranges[param]['max'] if isinstance(self.param_ranges[param], dict) else 1
                    )
                    for param in self.param_ranges.keys()
                ])
            else:
                # Complex vector operation (Eq 25)
                y_pos = (a_pos + G * (a_pos - b_pos) + 
                        G * (((a_pos - b_pos) - (c_pos - d_pos)) + 
                             ((c_pos - d_pos) - (e_pos - f_pos))))
            
            # Create new solution with crossover
            new_pos = current_pos.copy()
            j0 = random.randint(0, len(self.param_ranges) - 1)
            
            for j in range(len(self.param_ranges)):
                if j == j0 or random.random() <= pCR:
                    new_pos[j] = y_pos[j]
            
            # Convert back to dictionary and check boundaries
            new_solution = {}
            for j, param in enumerate(self.param_ranges.keys()):
                if isinstance(self.param_ranges[param], dict):
                    if self.param_ranges[param]['type'] == 'int':
                        new_solution[param] = int(np.clip(new_pos[j], 
                                                        self.param_ranges[param]['min'], 
                                                        self.param_ranges[param]['max']))
                    else:
                        new_solution[param] = float(np.clip(new_pos[j], 
                                                          self.param_ranges[param]['min'], 
                                                          self.param_ranges[param]['max']))
                else:
                    new_solution[param] = population[i][param]
            
            new_population.append(new_solution)
            pCR = pCR + p  # Eq 30
        
        return new_population

    def optimize(self):
        """Main PUMA optimization algorithm following MATLAB implementation"""
        try:
            print("Starting PUMA optimization for SVM...")
            
            # Initialize population
            population = [self.create_individual() for _ in range(self.population_size)]
            fitness_values = [self.evaluate_individual(ind) for ind in population]
            best_idx = np.argmax(fitness_values)
            self.best_individual = population[best_idx].copy()
            self.best_score = fitness_values[best_idx]
            initial_best = self.best_individual.copy()
            initial_best_score = self.best_score
            
            # Unexperienced Phase (first 3 iterations)
            for iteration in range(3):
                print(f"\nIteration {iteration + 1}/3 (Unexperienced Phase)")
                
                # Run Exploration
                new_population_explore = self.exploration_phase(population)
                explore_fitness = [self.evaluate_individual(ind) for ind in new_population_explore]
                self.seq_cost_explore[iteration] = max(explore_fitness)
                
                # Run Exploitation
                new_population_exploit = self.exploitation_phase(population, self.best_individual, iteration+1, self.generations)
                exploit_fitness = [self.evaluate_individual(ind) for ind in new_population_exploit]
                self.seq_cost_exploit[iteration] = max(exploit_fitness)
                
                # Combine populations and select best
                all_solutions = population + new_population_explore + new_population_exploit
                all_fitness = fitness_values + explore_fitness + exploit_fitness
                sorted_indices = np.argsort(all_fitness)[::-1]  # Descending order
                population = [all_solutions[i] for i in sorted_indices[:self.population_size]]
                fitness_values = [all_fitness[i] for i in sorted_indices[:self.population_size]]
                
                if fitness_values[0] > self.best_score:
                    self.best_score = fitness_values[0]
                    self.best_individual = population[0].copy()
                
                print(f"Best score: {self.best_score:.4f}")
                print(f"Average score: {np.mean(fitness_values):.4f}")
                
            # Initialize scores based on first 3 iterations
            # Calculate F1, F2 scores for both phases
            self.seq_cost_explore = [abs(initial_best_score - self.seq_cost_explore[0]),  # Eq 5
                                   abs(self.seq_cost_explore[1] - self.seq_cost_explore[0]),  # Eq 6
                                   abs(self.seq_cost_explore[2] - self.seq_cost_explore[1])]  # Eq 7
            
            self.seq_cost_exploit = [abs(initial_best_score - self.seq_cost_exploit[0]),  # Eq 8
                                   abs(self.seq_cost_exploit[1] - self.seq_cost_exploit[0]),  # Eq 9
                                   abs(self.seq_cost_exploit[2] - self.seq_cost_exploit[1])]  # Eq 10
            
            # Update PF_F3
            self.PF_F3.extend([x for x in self.seq_cost_explore if x != 0])
            self.PF_F3.extend([x for x in self.seq_cost_exploit if x != 0])
            
            # Calculate initial scores
            F1_explore = self.PF[0] * (self.seq_cost_explore[0] / self.seq_time_explore[0])  # Eq 1
            F1_exploit = self.PF[0] * (self.seq_cost_exploit[0] / self.seq_time_exploit[0])  # Eq 2
            
            F2_explore = self.PF[1] * (sum(self.seq_cost_explore) / sum(self.seq_time_explore))  # Eq 3
            F2_exploit = self.PF[1] * (sum(self.seq_cost_exploit) / sum(self.seq_time_exploit))  # Eq 4
            
            self.score_explore = (self.PF[0] * F1_explore) + (self.PF[1] * F2_explore)  # Eq 11
            self.score_exploit = (self.PF[0] * F1_exploit) + (self.PF[1] * F2_exploit)  # Eq 12
            
            # Experienced Phase
            flag_change = 1
            for iteration in range(3, self.generations):
                print(f"\nIteration {iteration + 1}/{self.generations} (Experienced Phase)")
                
                if self.score_explore > self.score_exploit:
                    # Run Exploration
                    select_flag = 1
                    population = self.exploration_phase(population)
                    count_select = self.unselected.copy()
                    self.unselected[1] += 1  # Increment exploitation unselected
                    self.unselected[0] = 1   # Reset exploration unselected
                    
                    self.F3_explore = self.PF[2]
                    self.F3_exploit += self.PF[2]
                    
                    fitness_values = [self.evaluate_individual(ind) for ind in population]
                    best_idx = np.argmax(fitness_values)
                    temp_best = population[best_idx]
                    temp_best_score = fitness_values[best_idx]
                    
                    # Update sequence costs
                    self.seq_cost_explore[2] = self.seq_cost_explore[1]
                    self.seq_cost_explore[1] = self.seq_cost_explore[0]
                    self.seq_cost_explore[0] = abs(self.best_score - temp_best_score)
                    
                    if self.seq_cost_explore[0] != 0:
                        self.PF_F3.append(self.seq_cost_explore[0])
                    
                    if temp_best_score > self.best_score:
                        self.best_individual = temp_best.copy()
                        self.best_score = temp_best_score
                
                else:
                    # Run Exploitation
                    select_flag = 2
                    population = self.exploitation_phase(population, self.best_individual, iteration+1, self.generations)
                    count_select = self.unselected.copy()
                    self.unselected[0] += 1  # Increment exploration unselected
                    self.unselected[1] = 1   # Reset exploitation unselected
                    
                    self.F3_explore += self.PF[2]
                    self.F3_exploit = self.PF[2]
                    
                    fitness_values = [self.evaluate_individual(ind) for ind in population]
                    best_idx = np.argmax(fitness_values)
                    temp_best = population[best_idx]
                    temp_best_score = fitness_values[best_idx]
                    
                    # Update sequence costs
                    self.seq_cost_exploit[2] = self.seq_cost_exploit[1]
                    self.seq_cost_exploit[1] = self.seq_cost_exploit[0]
                    self.seq_cost_exploit[0] = abs(self.best_score - temp_best_score)
                    
                    if self.seq_cost_exploit[0] != 0:
                        self.PF_F3.append(self.seq_cost_exploit[0])
                    
                    if temp_best_score > self.best_score:
                        self.best_individual = temp_best.copy()
                        self.best_score = temp_best_score
                
                # Update time sequences if phase changed
                if flag_change != select_flag:
                    flag_change = select_flag
                    
                    self.seq_time_explore[2] = self.seq_time_explore[1]
                    self.seq_time_explore[1] = self.seq_time_explore[0]
                    self.seq_time_explore[0] = count_select[0]
                    
                    self.seq_time_exploit[2] = self.seq_time_exploit[1]
                    self.seq_time_exploit[1] = self.seq_time_exploit[0]
                    self.seq_time_exploit[0] = count_select[1]
                
                # Update scores
                F1_explore = self.PF[0] * (self.seq_cost_explore[0] / self.seq_time_explore[0])  # Eq 14
                F1_exploit = self.PF[0] * (self.seq_cost_exploit[0] / self.seq_time_exploit[0])  # Eq 13
                
                F2_explore = self.PF[1] * (sum(self.seq_cost_explore) / sum(self.seq_time_explore))  # Eq 16
                F2_exploit = self.PF[1] * (sum(self.seq_cost_exploit) / sum(self.seq_time_exploit))  # Eq 15
                
                # Update mega parameters (Eq 17, 18)
                if self.score_explore < self.score_exploit:
                    self.mega_explore = max((self.mega_explore - 0.01), 0.01)
                    self.mega_exploit = 0.99
                elif self.score_explore > self.score_exploit:
                    self.mega_explore = 0.99
                    self.mega_exploit = max((self.mega_exploit - 0.01), 0.01)
                
                lmn_explore = 1 - self.mega_explore  # Eq 24
                lmn_exploit = 1 - self.mega_exploit  # Eq 22
                
                min_PF_F3 = min(self.PF_F3) if self.PF_F3 else 0
                
                self.score_explore = ((self.mega_explore * F1_explore) + 
                                    (self.mega_explore * F2_explore) + 
                                    (lmn_explore * min_PF_F3 * self.F3_explore))  # Eq 20
                
                self.score_exploit = ((self.mega_exploit * F1_exploit) + 
                                    (self.mega_exploit * F2_exploit) + 
                                    (lmn_exploit * min_PF_F3 * self.F3_exploit))  # Eq 19
                
                print(f"Best score: {self.best_score:.4f}")
                print(f"Average score: {np.mean(fitness_values):.4f}")
                print(f"Score Explore: {self.score_explore:.4f}")
                print(f"Score Exploit: {self.score_exploit:.4f}")
                
                self.history.append({
                    'iteration': iteration + 1,
                    'best_score': self.best_score,
                    'avg_score': float(np.mean(fitness_values)),
                    'best_params': self.best_individual.copy()
                })
            
            print("\n" + "=" * 50)
            print("PUMA Optimization completed!")
            if self.best_individual is not None:
                print(f"\nBest solution score: {self.best_score:.4f}")
                print("Best parameters:")
                for param, value in self.best_individual.items():
                    print(f"  {param}: {value}")
            return self.best_individual, self.best_score
            
        except Exception as e:
            print(f"Error in optimization: {str(e)}")
            return None, -np.inf

    def local_search(self, base_solution):
        """Perform local search around a base solution"""
        child = base_solution.copy()
        # Randomly select one parameter to modify
        param = random.choice(list(self.param_ranges.keys()))
        
        if isinstance(self.param_ranges[param], dict):  # Continuous parameter
            current_value = child[param]
            range_info = self.param_ranges[param]
            
            # Define search radius
            if range_info['type'] == 'int':
                radius = max(1, int(0.1 * (range_info['max'] - range_info['min'])))
                min_val = max(range_info['min'], current_value - radius)
                max_val = min(range_info['max'], current_value + radius)
                child[param] = random.randint(min_val, max_val)
            elif range_info['type'] == 'float':
                if param in ['C', 'gamma']:
                    # For log-scale parameters, use multiplicative perturbation
                    factor = random.uniform(0.5, 2.0)
                    new_value = current_value * factor
                    child[param] = np.clip(new_value, range_info['min'], range_info['max'])
                else:
                    radius = 0.1 * (range_info['max'] - range_info['min'])
                    min_val = max(range_info['min'], current_value - radius)
                    max_val = min(range_info['max'], current_value + radius)
                    child[param] = random.uniform(min_val, max_val)
        else:  # Discrete choices
            choices = self.param_ranges[param].copy()
            if child[param] in choices:
                choices.remove(child[param])
            if choices:
                child[param] = random.choice(choices)
        
        return child
    
    def evaluate_final_model(self):
        """Evaluate final SVM model on test set"""
        if self.best_individual is None:
            print("No optimized model available!")
            return None
        
        # Create SVM with best parameters
        svm_params = {
            'C': self.best_individual['C'],
            'kernel': self.best_individual['kernel'],
            'tol': self.best_individual['tol'],
            'random_state': 42,
            'class_weight': 'balanced',
            'probability': True
        }
        
        # Add kernel-specific parameters
        if self.best_individual['kernel'] == 'rbf':
            svm_params['gamma'] = self.best_individual['gamma']
        elif self.best_individual['kernel'] == 'poly':
            svm_params['gamma'] = self.best_individual['gamma']
            svm_params['degree'] = self.best_individual['degree']
            svm_params['coef0'] = self.best_individual['coef0']
        elif self.best_individual['kernel'] == 'sigmoid':
            svm_params['gamma'] = self.best_individual['gamma']
            svm_params['coef0'] = self.best_individual['coef0']
        
        best_svm = SVC(**svm_params)
        best_svm.fit(self.X_train_scaled, self.y_train)
        
        # Predict on test set
        y_pred = best_svm.predict(self.X_test_scaled)
        y_prob = best_svm.predict_proba(self.X_test_scaled)
        
        # Get probabilities for class 1
        if isinstance(y_prob, np.ndarray) and y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
        
        # Calculate metrics
        test_f1 = f1_score(self.y_test, y_pred)
        test_auc = roc_auc_score(self.y_test, y_prob)
        test_acc = accuracy_score(self.y_test, y_pred)
        
        print("\nTest Set Metrics:")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"AUC-ROC: {test_auc:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        
        # Get feature names
        feature_names = getattr(self, 'feature_names', None)
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
        
        # For SVM, we can't get feature importances directly like RF
        # But we can get support vectors information
        support_vector_info = {
            'n_support_vectors': best_svm.n_support_,
            'support_vectors_indices': best_svm.support_,
            'dual_coef': best_svm.dual_coef_
        }
        
        return {
            'model': best_svm,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': self.best_individual,
            'support_vector_info': support_vector_info
        }

def main():
    """Main function for SVM optimization"""
    print("Reading data from Excel file...")
    
    # Change this path to your data file
    file_path = "C:/Users/Admin/Downloads/prj/src/flood_data.xlsx"
    
    try:
        df = pd.read_excel(file_path)
        print(f"Read {len(df)} rows of data")
        
        # Feature columns (adjust according to your Excel file)
        feature_columns = [
            'Rainfall', 'Elevation', 'Slope', 'Aspect', 'Flow_direction',
            'Flow_accumulation', 'TWI', 'Distance_to_river', 'Drainage_capacity',
            'LandCover', 'Imperviousness', 'Surface_temperature'
        ]
        
        # Label column (adjust according to your Excel file)
        label_column = 'label_column'  # 1 = flood, 0 = no flood
        
        # Check for missing columns
        missing_cols = [col for col in feature_columns + [label_column] if col not in df.columns]
        if missing_cols:
            print(f"WARNING: Following columns not found: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Prepare data
        X = df[feature_columns].values
        y = df[label_column].values
        
        # Handle missing values
        if np.isnan(X).any():
            print("WARNING: Missing values found in data!")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        print(f"Features shape: {X.shape}")
        print("Label distribution:")
        y_array = np.asarray(y, dtype=int)
        unique_labels = np.unique(y_array)
        label_counts = np.bincount(y_array)
        for label, count in zip(unique_labels, label_counts):
            print(f"  Class {label}: {count}")
        
        # Initialize and run PUMA optimizer for SVM
        optimizer = PUMASVMOptimizer(X, y, population_size=15, generations=10)
        
        start_time = time.time()
        best_params, best_score = optimizer.optimize()
        end_time = time.time()
        
        print(f"\nOptimization time: {end_time - start_time:.2f} seconds")
        
        if best_params is not None:
            print("\nBest SVM parameters:")
            for param, value in best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")
            print(f"\nBest score: {best_score:.4f}")
            
            # Evaluate final model
            print("\nEvaluating SVM model on test set:")
            final_results = optimizer.evaluate_final_model()
            
            if final_results:
                print(f"Test F1-Score: {final_results['test_f1']:.4f}")
                print(f"Test AUC: {final_results['test_auc']:.4f}")
                print(f"Test Accuracy: {final_results['test_accuracy']:.4f}")
                
                # Print support vector information
                sv_info = final_results['support_vector_info']
                print(f"\nSupport Vector Information:")
                print(f"Number of support vectors per class: {sv_info['n_support_vectors']}")
                print(f"Total support vectors: {len(sv_info['support_vectors_indices'])}")
                
                # # Save model (optional)
                # import joblib
                # joblib.dump(final_results['model'], 'best_flood_svm_model.pkl')
                # print("\nModel saved to 'best_flood_svm_model.pkl'")
        else:
            print("\nOptimization failed to find valid parameters.")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please ensure your Excel file exists at the specified path")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()