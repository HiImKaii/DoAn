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
    
    def exploration_phase(self, population):
        """PUMA Exploration Phase"""
        new_population = []
        new_fitness = []
        
        for i in range(self.population_size):
            current = population[i]
            
            # Select 6 different solutions randomly
            available_indices = list(range(self.population_size))
            available_indices.remove(i)
            selected_indices = random.sample(available_indices, 6)
            a, b, c, d, e, f = [population[idx] for idx in selected_indices]
            
            # Create new solution
            new_individual = {}
            for param, range_info in self.param_ranges.items():
                if isinstance(range_info, dict):  # Continuous parameter
                    if random.random() < 0.5:
                        new_individual[param] = random.randint(range_info['min'], range_info['max'])
                    else:
                        G = 2 * random.random() - 1
                        term1 = a[param] + G * (a[param] - b[param])
                        term2 = G * (((a[param] - b[param]) - (c[param] - d[param])) + 
                                   ((c[param] - d[param]) - (e[param] - f[param])))
                        new_val = term1 + term2
                        
                        if range_info['type'] == 'int':
                            new_val = int(round(np.clip(new_val, range_info['min'], range_info['max'])))
                        else:
                            new_val = np.clip(new_val, range_info['min'], range_info['max'])
                        new_individual[param] = new_val
                else:  # Categorical parameter (like kernel)
                    new_individual[param] = random.choice(range_info)
            
            # Evaluate and update
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val > self.evaluate_individual(current):
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(self.evaluate_individual(current))
        
        return new_population, new_fitness

    def exploitation_phase(self, population, best_solution, iteration, max_iter):
        """PUMA Exploitation Phase"""
        Q = 0.67  # Exploitation constant
        Beta = 2  # Beta constant
        new_population = []
        new_fitness = []
        
        # Get best solution
        best_idx = np.argmax([self.evaluate_individual(ind) for ind in population])
        best_solution = population[best_idx]
        
        # Calculate mean position
        mbest = {}
        for param in self.param_ranges:
            if isinstance(self.param_ranges[param], dict):
                if self.param_ranges[param]['type'] == 'int':
                    mbest[param] = int(np.mean([p[param] for p in population]))
                else:
                    mbest[param] = np.mean([p[param] for p in population])
            else:  # categorical parameters
                # For categorical, use mode (most common value)
                values = [p[param] for p in population]
                mbest[param] = max(set(values), key=values.count)
        
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
                current_array = []
                best_array = []
                for param in self.param_ranges:
                    if isinstance(self.param_ranges[param], dict):
                        current_array.append(float(current[param]))
                        best_array.append(float(best_solution[param]))
                    else:  # categorical
                        # For categorical, use index in the possible values as numeric representation
                        current_array.append(float(self.param_ranges[param].index(current[param])))
                        best_array.append(float(self.param_ranges[param].index(best_solution[param])))
                
                current_array = np.array(current_array)
                best_array = np.array(best_array)
                
                S2 = F1 * R_1 * current_array + F2 * (1 - R_1) * best_array
                VEC = S2 / S1
                
                if random.random() > Q:
                    # Eq 32 first part
                    random_sol = random.choice(population)
                    random_array = []
                    for param in self.param_ranges:
                        if isinstance(self.param_ranges[param], dict):
                            random_array.append(float(random_sol[param]))
                        else:  # categorical
                            random_array.append(float(self.param_ranges[param].index(random_sol[param])))
                    random_array = np.array(random_array)
                    new_pos = best_array + beta1 * (np.exp(beta2)) * (random_array - current_array)
                else:
                    # Eq 32 second part
                    new_pos = beta1 * VEC - best_array
            else:
                # Eq 33
                r1 = random.randint(0, self.population_size-1)
                r1_sol = population[r1]
                
                current_array = []
                r1_array = []
                mbest_array = []
                for param in self.param_ranges:
                    if isinstance(self.param_ranges[param], dict):
                        current_array.append(float(current[param]))
                        r1_array.append(float(r1_sol[param]))
                        mbest_array.append(float(mbest[param]))
                    else:  # categorical
                        current_array.append(float(self.param_ranges[param].index(current[param])))
                        r1_array.append(float(self.param_ranges[param].index(r1_sol[param])))
                        mbest_array.append(float(self.param_ranges[param].index(mbest[param])))
                
                current_array = np.array(current_array)
                r1_array = np.array(r1_array)
                mbest_array = np.array(mbest_array)
                
                sign = 1 if random.random() > 0.5 else -1
                new_pos = (mbest_array * r1_array - sign * current_array) / (1 + (Beta * random.random()))
            
            # Convert back to dictionary and clip values
            for j, param in enumerate(self.param_ranges):
                if isinstance(self.param_ranges[param], dict):
                    if self.param_ranges[param]['type'] == 'int':
                        new_individual[param] = int(round(np.clip(new_pos[j], 
                                                                self.param_ranges[param]['min'], 
                                                                self.param_ranges[param]['max'])))
                    else:
                        new_individual[param] = np.clip(new_pos[j], 
                                                      self.param_ranges[param]['min'], 
                                                      self.param_ranges[param]['max'])
                else:  # categorical
                    # Convert back from numeric to categorical
                    idx = int(round(new_pos[j])) % len(self.param_ranges[param])
                    new_individual[param] = self.param_ranges[param][idx]
            
            # Evaluate and update
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val > self.evaluate_individual(current):
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(self.evaluate_individual(current))
        
        return new_population, new_fitness

    def optimize(self):
        """Main PUMA optimization algorithm"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness_values = [self.evaluate_individual(ind) for ind in population]
        
        # Initial best
        best_idx = np.argmax(fitness_values)
        self.best_individual = population[best_idx].copy()
        self.best_score = fitness_values[best_idx]
        initial_best_score = self.best_score
        
        # Parameters for phase selection
        unselected = [1, 1]  # [Exploration, Exploitation]
        seq_time_explore = [1, 1, 1]
        seq_time_exploit = [1, 1, 1]
        seq_cost_explore = [0.0, 0.0, 0.0]
        seq_cost_exploit = [0.0, 0.0, 0.0]
        pf = [0.5, 0.5, 0.3]  # Weights for F1, F2, F3
        mega_explor = 0.99
        mega_exploit = 0.99
        f3_explore = 0
        f3_exploit = 0
        pf_f3 = []
        flag_change = 1
        
        # Unexperienced Phase (first 3 iterations)
        for iteration in range(3):
            # Exploration
            pop_explore, fit_explore = self.exploration_phase(population)
            cost_explore = max(fit_explore)
            
            # Exploitation
            pop_exploit, fit_exploit = self.exploitation_phase(population, self.best_individual, iteration+1, self.generations)
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
        
        # Initialize sequence costs
        seq_cost_explore[0] = abs(initial_best_score - cost_explore)
        seq_cost_exploit[0] = abs(initial_best_score - cost_exploit)
        
        # Add non-zero costs to PF_F3
        for cost in seq_cost_explore + seq_cost_exploit:
            if cost != 0:
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
            if score_explore > score_exploit:
                # Exploration
                population, fitness_values = self.exploration_phase(population)
                count_select = unselected.copy()
                unselected[1] += 1
                unselected[0] = 1
                f3_explore = pf[2]
                f3_exploit += pf[2]
                
                # Update sequence costs
                if fitness_values[0] > self.best_score:
                    seq_cost_explore = [abs(self.best_score - fitness_values[0])] + seq_cost_explore[:2]
                    if seq_cost_explore[0] != 0:
                        pf_f3.append(seq_cost_explore[0])
            else:
                # Exploitation
                population, fitness_values = self.exploitation_phase(population, self.best_individual, iteration+1, self.generations)
                count_select = unselected.copy()
                unselected[0] += 1
                unselected[1] = 1
                f3_explore += pf[2]
                f3_exploit = pf[2]
                
                # Update sequence costs
                if fitness_values[0] > self.best_score:
                    seq_cost_exploit = [abs(self.best_score - fitness_values[0])] + seq_cost_exploit[:2]
                    if seq_cost_exploit[0] != 0:
                        pf_f3.append(seq_cost_exploit[0])
            
            # Update best solution
            if fitness_values[0] > self.best_score:
                self.best_score = fitness_values[0]
                self.best_individual = population[0].copy()
            
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
            
            min_pf_f3 = min(pf_f3) if pf_f3 else 0
            score_explore = (mega_explor * f1_explore) + (mega_explor * f2_explore) + (lmn_explore * min_pf_f3 * f3_explore)
            score_exploit = (mega_exploit * f1_exploit) + (mega_exploit * f2_exploit) + (lmn_exploit * min_pf_f3 * f3_exploit)
        
        return self.best_individual, self.best_score

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