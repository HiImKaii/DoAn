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
        self.alpha = 0.1  # Exploitation weight
        self.beta = 0.2   # Local search weight
        self.gamma = 0.7  # Global search weight
    
    def create_individual(self):
        """Create a random individual (parameter set) for SVM"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if isinstance(range_info, dict):  # Continuous range
                if range_info['type'] == 'int':
                    individual[param] = random.randint(range_info['min'], range_info['max'])
                elif range_info['type'] == 'float':
                    # Use log scale for C and gamma for better exploration
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
            # Create SVM parameters dict, only include relevant ones based on kernel
            svm_params = {
                'C': individual['C'],
                'kernel': individual['kernel'],
                'tol': individual['tol'],
                'random_state': 42,
                'class_weight': 'balanced',
                'probability': True  # Needed for ROC-AUC
            }
            
            # Add kernel-specific parameters
            if individual['kernel'] == 'rbf':
                svm_params['gamma'] = individual['gamma']
            elif individual['kernel'] == 'poly':
                svm_params['gamma'] = individual['gamma']
                svm_params['degree'] = individual['degree']
                svm_params['coef0'] = individual['coef0']
            elif individual['kernel'] == 'sigmoid':
                svm_params['gamma'] = individual['gamma']
                svm_params['coef0'] = individual['coef0']
            # linear kernel doesn't need additional parameters
            
            svm = SVC(**svm_params)
            
            # Use stratified K-fold CV for evaluation
            try:
                cv_scores = cross_val_score(svm, self.X_train_scaled, self.y_train, 
                                          cv=3, scoring='f1')
                
                # Return mean CV score
                return float(np.mean(cv_scores))
            except Exception as e:
                print(f"Error in cross validation: {str(e)}")
                return -np.inf
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            print("Individual parameters:", individual)
            return -np.inf
    
    def is_dominated(self, obj1, obj2):
        """
        Check if obj1 is dominated by obj2
        obj format: (combined_score, f1, auc)
        """
        return (obj2[1] >= obj1[1] and obj2[2] >= obj1[2] and 
                (obj2[1] > obj1[1] or obj2[2] > obj1[2]))
    
    def pareto_ranking(self, population_with_scores):
        """Simple Pareto ranking"""
        n = len(population_with_scores)
        ranks = np.zeros(n)
        
        for i in range(n):
            rank = 0
            for j in range(n):
                if i != j and self.is_dominated(
                    population_with_scores[i][1], 
                    population_with_scores[j][1]
                ):
                    rank += 1
            ranks[i] = rank
        
        return ranks
    
    def tournament_selection(self, population_with_scores, k=3):
        """Tournament selection"""
        tournament = random.sample(population_with_scores, min(k, len(population_with_scores)))
        return max(tournament, key=lambda x: x[1][0])
    
    def crossover(self, parent1, parent2):
        """Simple crossover - uniform crossover"""
        child = {}
        for param in self.param_ranges.keys():
            child[param] = random.choice([parent1[param], parent2[param]])
        return child
    
    def mutate(self, individual, mutation_rate=0.3):
        """Mutation - randomly change some parameters"""
        mutated = individual.copy()
        for param in self.param_ranges.keys():
            if random.random() < mutation_rate:
                if isinstance(self.param_ranges[param], dict):
                    range_info = self.param_ranges[param]
                    if range_info['type'] == 'int':
                        mutated[param] = random.randint(range_info['min'], range_info['max'])
                    elif range_info['type'] == 'float':
                        if param in ['C', 'gamma']:
                            log_min = np.log10(range_info['min'])
                            log_max = np.log10(range_info['max'])
                            mutated[param] = 10 ** random.uniform(log_min, log_max)
                        else:
                            mutated[param] = random.uniform(range_info['min'], range_info['max'])
                else:
                    mutated[param] = random.choice(self.param_ranges[param])
        return mutated
    
    def select_six_solutions_randomly(self, population):
        """Select six solutions randomly for PUMA exploration"""
        return random.sample(population, min(6, len(population)))

    def calculate_new_vector(self, selected_pumas):
        """Calculate new vector using weighted average and randomness (adapted for SVM params)"""
        new_individual = {}
        for param in self.param_ranges.keys():
            if isinstance(self.param_ranges[param], dict):
                values = [puma[param] for puma in selected_pumas]
                new_value = sum(values) / len(values)
                range_info = self.param_ranges[param]
                if range_info['type'] == 'int':
                    noise = random.randint(-1, 1)
                    new_value = int(max(range_info['min'], min(range_info['max'], new_value + noise)))
                else:
                    noise = random.uniform(-0.1, 0.1) * (range_info['max'] - range_info['min'])
                    new_value = max(range_info['min'], min(range_info['max'], new_value + noise))
                new_individual[param] = new_value
            else:
                new_individual[param] = random.choice([puma[param] for puma in selected_pumas])
        return new_individual

    def check_boundary(self, individual):
        """Check and fix boundary violations"""
        for param, range_info in self.param_ranges.items():
            if isinstance(range_info, dict):
                if range_info['type'] == 'int' or range_info['type'] == 'float':
                    individual[param] = max(range_info['min'], min(range_info['max'], individual[param]))
        return individual

    def update_solution(self, current, new_candidate):
        """Update solution based on fitness comparison"""
        current_fitness = self.evaluate_individual(current)
        new_fitness = self.evaluate_individual(new_candidate)
        if new_fitness > current_fitness:
            return new_candidate, new_fitness
        else:
            return current, current_fitness

    def exploitation_phase(self, population, fitness_values):
        """PUMA Exploitation Phase - Algorithm 1"""
        sorted_indices = np.argsort(fitness_values)
        sorted_population = [population[i] for i in sorted_indices]
        new_population = []
        new_fitness = []
        for i, puma in enumerate(sorted_population):
            selected_pumas = self.select_six_solutions_randomly(population)
            new_vector = self.calculate_new_vector(selected_pumas)
            new_vector = self.check_boundary(new_vector)
            updated_solution, updated_fitness = self.update_solution(puma, new_vector)
            new_population.append(updated_solution)
            new_fitness.append(updated_fitness)
        return new_population, new_fitness

    def exploration_phase(self, population, fitness_values):
        """PUMA Exploration Phase - Algorithm 2"""
        new_population = []
        new_fitness = []
        for i, puma in enumerate(population):
            new_candidate = self.create_individual()
            new_cost = self.evaluate_individual(new_candidate)
            if new_cost > fitness_values[i]:
                updated_solution = new_candidate
                updated_fitness = new_cost
            else:
                updated_solution = puma
                updated_fitness = fitness_values[i]
            new_population.append(updated_solution)
            new_fitness.append(updated_fitness)
        return new_population, new_fitness

    def optimize(self):
        """Main PUMA optimization algorithm following the pseudocode (like po_rf)"""
        try:
            print("Starting PUMA optimization for SVM...")
            population = [self.create_individual() for _ in range(self.population_size)]
            fitness_values = [self.evaluate_individual(ind) for ind in population]
            best_idx = np.argmax(fitness_values)
            self.best_individual = population[best_idx].copy()
            self.best_score = fitness_values[best_idx]
            for iteration in range(self.generations):
                print(f"\nIteration {iteration + 1}/{self.generations}")
                if iteration < self.generations // 4:
                    population, fitness_values = self.exploitation_phase(population, fitness_values)
                    print("Applied Exploitation Phase")
                else:
                    if random.random() < 0.5:
                        population, fitness_values = self.exploitation_phase(population, fitness_values)
                        print("Applied Exploitation Phase")
                    else:
                        population, fitness_values = self.exploration_phase(population, fitness_values)
                        print("Applied Exploration Phase")
                current_best_idx = np.argmax(fitness_values)
                if fitness_values[current_best_idx] > self.best_score:
                    self.best_score = fitness_values[current_best_idx]
                    self.best_individual = population[current_best_idx].copy()
                    print(f"New best solution found! Score: {self.best_score:.4f}")
                print(f"Best score: {self.best_score:.4f}")
                print(f"Average score: {np.mean(fitness_values):.4f}")
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