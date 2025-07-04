import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import time
import warnings
warnings.filterwarnings('ignore')

class PUMAOptimizer:
    def __init__(self, X, y, population_size=30, generations=500):
        self.X = np.array(X)
        self.y = np.array(y)
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = -np.inf
        self.history = []
        self.feature_names: list[str] = []
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        
        # Scale data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # XGBoost parameter ranges (bounds)
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 500},
            'max_depth': {'type': 'int', 'min': 3, 'max': 15},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.3},
            'subsample': {'type': 'float', 'min': 0.5, 'max': 1.0},
            'colsample_bytree': {'type': 'float', 'min': 0.5, 'max': 1.0},
            'min_child_weight': {'type': 'int', 'min': 1, 'max': 7},
            'gamma': {'type': 'float', 'min': 0, 'max': 5}
        }
        
        # Problem dimensions
        self.dimension = len(self.param_ranges)
        
        # Initialize PUMA parameters
        self.p = 0.5  # Probability for exploration/exploitation
        self.MaxIteration = generations
    
    def create_individual(self):
        """Create a random individual (parameter set)"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if isinstance(range_info, dict):
                if range_info['type'] == 'int':
                    individual[param] = random.randint(range_info['min'], range_info['max'])
                elif range_info['type'] == 'float':
                    individual[param] = random.uniform(range_info['min'], range_info['max'])
            else:  # Discrete choices
                individual[param] = random.choice(range_info)
        return individual
    
    def evaluate_individual(self, individual):
        """Evaluate fitness of an individual"""
        try:
            xgb_params = {
                'n_estimators': individual['n_estimators'],
                'max_depth': individual['max_depth'],
                'learning_rate': individual['learning_rate'],
                'subsample': individual['subsample'],
                'colsample_bytree': individual['colsample_bytree'],
                'min_child_weight': individual['min_child_weight'],
                'gamma': individual['gamma'],
                'objective': 'binary:logistic',
                'use_label_encoder': False,
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**xgb_params)
            
            # Use cross-validation for evaluation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=3, scoring='f1')
            return float(np.mean(cv_scores))
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return -np.inf
    
    def is_dominated(self, obj1, obj2):
        """Check if obj1 is dominated by obj2"""
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
        """Simple uniform crossover"""
        child = {}
        for param in self.param_ranges.keys():
            child[param] = random.choice([parent1[param], parent2[param]])
        return child
    
    def mutate(self, individual, mutation_rate=0.3):
        """Mutation - randomly change some parameters"""
        mutated = individual.copy()
        for param, range_info in self.param_ranges.items():
            if random.random() < mutation_rate:
                if isinstance(range_info, dict):
                    if range_info['type'] == 'int':
                        mutated[param] = random.randint(range_info['min'], range_info['max'])
                    elif range_info['type'] == 'float':
                        mutated[param] = random.uniform(range_info['min'], range_info['max'])
                else:
                    mutated[param] = random.choice(range_info)
        return mutated
    
    def select_six_solutions_randomly(self, population):
        """Select six solutions randomly for PUMA exploration"""
        return random.sample(population, min(6, len(population)))

    def calculate_new_vector(self, selected_pumas):
        """Calculate new vector using weighted average and randomness (adapted for XGB params)"""
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

    def exploration_phase(self, population, fitness_values):
        """PUMA Exploration Phase"""
        new_population = []
        new_fitness = []

        # Parameters for exploration
        pCR = 0.20
        PCR = 1 - pCR
        p = PCR / self.population_size
        
        for i in range(self.population_size):
            current = population[i]
            
            # Select 6 different solutions (excluding current)
            available_indices = list(range(self.population_size))
            available_indices.remove(i)
            
            if len(available_indices) < 6:
                # If not enough solutions, use what we have
                selected_indices = available_indices
            else:
                selected_indices = random.sample(available_indices, 6)
            
            # Create new solution using exploration strategy
            new_individual = {}
            
            for param, range_info in self.param_ranges.items():
                if isinstance(range_info, dict):  # Numerical parameters
                    if range_info['type'] in ['int', 'float']:
                        # For numerical parameters, use weighted average + mutation
                        values = [population[idx][param] for idx in selected_indices]
                        avg_value = sum(values) / len(values)
                        
                        # Add mutation
                        if range_info['type'] == 'int':
                            mutation = random.randint(-3, 3)
                            new_individual[param] = int(avg_value + mutation)
                        else:
                            range_span = range_info['max'] - range_info['min']
                            mutation = random.uniform(-0.1, 0.1) * range_span
                            new_individual[param] = avg_value + mutation
                else:  # Categorical parameters
                    # For categorical parameters, select randomly from selected solutions
                    values = [population[idx][param] for idx in selected_indices]
                    new_individual[param] = random.choice(values)
            
            # Apply bounds
            new_individual = self.check_boundary(new_individual)
            
            # Evaluate new solution
            new_fitness_val = self.evaluate_individual(new_individual)
            
            # Update solution if better
            if new_fitness_val > fitness_values[i]:
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
                pCR = min(1.0, pCR + p)  # Update pCR when no improvement
        
        return new_population, new_fitness

    def exploitation_phase(self, population, fitness_values):
        """PUMA Exploitation Phase - Fixed Implementation"""
        new_population = []
        new_fitness = []
        
        # Sort population to get best solution
        sorted_indices = np.argsort(fitness_values)[::-1]  # Reverse for descending order
        sorted_pop = [population[i] for i in sorted_indices]
        best_solution = sorted_pop[0]  # Best solution (highest fitness)
        
        # Parameters for exploitation
        Q = 0.67
        Beta = 2.0
        
        for i in range(self.population_size):
            current = population[i]
            new_individual = {}
            
            # Generate new solution using exploitation strategy
            for param, range_info in self.param_ranges.items():
                if isinstance(range_info, dict):  # Numerical parameters
                    current_val = current[param]
                    best_val = best_solution[param]
                    
                    # Exploitation formula with randomization
                    if random.random() <= 0.5:
                        # Move towards best solution
                        beta1 = 2 * random.random()
                        random_sol = random.choice(population)
                        random_val = random_sol[param]
                        
                        if random.random() > Q:
                            new_val = best_val + beta1 * (random_val - current_val)
                        else:
                            new_val = beta1 * best_val - current_val
                    else:
                        # Alternative exploitation strategy
                        mean_val = np.mean([ind[param] for ind in population])
                        random_ind = random.choice(population)
                        random_val = random_ind[param]
                        
                        denominator = 1 + (Beta * random.random())
                        new_val = (mean_val * random_val - current_val) / denominator
                    
                    new_individual[param] = new_val
                else:  # Categorical parameters
                    # For categorical, bias towards best solution
                    if random.random() < 0.7:  # 70% chance to use best
                        new_individual[param] = best_solution[param]
                    else:
                        new_individual[param] = current[param]
            
            # Apply bounds
            new_individual = self.check_boundary(new_individual)
            
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

    def optimize(self):
        """Main PUMA optimization algorithm, strictly following po_rf.py logic"""
        try:
            print("Starting PUMA optimization for XGBoost...")
            # 1. Initialize population
            population = [self.create_individual() for _ in range(self.population_size)]
            fitness_values = [self.evaluate_individual(ind) for ind in population]
            best_idx = np.argmax(fitness_values)
            self.best_individual = population[best_idx].copy()
            self.best_score = fitness_values[best_idx]
            self.history = []

            for iteration in range(self.generations):
                print(f"\nIteration {iteration + 1}/{self.generations}")
                # 2. Chọn phase dựa trên iteration (giống po_rf.py)
                if iteration < self.generations // 4:
                    # Giai đoạn khai thác (exploitation)
                    population, fitness_values = self.exploitation_phase(population, fitness_values)
                    print("Applied Exploitation Phase")
                else:
                    # Ngẫu nhiên chọn giữa khai thác và khám phá
                    if random.random() < 0.5:
                        population, fitness_values = self.exploitation_phase(population, fitness_values)
                        print("Applied Exploitation Phase")
                    else:
                        population, fitness_values = self.exploration_phase(population, fitness_values)
                        print("Applied Exploration Phase")

                # 3. Cập nhật cá thể tốt nhất
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
            
            # Define search radius (10% of parameter range)
            if range_info['type'] == 'int':
                radius = max(1, int(0.1 * (range_info['max'] - range_info['min'])))
                # Generate new value within radius, keeping within bounds
                min_val = max(range_info['min'], current_value - radius)
                max_val = min(range_info['max'], current_value + radius)
                child[param] = random.randint(min_val, max_val)
            elif range_info['type'] == 'float':
                radius = 0.1 * (range_info['max'] - range_info['min'])
                min_val = max(range_info['min'], current_value - radius)
                max_val = min(range_info['max'], current_value + radius)
                child[param] = random.uniform(min_val, max_val)
        else:  # Discrete choices
            current_idx = self.param_ranges[param].index(child[param])
            possible_idx = [i for i in range(len(self.param_ranges[param]))]
            possible_idx.remove(current_idx)
            if possible_idx:
                new_idx = random.choice(possible_idx)
                child[param] = self.param_ranges[param][new_idx]
        
        return child
    
    def evaluate_final_model(self):
        """Evaluate final model on test set"""
        if self.best_individual is None:
            print("No optimized model available!")
            return None
        
        # Train model with best parameters
        xgb_params = {
            'n_estimators': self.best_individual['n_estimators'],
            'max_depth': self.best_individual['max_depth'],
            'learning_rate': self.best_individual['learning_rate'],
            'subsample': self.best_individual['subsample'],
            'colsample_bytree': self.best_individual['colsample_bytree'],
            'min_child_weight': self.best_individual['min_child_weight'],
            'gamma': self.best_individual['gamma'],
            'objective': 'binary:logistic',
            'use_label_encoder': False,
            'random_state': 42
        }
        
        best_xgb = xgb.XGBClassifier(**xgb_params)
        best_xgb.fit(self.X_train_scaled, self.y_train)
        
        # Predict on test set
        y_pred = best_xgb.predict(self.X_test_scaled)
        y_prob = best_xgb.predict_proba(self.X_test_scaled)
        
        # Get probabilities for positive class
        if isinstance(y_prob, np.ndarray) and y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
        
        # Calculate metrics
        test_f1 = f1_score(self.y_test, y_pred)
        test_auc = roc_auc_score(self.y_test, y_prob)
        test_acc = accuracy_score(self.y_test, y_pred)
        
        print("\nFinal Model Test Results:")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"AUC-ROC: {test_auc:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        
        # Feature importance
        feature_names = getattr(self, 'feature_names', None)
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
            
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_xgb.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return {
            'model': best_xgb,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': self.best_individual,
            'feature_importance': feature_importance
        }

def main():
    """Main function"""
    try:
        
        # Get the absolute path of the script and data file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = "C:/Users/Admin/Downloads/prj/src/flood_data.xlsx"
        
        if not os.path.exists(file_path):
            print(f"ERROR: File not found at {file_path}")
            print("Attempting to list parent directory contents...")
            parent_dir = os.path.dirname(file_path)
            if os.path.exists(parent_dir):
                print(f"Contents of {parent_dir}:")
                for item in os.listdir(parent_dir):
                    print(f"  - {item}")
            else:
                print(f"Parent directory {parent_dir} does not exist")
            return
        
        df = pd.read_excel(file_path)
        
        # Feature columns (adjust according to your Excel file)
        feature_columns = [
            'Rainfall', 'Elevation', 'Slope', 'Aspect', 'Flow_direction',
            'Flow_accumulation', 'TWI', 'Distance_to_river', 'Drainage_capacity',
            'LandCover', 'Imperviousness', 'Surface_temperature'
        ]
        
        # Label column
        label_column = 'label_column'
        
        # Check for missing columns
        missing_cols = [col for col in feature_columns + [label_column] if col not in df.columns]
        if missing_cols:
            print(f"WARNING: Following columns not found: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Prepare data
        X = df[feature_columns].values
        y = df[label_column].values
        
        # Print class distribution
        unique_labels = np.unique(y)
        print("\nClass distribution:")
        for label in unique_labels:
            count = np.sum(y == label)
            print(f"  Class {label}: {count} ({count/len(y)*100:.2f}%)")
        
        # Handle missing values
        if np.isnan(X).any():
            print("\nWARNING: Missing values found in data!")
            print("Using median imputation...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            print("Missing values have been imputed.")
        
        # Initialize and run PUMA optimizer
        print("\nInitializing PUMA optimizer...")
        optimizer = PUMAOptimizer(X, y, population_size=20, generations=15)
        optimizer.feature_names = feature_columns
        
        start_time = time.time()
        best_params, best_score = optimizer.optimize()
        end_time = time.time()
        
        print(f"\nOptimization time: {end_time - start_time:.2f} seconds")
        
        if best_params is not None:
            print("\nBest parameters:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            print(f"\nBest score: {best_score:.4f}")
            
            # Evaluate final model
            print("\nEvaluating model on test set:")
            final_results = optimizer.evaluate_final_model()
            
            if final_results:
                print(f"\nTest Set Performance:")
                print(f"F1-Score: {final_results['test_f1']:.4f}")
                print(f"AUC-ROC: {final_results['test_auc']:.4f}")
                print(f"Accuracy: {final_results['test_accuracy']:.4f}")
                
                # model_path = os.path.join(os.path.dirname(file_path), 'best_flood_xgb_model.pkl')
                # # import joblib
                # # joblib.dump(final_results['model'], model_path)
                # # print(f"\nModel saved to: {model_path}")
        else:
            print("\nOptimization failed to find valid parameters.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
