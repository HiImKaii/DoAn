import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import time

class PUMAOptimizer:
    def __init__(self, X, y, population_size=20, generations=20):
        self.X = np.array(X)
        self.y = np.array(y)
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = -np.inf
        self.history = []
        self.feature_names = None  # Initialize feature_names
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # RF parameter ranges
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 200},
            'max_depth': {'type': 'int', 'min': 5, 'max': 30},
            'min_samples_split': {'type': 'int', 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 10},
            'max_features': ['sqrt', 'log2', None]
        }
        
        # PUMA-specific parameters
        self.alpha = 0.1  # Exploitation weight
        self.beta = 0.2   # Local search weight
        self.gamma = 0.7  # Global search weight
    
    def create_individual(self):
        """Create a random individual (parameter set) with continuous ranges"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if isinstance(range_info, dict):  # Continuous range
                if range_info['type'] == 'int':
                    # Random integer from range
                    individual[param] = random.randint(range_info['min'], range_info['max'])
                elif range_info['type'] == 'float':
                    # Random float from range
                    individual[param] = random.uniform(range_info['min'], range_info['max'])
            else:  # Discrete choices (like max_features)
                individual[param] = random.choice(range_info)
        return individual
    
    def evaluate_individual(self, individual):
        """Evaluate individual based on multiple objectives"""
        try:
            rf = RandomForestClassifier(
                n_estimators=individual['n_estimators'],
                max_depth=individual['max_depth'],
                min_samples_split=individual['min_samples_split'],
                min_samples_leaf=individual['min_samples_leaf'],
                max_features=individual['max_features'],
                random_state=42,
                class_weight='balanced'
            )
            
            # Use stratified K-fold CV for better evaluation
            try:
                cv_scores = cross_val_score(rf, self.X_train_scaled, self.y_train, 
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
        Kiểm tra xem obj1 có bị dominated bởi obj2 không
        obj format: (combined_score, f1, auc)
        """
        return (obj2[1] >= obj1[1] and obj2[2] >= obj1[2] and 
                (obj2[1] > obj1[1] or obj2[2] > obj1[2]))
    
    def pareto_ranking(self, population_with_scores):
        """Xếp hạng Pareto đơn giản"""
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
        # Chọn cá thể có combined_score cao nhất
        return max(tournament, key=lambda x: x[1][0])
    
    def crossover(self, parent1, parent2):
        """Crossover đơn giản - uniform crossover"""
        child = {}
        for param in self.param_ranges.keys():
            child[param] = random.choice([parent1[param], parent2[param]])
        return child
    
    def mutate(self, individual, mutation_rate=0.3):
        """Mutation - thay đổi ngẫu nhiên một số tham số"""
        mutated = individual.copy()
        for param in self.param_ranges.keys():
            if random.random() < mutation_rate:
                mutated[param] = random.choice(self.param_ranges[param])
        return mutated
    
    def optimize(self):
        """Main PUMA optimization algorithm"""
        try:
            print("Starting PUMA optimization...")
            print(f"Data: {len(self.X)} points, {self.X.shape[1]} features")
            
            # Convert y to numpy array if it's not already
            if not isinstance(self.y, np.ndarray):
                self.y = np.array(self.y)
                
            unique_labels = np.unique(self.y)
            label_counts = np.bincount(self.y.astype(int))
            print("Class distribution:")
            for label, count in zip(unique_labels, label_counts):
                print(f"  Class {label}: {count}")
            print("-" * 50)
            
            # Initialize population
            population = [self.create_individual() for _ in range(self.population_size)]
            population_scores = np.array([self.evaluate_individual(ind) for ind in population])
            
            # Main optimization loop
            for generation in range(self.generations):
                try:
                    print(f"\nGeneration {generation + 1}/{self.generations}")
                    
                    # Sort population by fitness
                    sorted_indices = np.argsort(population_scores)[::-1]
                    population = [population[i] for i in sorted_indices]
                    population_scores = population_scores[sorted_indices]
                    
                    # Update best solution
                    if population_scores[0] > self.best_score:
                        self.best_score = population_scores[0]
                        self.best_individual = population[0].copy()
                        print("\nNew best solution found!")
                        print(f"Parameters:")
                        for param, value in population[0].items():
                            print(f"  {param}: {value}")
                    
                    print(f"\nBest score in generation {generation + 1}: {population_scores[0]:.4f}")
                    print(f"Average score in generation: {np.mean(population_scores):.4f}")
                    print(f"Best parameters in this generation:")
                    for param, value in population[0].items():
                        print(f"  {param}: {value}")
                    
                    # Store history
                    history_entry = {
                        'generation': generation + 1,
                        'best_score': self.best_score,
                        'avg_score': float(np.mean(population_scores)),
                        'best_params': population[0].copy()
                    }
                    self.history.append(history_entry)
                    
                    # Create new population
                    new_population = []
                    new_scores = []
                    
                    # Elitism - keep best solution
                    new_population.append(population[0].copy())
                    new_scores.append(population_scores[0])
                    
                    # Generate new solutions
                    while len(new_population) < self.population_size:
                        # Select parent
                        parent_idx = np.random.randint(len(population))
                        parent = population[parent_idx].copy()
                        
                        # Apply PUMA operators
                        if np.random.random() < self.alpha:
                            # Exploitation - local search around best solution
                            child = self.local_search(population[0])
                        elif np.random.random() < self.beta:
                            # Local search around parent
                            child = self.local_search(parent)
                        else:
                            # Global search - mutation
                            child = self.mutate(parent, mutation_rate=0.3)
                        
                        # Evaluate and add new solution
                        score = self.evaluate_individual(child)
                        new_population.append(child)
                        new_scores.append(score)
                    
                    # Update population
                    population = new_population
                    population_scores = np.array(new_scores)
                    
                except Exception as e:
                    print(f"Error in generation {generation + 1}: {str(e)}")
                    continue
            
            print("\n" + "=" * 50)
            print("Optimization completed!")
            if self.best_individual is not None:
                print("\nBest solution found:")
                print(f"Score: {self.best_score:.4f}")
                print("Parameters:")
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
        best_rf = RandomForestClassifier(
            n_estimators=self.best_individual['n_estimators'],
            max_depth=self.best_individual['max_depth'],
            min_samples_split=self.best_individual['min_samples_split'],
            min_samples_leaf=self.best_individual['min_samples_leaf'],
            max_features=self.best_individual['max_features'],
            random_state=42,
            class_weight='balanced'
        )
        
        best_rf.fit(self.X_train_scaled, self.y_train)
        
        # Predict on test set
        y_pred = best_rf.predict(self.X_test_scaled)
        y_prob = best_rf.predict_proba(self.X_test_scaled)
        
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
        
        return {
            'model': best_rf,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': self.best_individual,
            'feature_importances': dict(zip(feature_names, best_rf.feature_importances_))
        }

def main():
    """Main function"""
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
        # Convert y to numpy array and ensure it's integer type
        y_array = np.asarray(y, dtype=int)
        unique_labels = np.unique(y_array)
        label_counts = np.bincount(y_array)
        for label, count in zip(unique_labels, label_counts):
            print(f"  Class {label}: {count}")
        
        # Initialize and run PUMA optimizer
        optimizer = PUMAOptimizer(X, y, population_size=15, generations=10)
        
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
                print(f"Test F1-Score: {final_results['test_f1']:.4f}")
                print(f"Test AUC: {final_results['test_auc']:.4f}")
                print(f"Test Accuracy: {final_results['test_accuracy']:.4f}")
                
                # Save model (optional)
                import joblib
                joblib.dump(final_results['model'], 'best_flood_rf_model.pkl')
                print("\nModel saved to 'best_flood_rf_model.pkl'")
        else:
            print("\nOptimization failed to find valid parameters.")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please ensure your Excel file exists at the specified path")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()