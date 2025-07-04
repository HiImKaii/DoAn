import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import time
import math

class PUMAOptimizer:
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
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
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
            rf = RandomForestClassifier(
                n_estimators=individual['n_estimators'],
                max_depth=individual['max_depth'],
                min_samples_split=individual['min_samples_split'],
                min_samples_leaf=individual['min_samples_leaf'],
                max_features=individual['max_features'],
                class_weight='balanced',
                random_state=42
            )
            
            # Use cross-validation for evaluation
            cv_scores = cross_val_score(rf, self.X_train_scaled, self.y_train, 
                                      cv=3, scoring='f1')
            return float(np.mean(cv_scores))
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return -np.inf
    
    def select_six_solutions_randomly(self, population):
        """Select six solutions randomly for PUMA exploration """
        return random.sample(population, min(6, len(population)))
    
    def calculate_new_vector(self, selected_pumas):
        """Calculate new vector using Eq. (24) from PUMA algorithm"""
        new_individual = {}
        
        for param in self.param_ranges.keys():
            if isinstance(self.param_ranges[param], dict):
                # For continuous parameters, take weighted average
                values = [puma[param] for puma in selected_pumas]
                new_value = sum(values) / len(values)
                
                # Add some randomness
                range_info = self.param_ranges[param]
                if range_info['type'] == 'int':
                    noise = random.randint(-2, 2)
                    new_value = int(max(range_info['min'], 
                                      min(range_info['max'], new_value + noise)))
                else:
                    noise = random.uniform(-0.1, 0.1) * (range_info['max'] - range_info['min'])
                    new_value = max(range_info['min'], 
                                  min(range_info['max'], new_value + noise))
                
                new_individual[param] = new_value
            else:
                # For discrete parameters, choose randomly from selected
                new_individual[param] = random.choice([puma[param] for puma in selected_pumas])
        
        return new_individual
    
    def check_boundary(self, individual):
        """Check and fix boundary violations"""
        for param, range_info in self.param_ranges.items():
            if isinstance(range_info, dict):
                if range_info['type'] == 'int':
                    individual[param] = max(range_info['min'], 
                                          min(range_info['max'], individual[param]))
                elif range_info['type'] == 'float':
                    individual[param] = max(range_info['min'], 
                                          min(range_info['max'], individual[param]))
            # Discrete parameters don't need boundary checking
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
        # Sort population in ascending order of fitness
        sorted_indices = np.argsort(fitness_values)
        sorted_population = [population[i] for i in sorted_indices]
        
        new_population = []
        new_fitness = []
        
        for i, puma in enumerate(sorted_population):
            # Calculate p for this iteration
            p_current = self.p
            
            # Select four solutions randomly
            selected_pumas = self.select_four_solutions_randomly(population)
            
            # Calculate new vector by Eq. (24)
            new_vector = self.calculate_new_vector(selected_pumas)
            
            # Check boundary
            new_vector = self.check_boundary(new_vector)
            
            # Update current solution
            updated_solution, updated_fitness = self.update_solution(puma, new_vector)
            
            new_population.append(updated_solution)
            new_fitness.append(updated_fitness)
        
        return new_population, new_fitness
    
    def exploration_phase(self, population, fitness_values):
        """PUMA Exploration Phase - Algorithm 2"""
        new_population = []
        new_fitness = []
        
        for i, puma in enumerate(population):
            # Calculate R, P, and P2 by equations (34), (35), (36)
            R = random.random()
            P = random.random()
            P2 = random.random()
            
            # Calculate cost of NewX
            new_candidate = self.create_individual()  # Generate random solution
            new_cost = self.evaluate_individual(new_candidate)
            
            # Apply exploration logic
            if new_cost > fitness_values[i]:  # If NewX.Cost < X.Cost in minimization
                updated_solution = new_candidate
                updated_fitness = new_cost
            else:
                updated_solution = puma
                updated_fitness = fitness_values[i]
            
            new_population.append(updated_solution)
            new_fitness.append(updated_fitness)
        
        return new_population, new_fitness
    
    def optimize(self):
        """Main PUMA optimization algorithm following the pseudocode"""
        try:
            print("Starting PUMA optimization...")
            
            # PO setting: Initialize population
            population = [self.create_individual() for _ in range(self.population_size)]
            fitness_values = [self.evaluate_individual(ind) for ind in population]
            
            # Find initial best solution
            best_idx = np.argmax(fitness_values)
            self.best_individual = population[best_idx].copy()
            self.best_score = fitness_values[best_idx]
            
            # Main optimization loop
            for iteration in range(self.MaxIteration):
                print(f"\nIteration {iteration + 1}/{self.MaxIteration}")
                
                # Apply Unexperienced Phase
                if iteration < self.MaxIteration // 4:  # First quarter iterations
                    # Apply experienced Phase
                    population, fitness_values = self.exploitation_phase(population, fitness_values)
                    print("Applied Exploitation Phase")
                else:
                    # Apply exploitation phase (Algorithm 1)
                    if random.random() < 0.5:  # 50% chance
                        population, fitness_values = self.exploitation_phase(population, fitness_values)
                        print("Applied Exploitation Phase")
                    else:
                        # Apply exploration phase (Algorithm 2)
                        population, fitness_values = self.exploration_phase(population, fitness_values)
                        print("Applied Exploration Phase")
                
                # Update best solution
                current_best_idx = np.argmax(fitness_values)
                if fitness_values[current_best_idx] > self.best_score:
                    self.best_score = fitness_values[current_best_idx]
                    self.best_individual = population[current_best_idx].copy()
                    print(f"New best solution found! Score: {self.best_score:.4f}")
                
                # Update Score_Leader and Score_Follower by Eq. (17)
                # This is implicitly done by tracking best_score
                
                print(f"Best score: {self.best_score:.4f}")
                print(f"Average score: {np.mean(fitness_values):.4f}")
                
                # Store history
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
            class_weight='balanced',
            random_state=42
        )
        
        best_rf.fit(self.X_train_scaled, self.y_train)
        
        # Predict on test set
        y_pred = best_rf.predict(self.X_test_scaled)
        y_prob = best_rf.predict_proba(self.X_test_scaled)
        
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
        
        return {
            'model': best_rf,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': self.best_individual,
            'feature_importances': dict(zip(feature_names, best_rf.feature_importances_))
        }

def main():
    """Main function to run PUMA optimization"""
    print("Reading data from Excel file...")
    
    # Update this path to your data file
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
            print(f"WARNING: Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Prepare data
        X = df[feature_columns].values
        y = df[label_column].values
        
        # Handle missing values
        if np.isnan(X).any():
            print("WARNING: Found missing values, applying imputation...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        print(f"Features shape: {X.shape}")
        print("Label distribution:")
        y_array = np.asarray(y, dtype=int)
        unique_labels, counts = np.unique(y_array, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count}")
        
        # Initialize and run PUMA optimizer
        optimizer = PUMAOptimizer(X, y, population_size=20, generations=15)
        optimizer.feature_names = feature_columns
        
        start_time = time.time()
        best_params, best_score = optimizer.optimize()
        end_time = time.time()
        
        print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
        
        if best_params is not None:
            print("\nFinal Results:")
            print(f"Best CV Score: {best_score:.4f}")
            print("\nBest Parameters:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            
            # Evaluate on test set
            final_results = optimizer.evaluate_final_model()
            
            if final_results:
                print("\nFeature Importance (Top 5):")
                sorted_features = sorted(final_results['feature_importances'].items(), 
                                       key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features[:5]:
                    print(f"  {feature}: {importance:.4f}")
        else:
            print("\nOptimization failed!")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please check the file path and ensure the Excel file exists.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()