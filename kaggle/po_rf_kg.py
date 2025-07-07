# Try to import GPU libraries, fallback to CPU if not available
try:
    import cudf
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as cuRF
    from sklearn.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = False

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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

class ImprovedPUMAOptimizer:
    def __init__(self, X, y, population_size=10, generations=100):
        self.X = X
        self.y = y
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = -np.inf
        self.best_scores_history = []
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
            X_np, y_np, test_size=0.2, stratify=y_np, random_state=42
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
        
        # RF parameter ranges
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 800},
            'max_depth': {'type': 'int', 'min': 5, 'max': 50},
            'min_samples_split': {'type': 'int', 'min': 2, 'max': 50},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 50}
        }
        
        # Get numerical parameters for consistent vector operations
        self.numerical_params = list(self.param_ranges.keys())
        self.num_numerical = len(self.numerical_params)

    def create_individual(self):
        """Create a random individual"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            individual[param] = random.randint(range_info['min'], range_info['max'])
        return individual

    def create_initial_population(self):
        """Create initial population with random individuals"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def evaluate_individual(self, individual):
        """Đánh giá fitness của một nghiệm"""
        try:
            # Create model với tham số phù hợp
            if self.has_gpu:
                model = cuRF(
                    n_estimators=individual['n_estimators'],
                    max_depth=individual['max_depth'],
                    min_samples_split=individual['min_samples_split'],
                    min_samples_leaf=individual['min_samples_leaf'],
                    random_state=42  # Add random seed for reproducibility
                )
            else:
                model = cuRF(
                    n_estimators=individual['n_estimators'],
                    max_depth=individual['max_depth'],
                    min_samples_split=individual['min_samples_split'],
                    min_samples_leaf=individual['min_samples_leaf'],
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42  # Add random seed for reproducibility
                )

            # Đánh giá model
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
                
                X_train_val = cudf.DataFrame(X_train_val)
                X_val = cudf.DataFrame(X_val)
                y_train_val = cudf.Series(y_train_val)
                y_val = cudf.Series(y_val)
                
                model.fit(X_train_val, y_train_val)
                y_pred = model.predict(X_val)
                probabilities = model.predict_proba(X_val)
                
                if isinstance(y_pred, (cudf.Series, cudf.DataFrame)):
                    y_pred = y_pred.to_numpy()
                if isinstance(y_val, (cudf.Series, cudf.DataFrame)):
                    y_val = y_val.to_numpy()
                if isinstance(probabilities, (cudf.Series, cudf.DataFrame)):
                    probabilities = probabilities.to_numpy()
                
                y_pred_proba = probabilities[:, 1]
                
                # Calculate all metrics
                acc = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred, average='binary')
                rec = recall_score(y_val, y_pred, average='binary')
                f1 = f1_score(y_val, y_pred, average='binary')
                roc = roc_auc_score(y_val, y_pred_proba)
                r2 = r2_score(y_val, y_pred)
                
                # Calculate fitness as sum of all metrics
                fitness = acc + prec + rec + f1 + roc + r2
                
            else:
                # Define KFold with fixed random state
                kfold = KFold(n_splits=3, shuffle=True, random_state=42)
                
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                          cv=kfold, scoring='f1_weighted')
                f1 = float(np.mean(cv_scores))
                
                # For CPU version, we'll estimate other metrics through cross-validation
                acc_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                           cv=kfold, scoring='accuracy')
                prec_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                            cv=kfold, scoring='precision_weighted')
                rec_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                           cv=kfold, scoring='recall_weighted')
                roc_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                           cv=kfold, scoring='roc_auc')
                r2_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                          cv=kfold, scoring='r2')
                
                acc = float(np.mean(acc_scores))
                prec = float(np.mean(prec_scores))
                rec = float(np.mean(rec_scores))
                roc = float(np.mean(roc_scores))
                r2 = float(np.mean(r2_scores))
                
                # Calculate fitness as sum of all metrics
                fitness = acc + prec + rec + f1 + roc + r2
            
            return fitness, acc, prec, rec, f1, roc, r2
            
        except Exception as e:
            print(f"Error in evaluate_individual: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    def adaptive_exploration_phase(self, population, fitness_values, generation):
        """PUMA Exploration Phase"""
        new_population = []
        new_fitness = []
        new_metrics = []
        
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
                if random.random() < 0.5:
                    new_individual[param] = random.randint(range_info['min'], range_info['max'])
                else:
                    G = 2 * random.random() - 1
                    term1 = a[param] + G * (a[param] - b[param])
                    term2 = G * (((a[param] - b[param]) - (c[param] - d[param])) + 
                               ((c[param] - d[param]) - (e[param] - f[param])))
                    new_val = int(round(np.clip(term1 + term2, range_info['min'], range_info['max'])))
                    new_individual[param] = new_val
            
            # Evaluate and update
            new_fitness_vals = self.evaluate_individual(new_individual)
            current_fitness_vals = fitness_values[i]
            
            if new_fitness_vals[0] > current_fitness_vals[0]:  # Compare only fitness value
                new_population.append(new_individual)
                new_fitness.append(new_fitness_vals)
            else:
                new_population.append(current)
                new_fitness.append(current_fitness_vals)
        
        return new_population, new_fitness
    
    def exploitation_phase(self, population, fitness_values):
        """PUMA Exploitation Phase"""
        Q = 0.67
        Beta = 2
        new_population = []
        new_fitness = []
        
        # Find best solution based on fitness value only
        best_idx = np.argmax([f[0] for f in fitness_values])  # Get index of best fitness
        best_solution = population[best_idx]
        
        # Tính mean position
        mbest = {}
        for param in self.numerical_params:
            mbest[param] = int(np.mean([p[param] for p in population]))
        
        for i in range(self.population_size):
            current = population[i]
            new_individual = {}

            beta1 = 2 * random.random()
            beta2 = np.random.randn(self.num_numerical)
            
            w = np.random.randn(self.num_numerical)
            v = np.random.randn(self.num_numerical)
            
            F1 = np.random.randn(self.num_numerical) * np.exp(2 - i * (2/self.generations))
            F2 = w * np.power(v, 2) * np.cos((2 * random.random()) * w)
            
            R_1 = 2 * random.random() - 1
            
            if random.random() <= 0.5:
                S1 = 2 * random.random() - 1 + np.random.randn(self.num_numerical)
                S1 = np.where(np.abs(S1) < 1e-10, 1e-10, S1)
                
                current_array = np.array([current[param] for param in self.numerical_params])
                best_array = np.array([best_solution[param] for param in self.numerical_params])
                
                S2 = F1 * R_1 * current_array + F2 * (1 - R_1) * best_array
                VEC = S2 / S1
                
                if random.random() > Q:
                    random_sol = random.choice(population)
                    random_array = np.array([random_sol[param] for param in self.numerical_params])
                    new_pos = best_array + beta1 * (np.exp(np.clip(beta2, -10, 10))) * (random_array - current_array)
                else:
                    new_pos = beta1 * VEC - best_array
            else:
                r1 = random.randint(0, self.population_size-1)
                r1_sol = population[r1]
                r1_array = np.array([r1_sol[param] for param in self.numerical_params])
                mbest_array = np.array([mbest[param] for param in self.numerical_params])
                current_array = np.array([current[param] for param in self.numerical_params])
                
                sign = 1 if random.random() > 0.5 else -1
                denominator = 1 + (Beta * random.random())
                new_pos = (mbest_array * r1_array - sign * current_array) / denominator
            
            # Chuyển đổi về dictionary và clip values
            for j, param in enumerate(self.numerical_params):
                range_info = self.param_ranges[param]
                new_individual[param] = int(round(np.clip(new_pos[j], 
                                            range_info['min'], 
                                            range_info['max'])))
            
            # Đánh giá và cập nhật
            new_fitness_vals = self.evaluate_individual(new_individual)
            current_fitness_vals = fitness_values[i]
            
            if new_fitness_vals[0] > current_fitness_vals[0]:  # Compare only fitness value
                new_population.append(new_individual)
                new_fitness.append(new_fitness_vals)
            else:
                new_population.append(current)
                new_fitness.append(current_fitness_vals)
        
        return new_population, new_fitness
    
    def optimize(self):
        """Run the PUMA optimization process"""
        # Initialize population
        print("\nInitializing population...")
        population = self.create_initial_population()
        fitness_values = []
        metrics_history = []
        
        # Evaluate initial population
        for ind in population:
            fitness_vals = self.evaluate_individual(ind)
            fitness_values.append(fitness_vals)
            metrics_history.append({
                'generation': 0,  # Add generation number for initial population
                'params': ind.copy(),  # Add parameters
                'fitness': fitness_vals[0],
                'accuracy': fitness_vals[1],
                'precision': fitness_vals[2],
                'recall': fitness_vals[3],
                'f1': fitness_vals[4],
                'roc_auc': fitness_vals[5],
                'r2': fitness_vals[6]
            })
        
        # Track best solution
        best_idx = np.argmax([f[0] for f in fitness_values])  # Get index of best fitness
        self.best_individual = population[best_idx].copy()
        self.best_score = fitness_values[best_idx][0]  # Store best fitness value
        self.best_scores_history.append(self.best_score)
        
        print("\nOptimization Progress:")
        print("Generation | Best F1 | Accuracy | Precision | Recall | ROC AUC | R2 | Fitness")
        print("-" * 80)
        
        for generation in range(self.generations):
            # Exploration phase
            population, fitness_values = self.adaptive_exploration_phase(population, fitness_values, generation)
            
            # Exploitation phase
            population, fitness_values = self.exploitation_phase(population, fitness_values)
            
            # Update best solution
            current_best_idx = np.argmax([f[0] for f in fitness_values])  # Get index of best fitness
            if fitness_values[current_best_idx][0] > self.best_score:
                self.best_individual = population[current_best_idx].copy()
                self.best_score = fitness_values[current_best_idx][0]  # Update best fitness value
            
            self.best_scores_history.append(self.best_score)
            
            # Evaluate current best model
            fitness_vals = self.evaluate_individual(self.best_individual)
            
            # Print progress with all metrics including fitness
            print(f"{generation+1:^10d} | {fitness_vals[4]:.4f} | {fitness_vals[1]:.4f} | {fitness_vals[2]:.4f} | {fitness_vals[3]:.4f} | {fitness_vals[5]:.4f} | {fitness_vals[6]:.4f} | {fitness_vals[0]:.4f}")
            
            metrics_history.append({
                'generation': generation,
                'params': self.best_individual.copy(),
                'fitness': fitness_vals[0],
                'accuracy': fitness_vals[1],
                'precision': fitness_vals[2],
                'recall': fitness_vals[3],
                'f1': fitness_vals[4],
                'roc_auc': fitness_vals[5],
                'r2': fitness_vals[6]
            })
        
        # Print final results
        print("\n" + "="*50)
        print("FINAL OPTIMIZATION RESULTS")
        print("="*50)
        print(f"Best F1 Score: {self.best_score:.6f}")
        print(f"Improvement from initial: {self.best_score - self.best_scores_history[0]:.6f}")
        print(f"Improvement percentage: {((self.best_score - self.best_scores_history[0]) / self.best_scores_history[0] * 100):.2f}%")
        
        final_metrics = metrics_history[-1]
        print("\nFinal Metrics:")
        print(f"Accuracy:  {final_metrics['accuracy']:.6f}")
        print(f"Precision: {final_metrics['precision']:.6f}")
        print(f"Recall:    {final_metrics['recall']:.6f}")
        print(f"ROC AUC:   {final_metrics['roc_auc']:.6f}")
        print(f"R2 Score:  {final_metrics['r2']:.6f}")
        
        print("\nBest parameters:")
        for param, value in self.best_individual.items():
            print(f"  {param:20}: {value}")
        
        # Save results to CSV
        results_data = []
        for hist in metrics_history:
            row = {
                'generation': hist['generation'],
                'f1_score': hist['f1'],
                'accuracy': hist['accuracy'],
                'precision': hist['precision'],
                'recall': hist['recall'],
                'roc_auc': hist['roc_auc'],
                'r2': hist['r2'],
                'fitness': hist['fitness']
            }
            row.update({f'param_{k}': v for k, v in hist['params'].items()})
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv('puma_optimization_results.csv', index=False)
        print("\nDetailed optimization results saved to 'puma_optimization_results.csv'")
        
        return self.best_individual, self.best_score

def plot_optimization_analysis(optimizer):
    """Plot optimization analysis"""
    plt.figure(figsize=(10, 6))
    
    # Plot F1 score progression
    plt.plot(optimizer.best_scores_history, 'b-', linewidth=2, label='Best F1 Score')
    plt.title('F1 Score Progression')
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('puma_optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    try:
        # Check GPU availability
        has_gpu = check_gpu()
        
        # Read CSV
        df = pd.read_csv('/kaggle/input/flood-trainning/flood_training.csv', sep=';', na_values='<Null>')
        
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
        optimizer = ImprovedPUMAOptimizer(
            X, y, 
            population_size=25,
            generations=100
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
        max_features = best_params['max_features']
        if max_features == 'auto':
            max_features = 'sqrt'
        
        # Create final model
        if has_gpu:
            final_model = cuRF(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                random_state=42  # Add random seed for reproducibility
            )
        else:
            final_model = cuRF(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                class_weight='balanced',
                n_jobs=-1,
                random_state=42  # Add random seed for reproducibility
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
        metrics_df.to_csv('puma_final_metrics.csv', index=False)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Flood', 'Flood'],
                   yticklabels=['No Flood', 'Flood'])
        plt.title('Confusion Matrix - PUMA Optimized RF')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('puma_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot feature importance
        if hasattr(final_model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            importances = final_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.title('Feature Importances - PUMA Optimized RF')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), 
                      [feature_columns[i] for i in indices], 
                      rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('puma_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("\n" + "="*60)
        print("ALL RESULTS HAVE BEEN SAVED")
        print("="*60)
        print("• puma_results.csv - Detailed iteration results")
        print("• puma_final_metrics.csv - Final metrics")
        print("• puma_optimization_analysis.png - Analysis plots")
        print("• puma_confusion_matrix.png - Confusion matrix")
        print("• puma_feature_importance.png - Feature importance")
        
    except FileNotFoundError:
        print("File not found! Please check the dataset path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()