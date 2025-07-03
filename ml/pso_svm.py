import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import time

class Particle:
    """Particle class for PSO algorithm"""
    def __init__(self, param_bounds):
        self.param_bounds = param_bounds
        self.position = self._initialize_position()
        self.velocity = self._initialize_velocity()
        self.best_position = self.position.copy()
        self.best_score = -np.inf
        self.current_score = -np.inf
    
    def _initialize_position(self):
        """Initialize particle position (parameters)"""
        position = {}
        for param, bounds in self.param_bounds.items():
            if bounds['type'] == 'continuous':
                # For log-scale parameters like C and gamma
                if bounds.get('log_scale', False):
                    log_min = np.log10(bounds['min'])
                    log_max = np.log10(bounds['max'])
                    position[param] = 10 ** np.random.uniform(log_min, log_max)
                else:
                    position[param] = np.random.uniform(bounds['min'], bounds['max'])
            elif bounds['type'] == 'discrete':
                position[param] = np.random.choice(bounds['values'])
            elif bounds['type'] == 'integer':
                position[param] = np.random.randint(bounds['min'], bounds['max'] + 1)
        return position
    
    def _initialize_velocity(self):
        """Initialize particle velocity"""
        velocity = {}
        for param, bounds in self.param_bounds.items():
            if bounds['type'] == 'continuous':
                range_size = bounds['max'] - bounds['min']
                velocity[param] = np.random.uniform(-range_size * 0.1, range_size * 0.1)
            elif bounds['type'] == 'discrete':
                velocity[param] = 0  # Discrete parameters don't have velocity
            elif bounds['type'] == 'integer':
                range_size = bounds['max'] - bounds['min']
                velocity[param] = np.random.uniform(-range_size * 0.1, range_size * 0.1)
        return velocity
    
    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """Update particle velocity"""
        for param in self.velocity.keys():
            if self.param_bounds[param]['type'] == 'discrete':
                continue  # Skip discrete parameters
            
            r1, r2 = np.random.random(), np.random.random()
            
            cognitive = c1 * r1 * (self.best_position[param] - self.position[param])
            social = c2 * r2 * (global_best_position[param] - self.position[param])
            
            self.velocity[param] = w * self.velocity[param] + cognitive + social
    
    def update_position(self):
        """Update particle position"""
        for param, bounds in self.param_bounds.items():
            if bounds['type'] == 'continuous':
                self.position[param] += self.velocity[param]
                # Apply bounds
                if bounds.get('log_scale', False):
                    self.position[param] = np.clip(self.position[param], bounds['min'], bounds['max'])
                else:
                    self.position[param] = np.clip(self.position[param], bounds['min'], bounds['max'])
            
            elif bounds['type'] == 'integer':
                self.position[param] += self.velocity[param]
                self.position[param] = int(np.clip(self.position[param], bounds['min'], bounds['max']))
            
            elif bounds['type'] == 'discrete':
                # For discrete parameters, occasionally change randomly
                if np.random.random() < 0.1:  # 10% chance to change
                    self.position[param] = np.random.choice(bounds['values'])

class SVMPSOOptimizer:
    """SVM parameter optimization using Particle Swarm Optimization"""
    
    def __init__(self, X, y, n_particles=20, n_iterations=50):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Define parameter bounds for PSO
        self.param_bounds = {
            'C': {'type': 'continuous', 'min': 0.001, 'max': 1000, 'log_scale': True},
            'gamma': {'type': 'continuous', 'min': 0.0001, 'max': 10, 'log_scale': True},
            'kernel': {'type': 'discrete', 'values': ['linear', 'poly', 'rbf', 'sigmoid']},
            'degree': {'type': 'integer', 'min': 2, 'max': 5},
            'coef0': {'type': 'continuous', 'min': 0.0, 'max': 10.0},
            'tol': {'type': 'continuous', 'min': 1e-5, 'max': 1e-2, 'log_scale': True}
        }
        
        # Initialize swarm
        self.swarm = [Particle(self.param_bounds) for _ in range(self.n_particles)]
        self.global_best_position = None
        self.global_best_score = -np.inf
        self.score_history = []
    
    def _params_to_svm_params(self, position):
        """Convert particle position to SVM parameters"""
        params = position.copy()
        
        # Handle kernel-specific parameters
        kernel = params['kernel']
        if kernel == 'linear':
            params.pop('gamma', None)
            params.pop('degree', None)
            params.pop('coef0', None)
        elif kernel == 'poly':
            pass  # Uses all parameters
        elif kernel == 'rbf':
            params.pop('degree', None)
            params.pop('coef0', None)
        elif kernel == 'sigmoid':
            params.pop('degree', None)
        
        # Add class_weight for imbalanced datasets
        class_counts = np.bincount(self.y_train.astype(int))
        if len(class_counts) == 2 and min(class_counts) / max(class_counts) < 0.3:
            params['class_weight'] = 'balanced'
        
        return params
    
    def evaluate_particle(self, particle):
        """Evaluate a particle's fitness (SVM performance)"""
        try:
            svm_params = self._params_to_svm_params(particle.position)
            
            # Create SVM model
            model = SVC(
                C=svm_params.get('C', 1.0),
                gamma=svm_params.get('gamma', 'scale'),
                kernel=svm_params.get('kernel', 'rbf'),
                degree=svm_params.get('degree', 3),
                coef0=svm_params.get('coef0', 0.0),
                class_weight=svm_params.get('class_weight', None),
                tol=svm_params.get('tol', 1e-3),
                probability=True,
                random_state=42,
                max_iter=10000
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train_scaled, self.y_train, 
                cv=3, scoring='f1', n_jobs=1
            )
            
            score = float(np.mean(cv_scores))
            return score
            
        except Exception as e:
            print(f"Error evaluating particle: {str(e)}")
            return -np.inf
    
    def optimize(self):
        """Main PSO optimization loop"""
        print("Starting SVM Parameter Optimization with PSO...")
        print(f"Data: {len(self.X)} points, {self.X.shape[1]} features")
        print(f"Swarm size: {self.n_particles}")
        print(f"Number of iterations: {self.n_iterations}")
        
        # Class distribution
        unique_labels = np.unique(self.y)
        label_counts = np.bincount(self.y.astype(int))
        print("\nClass distribution:")
        for label, count in zip(unique_labels, label_counts):
            print(f"  Class {label}: {count}")
        print("-" * 60)
        
        # PSO main loop
        for iteration in range(self.n_iterations):
            print(f"\nIteration {iteration + 1}/{self.n_iterations}")
            
            # Evaluate all particles
            for i, particle in enumerate(self.swarm):
                score = self.evaluate_particle(particle)
                particle.current_score = score
                
                # Update particle's best position
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
                    print(f"*** NEW GLOBAL BEST FOUND! Score: {score:.4f} ***")
            
            # Update velocities and positions
            w = 0.9 - 0.4 * iteration / self.n_iterations  # Decreasing inertia weight
            for particle in self.swarm:
                if self.global_best_position is not None:
                    particle.update_velocity(self.global_best_position, w=w)
                    particle.update_position()
            
            # Record best score
            self.score_history.append(self.global_best_score)
            
            # Print iteration summary
            current_scores = [p.current_score for p in self.swarm if p.current_score != -np.inf]
            if current_scores:
                avg_score = np.mean(current_scores)
                print(f"Average score: {avg_score:.4f}")
                print(f"Best score: {max(current_scores):.4f}")
                print(f"Global best: {self.global_best_score:.4f}")
            
            # Print best parameters so far
            if self.global_best_position:
                print("Current best parameters:")
                for param, value in self.global_best_position.items():
                    if isinstance(value, float):
                        print(f"  {param}: {value:.6f}")
                    else:
                        print(f"  {param}: {value}")
        
        print("\n" + "=" * 60)
        print("PSO Optimization completed!")
        
        if self.global_best_position:
            print(f"\nBest F1-Score: {self.global_best_score:.4f}")
            print("Best parameters:")
            for param, value in self.global_best_position.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")
        
        return self.global_best_position, self.global_best_score
    
    def evaluate_final_model(self):
        """Evaluate final optimized model on test set"""
        if self.global_best_position is None:
            print("No optimized model available!")
            return None
        
        # Create final model with best parameters
        svm_params = self._params_to_svm_params(self.global_best_position)
        
        best_model = SVC(
            C=svm_params.get('C', 1.0),
            gamma=svm_params.get('gamma', 'scale'),
            kernel=svm_params.get('kernel', 'rbf'),
            degree=svm_params.get('degree', 3),
            coef0=svm_params.get('coef0', 0.0),
            class_weight=svm_params.get('class_weight', None),
            tol=svm_params.get('tol', 1e-3),
            probability=True,
            random_state=42,
            max_iter=10000
        )
        
        # Train and evaluate
        best_model.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_pred = best_model.predict(self.X_test_scaled)
        y_prob = best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Metrics
        test_f1 = f1_score(self.y_test, y_pred)
        test_auc = roc_auc_score(self.y_test, y_prob)
        test_acc = accuracy_score(self.y_test, y_pred)
        
        print("\n" + "=" * 40)
        print("FINAL TEST SET RESULTS:")
        print("=" * 40)
        print(f"F1-Score: {test_f1:.4f}")
        print(f"AUC-ROC:  {test_auc:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        
        # SVM specific info
        print(f"\nModel Information:")
        print(f"Kernel: {svm_params.get('kernel', 'rbf')}")
        print(f"Total support vectors: {sum(best_model.n_support_)}")
        print(f"Support vectors per class: {dict(zip(best_model.classes_, best_model.n_support_))}")
        
        return {
            'model': best_model,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': self.global_best_position,
            'n_support_vectors': best_model.n_support_,
            'score_history': self.score_history
        }
    
    def plot_convergence(self):
        """Plot PSO convergence curve"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.score_history) + 1), self.score_history, 'b-', linewidth=2)
            plt.title('PSO Convergence - SVM F1-Score Optimization', fontsize=14)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Best F1-Score', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Cannot plot convergence curve.")

def main():
    """Main function"""
    print("PSO-SVM Optimizer")
    print("Reading data from Excel file...")
    
    # Change this path to your data file
    file_path = "C:/Users/Admin/Downloads/prj/src/flood_data.xlsx"
    
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully read {len(df)} rows of data")
        
        # Feature columns
        feature_columns = [
            'Rainfall', 'Elevation', 'Slope', 'Aspect', 'Flow_direction',
            'Flow_accumulation', 'TWI', 'Distance_to_river', 'Drainage_capacity',
            'LandCover', 'Imperviousness', 'Surface_temperature'
        ]
        
        # Label column
        label_column = 'label_column'  # 1 = flood, 0 = no flood
        
        # Check for missing columns
        missing_cols = [col for col in feature_columns + [label_column] if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Following columns not found: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Prepare data
        X = df[feature_columns].values
        y = df[label_column].values
        
        # Handle missing values
        if np.isnan(X).any():
            print("WARNING: Missing values detected! Imputing with median...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            print("Missing values handled successfully.")
        
        # Data size warning
        if len(X) > 10000:
            print(f"INFO: Large dataset ({len(X)} samples). Consider reducing PSO parameters for faster execution.")
        
        # Initialize PSO optimizer
        print("\nInitializing PSO-SVM Optimizer...")
        optimizer = SVMPSOOptimizer(
            X, y, 
            n_particles=15,  # Smaller swarm for efficiency
            n_iterations=30   # Reasonable number of iterations
        )
        
        # Run optimization
        start_time = time.time()
        best_params, best_score = optimizer.optimize()
        end_time = time.time()
        
        print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
        
        if best_params is not None:
            # Evaluate final model
            print("\nEvaluating optimized model...")
            final_results = optimizer.evaluate_final_model()
            
            if final_results:
                print(f"\nSUMMARY:")
                print(f"Cross-validation F1-Score: {best_score:.4f}")
                print(f"Test F1-Score: {final_results['test_f1']:.4f}")
                print(f"Test AUC: {final_results['test_auc']:.4f}")
                print(f"Test Accuracy: {final_results['test_accuracy']:.4f}")
                
                # Plot convergence if possible
                optimizer.plot_convergence()
        else:
            print("\nOptimization failed to find valid parameters.")
            
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        print("Please ensure your Excel file exists at the specified path")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()