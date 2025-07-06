import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import time
import joblib


class PSOSVMOptimizer:
    """Particle Swarm Optimization for SVM hyperparameter tuning."""
    
    def __init__(self, X, y, n_particles=30, n_iterations=50, random_state=42):
        """Initialize PSO optimizer."""
        self.X = np.array(X)
        self.y = np.array(y, dtype=int)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        
        # Prepare data
        self._prepare_data()
        
        # PSO parameters
        self.w = 0.9    # Inertia weight
        self.c1 = 2.0   # Cognitive parameter
        self.c2 = 2.0   # Social parameter
        self.w_min = 0.4 # Minimum inertia weight
        
        # Parameter search space
        self.param_ranges = {
            'C': {'type': 'continuous', 'min': 0.001, 'max': 1000, 'log_scale': True},
            'gamma': {'type': 'continuous', 'min': 0.0001, 'max': 10, 'log_scale': True},
            'kernel': {'type': 'discrete', 'values': ['linear', 'rbf', 'poly', 'sigmoid']},
            'degree': {'type': 'integer', 'min': 2, 'max': 5},
            'coef0': {'type': 'continuous', 'min': 0.0, 'max': 10.0}
        }
        
        # Initialize swarm
        self._initialize_swarm()
        
        # Optimization results
        self.global_best_position = {}
        self.global_best_score = -np.inf
        self.optimization_history = []
        self.avg_scores_history = []
    
    def _prepare_data(self):
        """Prepare and split data for training."""
        # Handle missing values
        if np.isnan(self.X).any():
            imputer = SimpleImputer(strategy='median')
            self.X = imputer.fit_transform(self.X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, 
            stratify=self.y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def _initialize_swarm(self):
        """Initialize particle positions and velocities."""
        # Initialize positions
        self.positions = []
        for _ in range(self.n_particles):
            position = {}
            for param, range_info in self.param_ranges.items():
                if range_info['type'] == 'continuous':
                    if range_info.get('log_scale', False):
                        position[param] = np.random.uniform(
                            np.log10(range_info['min']), 
                            np.log10(range_info['max'])
                        )
                    else:
                        position[param] = np.random.uniform(
                            range_info['min'], 
                            range_info['max']
                        )
                elif range_info['type'] == 'integer':
                    position[param] = np.random.randint(
                        range_info['min'], 
                        range_info['max'] + 1
                    )
                else:  # discrete
                    position[param] = np.random.choice(range_info['values'])
            self.positions.append(position)
        
        # Initialize velocities
        self.velocities = []
        for _ in range(self.n_particles):
            velocity = {}
            for param, range_info in self.param_ranges.items():
                if range_info['type'] == 'continuous':
                    if range_info.get('log_scale', False):
                        max_velocity = (np.log10(range_info['max']) - np.log10(range_info['min'])) * 0.1
                    else:
                        max_velocity = (range_info['max'] - range_info['min']) * 0.1
                    velocity[param] = np.random.uniform(-max_velocity, max_velocity)
                elif range_info['type'] == 'integer':
                    max_velocity = (range_info['max'] - range_info['min']) * 0.1
                    velocity[param] = np.random.uniform(-max_velocity, max_velocity)
                else:  # discrete
                    velocity[param] = 0
            self.velocities.append(velocity)
        
        # Initialize personal best
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.n_particles, -np.inf)
    
    def _evaluate_fitness(self, position):
        """Evaluate particle fitness using cross-validation."""
        try:
            # Convert log-scale parameters
            params = position.copy()
            for param, range_info in self.param_ranges.items():
                if range_info['type'] == 'continuous' and range_info.get('log_scale', False):
                    params[param] = 10 ** params[param]
            
            svm = SVC(
                C=params['C'],
                kernel=params['kernel'],
                gamma=params['gamma'] if params['kernel'] != 'linear' else 'auto',
                degree=params['degree'] if params['kernel'] == 'poly' else 3,
                coef0=params['coef0'] if params['kernel'] in ['poly', 'sigmoid'] else 0.0,
                random_state=self.random_state,
                probability=True
            )
            
            cv_scores = cross_val_score(
                svm, self.X_train_scaled, self.y_train, 
                cv=3, scoring='f1', n_jobs=-1
            )
            
            return float(np.mean(cv_scores))
        except:
            return -np.inf
    
    def _update_particle(self, particle_idx):
        """Update particle velocity and position."""
        # Update velocity
        w = self.w - (self.w - self.w_min) * (particle_idx / self.n_particles)
        
        for param, range_info in self.param_ranges.items():
            if range_info['type'] != 'discrete':
                # Standard PSO velocity update formula
                r1, r2 = np.random.random(2)
                cognitive = self.c1 * r1 * (self.personal_best_positions[particle_idx][param] - 
                                          self.positions[particle_idx][param])
                social = self.c2 * r2 * (self.global_best_position[param] - 
                                       self.positions[particle_idx][param])
                
                self.velocities[particle_idx][param] = (w * self.velocities[particle_idx][param] + 
                                                      cognitive + social)
                
                # Update position
                self.positions[particle_idx][param] += self.velocities[particle_idx][param]
                
                # Clamp position to bounds
                if range_info['type'] == 'continuous':
                    if range_info.get('log_scale', False):
                        self.positions[particle_idx][param] = np.clip(
                            self.positions[particle_idx][param],
                            np.log10(range_info['min']),
                            np.log10(range_info['max'])
                        )
                    else:
                        self.positions[particle_idx][param] = np.clip(
                            self.positions[particle_idx][param],
                            range_info['min'],
                            range_info['max']
                        )
                else:  # integer
                    self.positions[particle_idx][param] = int(np.clip(
                        self.positions[particle_idx][param],
                        range_info['min'],
                        range_info['max']
                    ))
            else:  # discrete parameters
                if np.random.random() < 0.1:  # 10% chance to change
                    self.positions[particle_idx][param] = np.random.choice(range_info['values'])
    
    def optimize(self):
        """Execute PSO optimization algorithm."""
        print("Starting PSO optimization...")
        print(f"Dataset: {len(self.X)} samples, {self.X.shape[1]} features")
        print(f"Class distribution: {np.bincount(self.y)}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Evaluate initial swarm
        for i in range(self.n_particles):
            score = self._evaluate_fitness(self.positions[i])
            self.personal_best_scores[i] = score
            
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i].copy()
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            # Update particles
            current_scores = []
            for i in range(self.n_particles):
                self._update_particle(i)
                
                # Evaluate new position
                score = self._evaluate_fitness(self.positions[i])
                current_scores.append(score)
                
                # Update personal best
                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()
                    
                    # Update global best
                    if score > self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.positions[i].copy()
            
            # Calculate average score
            avg_score = np.mean(current_scores)
            self.avg_scores_history.append(avg_score)
            
            # Log progress
            print(f"Iteration {iteration + 1:2d}/{self.n_iterations}: "
                  f"Best F1={self.global_best_score:.4f}, Avg F1={avg_score:.4f}")
            
            # Store history
            self.optimization_history.append({
                'iteration': iteration + 1,
                'best_score': self.global_best_score,
                'best_params': self.global_best_position.copy(),
                'population_mean_score': np.mean(current_scores),
                'population_min_score': np.min(current_scores),
                'population_max_score': np.max(current_scores),
                'inertia_weight': self.w,
                'cognitive_param': self.c1,
                'social_param': self.c2
            })
        
        optimization_time = time.time() - start_time
        
        print("-" * 60)
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Best F1 Score: {self.global_best_score:.4f}")
        print("Optimal Parameters:")
        for param, value in self.global_best_position.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")
        
        # Export convergence data to CSV
        convergence_data = pd.DataFrame(self.optimization_history)
        convergence_data.to_csv('pso_svm_iterations.csv', index=False)
        print("\nConvergence data exported to 'pso_svm_iterations.csv'")
        
        return self.global_best_position, self.global_best_score
    
    def evaluate_test_performance(self):
        """Train final model and evaluate on test set."""
        if not self.global_best_position:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        # Convert parameters
        params = self.global_best_position.copy()
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'continuous' and range_info.get('log_scale', False):
                params[param] = 10 ** params[param]
        
        # Train final model
        final_model = SVC(
            C=params['C'],
            kernel=params['kernel'],
            gamma=params['gamma'] if params['kernel'] != 'linear' else 'auto',
            degree=params['degree'] if params['kernel'] == 'poly' else 3,
            coef0=params['coef0'] if params['kernel'] in ['poly', 'sigmoid'] else 0.0,
            random_state=self.random_state,
            probability=True
        )
        
        final_model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate on test set
        y_pred = final_model.predict(self.X_test_scaled)
        y_prob = final_model.predict_proba(self.X_test_scaled)[:, 1]
        
        test_metrics = {
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_prob),
            'accuracy': accuracy_score(self.y_test, y_pred),
            'model': final_model,
            'best_params': params
        }
        
        print("\nTest Set Performance:")
        print(f"F1 Score:  {test_metrics['f1_score']:.4f}")
        print(f"ROC AUC:   {test_metrics['roc_auc']:.4f}")
        print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
        
        return test_metrics
    
    def plot_optimization_progress(self):
        """Plot optimization progress."""
        plt.figure(figsize=(10, 6))
        iterations = range(1, len(self.optimization_history) + 1)
        best_scores = [h['best_score'] for h in self.optimization_history]
        
        plt.plot(iterations, best_scores, 'b-', label='Best F1 Score')
        plt.plot(iterations, self.avg_scores_history, 'r--', label='Average F1 Score')
        
        plt.title('PSO Optimization Progress')
        plt.xlabel('Iteration')
        plt.ylabel('F1 Score')
        plt.grid(True)
        plt.legend()
        plt.show()


def load_and_preprocess_data(file_path):
    """Load and preprocess data from CSV file."""
    try:
        # Read CSV with semicolon separator
        df = pd.read_csv(file_path, sep=';', na_values='<Null>')
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Feature columns for flood prediction
        feature_columns = [
            'Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road',
            'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
            'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall'
        ]
        label_column = 'Nom'
        
        # Convert Yes/No to 1/0
        df[label_column] = (df[label_column] == 'Yes').astype(int)
        
        # Replace comma with dot in numeric columns and convert to float
        for col in feature_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
        
        X = df[feature_columns].values
        y = df[label_column].values
        
        return X, y, feature_columns
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return None, None, None
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return None, None, None


def main():
    """Main execution function."""
    # Load data
    X, y, feature_names = load_and_preprocess_data("training.csv")
    if X is None:
        return
    
    # Initialize and run optimizer
    optimizer = PSOSVMOptimizer(
        X=X, 
        y=y, 
        n_particles=30, 
        n_iterations=50,
        random_state=42
    )
    
    # Optimize hyperparameters
    best_params, best_score = optimizer.optimize()
    
    # Plot optimization progress
    optimizer.plot_optimization_progress()
    
    # Evaluate final model
    test_results = optimizer.evaluate_test_performance()
    
    # Save best model
    joblib.dump(test_results['model'], 'pso_svm_model.joblib')
    print("\nModel saved as: pso_svm_model.joblib")


if __name__ == "__main__":
    main()