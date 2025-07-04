import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import time
import joblib


class PSORandomForestOptimizer:
    """Particle Swarm Optimization for Random Forest hyperparameter tuning."""
    
    def __init__(self, X, y, n_particles=15, n_iterations=10, random_state=42):
        """Initialize PSO optimizer."""
        self.X = np.array(X)
        self.y = np.array(y, dtype=int)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.random_state = random_state
        
        # Set random seed
        # np.random.seed(random_state)
        
        # Prepare data
        self._prepare_data()
        
        # PSO parameters (constriction factor approach)
        self.w = 0.729    # Inertia weight
        self.c1 = 1.49445 # Cognitive parameter
        self.c2 = 1.49445 # Social parameter
        
        # Parameter search space (chỉ tối ưu các tham số yêu cầu)
        self.param_ranges = {
            'n_estimators': {'min': 50, 'max': 500}, 
            'min_samples_leaf': {'min': 1, 'max': 20}, 
            'max_samples': {'min': 0.1, 'max': 1},  
            'max_leaf_nodes': {'min': 16, 'max': 256}
        }
        
        # Initialize swarm
        self._initialize_swarm()
        
        # Optimization results
        self.global_best_position = None
        self.global_best_score = -np.inf
        self.optimization_history = []
    
    def _prepare_data(self):
        """Prepare and split data for training."""
        # Handle missing values // nội suy dữ liệu bị thiếu.
        if np.isnan(self.X).any():
            imputer = SimpleImputer(strategy='median')
            self.X = imputer.fit_transform(self.X)
        
        # Split data // chia dữ liệu thành tập huấn luyện và kiểm tra.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, 
            stratify=self.y
        )
        
        # Scale features // chuẩn hóa dữ liệu để đưa về giá trị từ 0 đến 1.
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def _initialize_swarm(self):
        """Initialize particle positions and velocities."""
        n_params = len(self.param_ranges)
        
        self.particles = np.random.uniform(0, 1, (self.n_particles, n_params))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.n_particles, n_params))
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = np.full(self.n_particles, -np.inf)
    
    def _decode_position(self, position):
        """Convert normalized position to Random Forest parameters."""
        params = {}
        param_names = list(self.param_ranges.keys())
        
        for i, param_name in enumerate(param_names):
            if param_name == 'max_features':
                # Categorical parameter
                options = self.param_ranges[param_name]['options']  # lấy danh sách các tùy chọn
                idx = min(int(position[i] * len(options)), len(options) - 1)    # đảm bảo không vượt quá chỉ số
                params[param_name] = options[idx]
            else:
                # Numerical parameter
                min_val = self.param_ranges[param_name]['min']
                max_val = self.param_ranges[param_name]['max']
                params[param_name] = int(min_val + position[i] * (max_val - min_val))
        
        return params
    
    def _evaluate_fitness(self, position):
        """Evaluate particle fitness using cross-validation."""
        try:
            params = self._decode_position(position)
            
            rf = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                random_state=self.random_state,
                class_weight='balanced'
            )
            
            # 3-fold cross-validation for speed
            cv_scores = cross_val_score(
                rf, self.X_train_scaled, self.y_train, 
                cv=3, scoring='f1', n_jobs=-1
            )
            
            return np.mean(cv_scores)
            
        except Exception:
            return -np.inf
    
    def _update_swarm(self):
        """Update particle velocities and positions."""
        for i in range(self.n_particles):
            # Random factors
            r1 = np.random.random(len(self.particles[i]))
            r2 = np.random.random(len(self.particles[i]))
            
            # Update velocity
            inertia = self.w * self.velocities[i]
            cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
            social = self.c2 * r2 * (self.global_best_position - self.particles[i])
            
            self.velocities[i] = inertia + cognitive + social
            
            # Clamp velocity
            self.velocities[i] = np.clip(self.velocities[i], -0.2, 0.2)
            
            # Update position
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], 0, 1)
    
    def optimize(self):
        """Execute PSO optimization algorithm."""
        print("Starting PSO optimization...")
        print(f"Dataset: {len(self.X)} samples, {self.X.shape[1]} features")
        print(f"Class distribution: {np.bincount(self.y)}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Evaluate initial swarm
        for i in range(self.n_particles):
            score = self._evaluate_fitness(self.particles[i])
            self.personal_best_scores[i] = score
            
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[i].copy()
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            # Update swarm
            self._update_swarm()
            
            # Evaluate particles
            current_scores = []
            for i in range(self.n_particles):
                score = self._evaluate_fitness(self.particles[i])
                current_scores.append(score)
                
                # Update personal best
                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i].copy()
                    
                    # Update global best
                    if score > self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.particles[i].copy()
            
            # Log iteration results
            best_params = self._decode_position(self.global_best_position)
            avg_score = np.mean(current_scores)
            
            print(f"Iteration {iteration + 1:2d}/{self.n_iterations}: "
                  f"Best F1={self.global_best_score:.4f}, Avg F1={avg_score:.4f}")
            print(f"  Parameters: {best_params}")
            
            # Store history
            self.optimization_history.append({
                'iteration': iteration + 1,
                'best_score': self.global_best_score,
                'avg_score': avg_score,
                'best_params': best_params.copy()
            })
        
        optimization_time = time.time() - start_time
        
        print("-" * 60)
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Best F1 Score: {self.global_best_score:.4f}")
        print(f"Optimal Parameters: {self._decode_position(self.global_best_position)}")
        
        return self._decode_position(self.global_best_position), self.global_best_score
    
    def evaluate_test_performance(self):
        """Train final model and evaluate on test set."""
        if self.global_best_position is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        best_params = self._decode_position(self.global_best_position)
        
        # Train final model
        final_model = RandomForestClassifier(
            **best_params,
            random_state=self.random_state,
            class_weight='balanced'
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
            'best_params': best_params
        }
        
        print("\nTest Set Performance:")
        print(f"F1 Score:  {test_metrics['f1_score']:.4f}")
        print(f"ROC AUC:   {test_metrics['roc_auc']:.4f}")
        print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
        
        return test_metrics


def load_and_preprocess_data(file_path):
    """
    Load and preprocess data from Excel file.
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        X: Feature matrix
        y: Target labels
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Define feature and target columns
        feature_columns = [
            'Rainfall', 'Elevation', 'Slope', 'Aspect', 'Flow_direction',
            'Flow_accumulation', 'TWI', 'Distance_to_river', 'Drainage_capacity',
            'LandCover', 'Imperviousness', 'Surface_temperature'
        ]
        
        target_column = 'label_column'  # Adjust based on your data
        
        # Validate columns
        missing_cols = [col for col in feature_columns + [target_column] 
                       if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return None, None
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        return X, y
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return None, None
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return None, None


def main():
    """Main execution function."""
    # Configuration
    DATA_FILE = "C:/Users/Admin/Downloads/prj/src/flood_data.xlsx"
    MODEL_OUTPUT = "optimized_flood_model.pkl"
    
    # Load data
    X, y = load_and_preprocess_data(DATA_FILE)
    if X is None or y is None:
        return
    
    # Initialize and run optimizer
    optimizer = PSORandomForestOptimizer(
        X=X, 
        y=y, 
        n_particles=15, 
        n_iterations=10,
        random_state=42
    )
    
    # Optimize hyperparameters
    best_params, best_score = optimizer.optimize()
    
    # Evaluate final model
    test_results = optimizer.evaluate_test_performance()
    
    # # Save model
    # joblib.dump(test_results['model'], MODEL_OUTPUT)
    # print(f"\nModel saved to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()