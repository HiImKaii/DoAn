import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import time

class XGBPSOOptimizer:
    def __init__(self, X, y, n_particles=20, n_iterations=20):
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
        
        # XGBoost parameter ranges (normalized to [0, 1])
        self.param_ranges = {
            'n_estimators': {'min': 50, 'max': 300},
            'max_depth': {'min': 3, 'max': 10},
            'learning_rate': {'min': 0.01, 'max': 0.3},
            'subsample': {'min': 0.6, 'max': 1.0},
            'colsample_bytree': {'min': 0.6, 'max': 1.0},
            'reg_alpha': {'min': 0, 'max': 1.0},
            'reg_lambda': {'min': 1, 'max': 10}
        }
        
        # PSO parameters
        self.w = 0.729  # Inertia weight
        self.c1 = 1.49445  # Cognitive parameter
        self.c2 = 1.49445  # Social parameter
        
        # Initialize swarm
        self.particles = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_scores = []
        self.global_best_position = None
        self.global_best_score = -np.inf
        
        # Initialize particles
        self._initialize_swarm()
    
    def _initialize_swarm(self):
        """Initialize particle swarm"""
        n_params = len(self.param_ranges)
        
        for _ in range(self.n_particles):
            # Random position in [0, 1] for each parameter
            position = np.random.uniform(0, 1, n_params)
            velocity = np.random.uniform(-0.1, 0.1, n_params)  # Small initial velocities
            
            self.particles.append(position)
            self.velocities.append(velocity)
            self.personal_best_positions.append(position.copy())
            self.personal_best_scores.append(-np.inf)
    
    def _position_to_params(self, position):
        """Convert normalized position to actual XGBoost parameters"""
        params = {}
        param_names = list(self.param_ranges.keys())
        
        for i, param_name in enumerate(param_names):
            min_val = self.param_ranges[param_name]['min']
            max_val = self.param_ranges[param_name]['max']
            actual_val = min_val + position[i] * (max_val - min_val)
            
            if param_name in ['n_estimators', 'max_depth']:
                params[param_name] = int(actual_val)
            else:
                params[param_name] = actual_val
        
        return params
    
    def _evaluate_particle(self, position):
        """Evaluate a particle's fitness"""
        try:
            params = self._position_to_params(position)
            
            xgb = XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )
            
            # Use stratified K-fold CV for evaluation
            cv_scores = cross_val_score(xgb, self.X_train_scaled, self.y_train, 
                                      cv=3, scoring='f1')
            
            return float(np.mean(cv_scores))
            
        except Exception as e:
            return -np.inf
    
    def _update_velocity(self, particle_idx):
        """Update particle velocity using PSO formula"""
        current_velocity = self.velocities[particle_idx]
        current_position = self.particles[particle_idx]
        personal_best = self.personal_best_positions[particle_idx]
        
        # Random factors
        r1 = np.random.random(len(current_position))
        r2 = np.random.random(len(current_position))
        
        # PSO velocity update formula
        inertia = self.w * current_velocity
        cognitive = self.c1 * r1 * (personal_best - current_position)
        social = self.c2 * r2 * (self.global_best_position - current_position)
        
        new_velocity = inertia + cognitive + social
        
        # Velocity clamping to prevent explosion
        velocity_max = 0.2  # Maximum velocity as fraction of search space
        new_velocity = np.clip(new_velocity, -velocity_max, velocity_max)
        
        self.velocities[particle_idx] = new_velocity
    
    def _update_position(self, particle_idx):
        """Update particle position"""
        new_position = self.particles[particle_idx] + self.velocities[particle_idx]
        
        # Boundary handling - reflect particles that go out of bounds
        new_position = np.clip(new_position, 0, 1)
        
        self.particles[particle_idx] = new_position
    
    def optimize(self):
        """Main PSO optimization algorithm"""
        print("Starting PSO optimization for XGBoost...")
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
        
        # Evaluate initial particles
        for i in range(self.n_particles):
            score = self._evaluate_particle(self.particles[i])
            self.personal_best_scores[i] = score
            
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[i].copy()
        
        print(f"Initial best score: {self.global_best_score:.4f}")
        
        # Main PSO loop
        for iteration in range(self.n_iterations):
            print(f"\nIteration {iteration + 1}/{self.n_iterations}")
            
            # Update each particle
            for i in range(self.n_particles):
                # Update velocity and position
                self._update_velocity(i)
                self._update_position(i)
                
                # Evaluate new position
                score = self._evaluate_particle(self.particles[i])
                
                # Update personal best
                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i].copy()
                    
                    # Update global best
                    if score > self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.particles[i].copy()
            
            # Print results for this iteration
            best_params = self._position_to_params(self.global_best_position)
            print(f"Best score: {self.global_best_score:.4f}")
            print(f"Best parameters:")
            for param, value in best_params.items():
                if param in ['n_estimators', 'max_depth']:
                    print(f"  {param}: {value}")
                else:
                    print(f"  {param}: {value:.4f}")
        
        print("\n" + "=" * 50)
        print("PSO Optimization completed!")
        
        if self.global_best_position is not None:
            best_params = self._position_to_params(self.global_best_position)
            print(f"\nBest solution found:")
            print(f"Score: {self.global_best_score:.4f}")
            print("Parameters:")
            for param, value in best_params.items():
                if param in ['n_estimators', 'max_depth']:
                    print(f"  {param}: {value}")
                else:
                    print(f"  {param}: {value:.4f}")
            
            return best_params, self.global_best_score
        else:
            return None, -np.inf
    
    def evaluate_final_model(self):
        """Evaluate final model on test set"""
        if self.global_best_position is None:
            print("No optimized model available!")
            return None
        
        # Get best parameters
        best_params = self._position_to_params(self.global_best_position)
        
        # Train model with best parameters
        best_xgb = XGBClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            reg_alpha=best_params['reg_alpha'],
            reg_lambda=best_params['reg_lambda'],
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        best_xgb.fit(self.X_train_scaled, self.y_train)
        
        # Predict on test set
        y_pred = best_xgb.predict(self.X_test_scaled)
        y_prob = best_xgb.predict_proba(self.X_test_scaled)
        
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
        
        return {
            'model': best_xgb,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': best_params
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
        
        # Initialize and run PSO optimizer
        optimizer = XGBPSOOptimizer(X, y, n_particles=15, n_iterations=10)
        
        start_time = time.time()
        best_params, best_score = optimizer.optimize()
        end_time = time.time()
        
        print(f"\nOptimization time: {end_time - start_time:.2f} seconds")
        
        if best_params is not None:
            # Evaluate final model
            print("\nEvaluating model on test set:")
            final_results = optimizer.evaluate_final_model()
        else:
            print("\nPSO optimization failed to find valid parameters.")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please ensure your Excel file exists at the specified path")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()