# GPU is required and available by default
import cudf
import cupy as cp
import xgboost as xgb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import time
import random
import warnings
warnings.filterwarnings('ignore')

# Add constant random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def check_gpu():
    """GPU is available and ready to use"""
    return True


class PSOXGBoostOptimizer:
    """Particle Swarm Optimization for XGBoost hyperparameter tuning for regression."""
    
    def __init__(self, X, y, n_particles=30, n_iterations=50):
        """Initialize PSO optimizer."""
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.has_gpu = check_gpu()
        
        # Prepare data
        self._prepare_data()
        
        # PSO parameters
        self.w = 0.9    # Inertia weight
        self.c1 = 1.5   # Cognitive parameter
        self.c2 = 1.5   # Social parameter
        self.w_min = 0.4 # Minimum inertia weight
        
        # Parameter search space for XGBoost regression
        self.param_ranges = {
            'n_estimators': {'min': 50, 'max': 1000},
            'max_depth': {'min': 3, 'max': 100},
            'learning_rate': {'min': 0.01, 'max': 0.5},
            'subsample': {'min': 0.6, 'max': 1.0},
            'colsample_bytree': {'min': 0.6, 'max': 1.0},
            'reg_alpha': {'min': 0.0, 'max': 1.0},
            'reg_lambda': {'min': 0.0, 'max': 1.0},
            'min_child_weight': {'min': 1, 'max': 100}
        }
        
        # Initialize swarm
        self._initialize_swarm()
        
        # Optimization results
        self.global_best_position = {}
        self.global_best_score = -np.inf
        self.optimization_history = []
        self.avg_scores_history = []
        self.best_metrics = None
        self.metrics_history = []
    
    def _prepare_data(self):
        """Prepare and split data for training."""
        # Handle missing values
        if np.isnan(self.X).any():
            imputer = SimpleImputer(strategy='median')
            self.X = imputer.fit_transform(self.X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def _initialize_swarm(self):
        # Initialize positions randomly
        self.positions = []
        for _ in range(self.n_particles):
            position = {}
            for param, range_info in self.param_ranges.items():
                if param in ['n_estimators', 'max_depth', 'min_child_weight']:
                    position[param] = np.random.randint(range_info['min'], range_info['max'] + 1)
                else:
                    position[param] = np.random.uniform(range_info['min'], range_info['max'])
            self.positions.append(position)
        
        # Initialize velocities randomly
        self.velocities = []
        for i in range(self.n_particles):
            velocity = {}
            for param, range_info in self.param_ranges.items():
                max_velocity = (range_info['max'] - range_info['min']) * 0.1
                velocity[param] = np.random.uniform(-max_velocity, max_velocity)
            self.velocities.append(velocity)
        
        # Initialize personal best
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.n_particles, -np.inf)
        
        # Show parameter ranges
        print("\n📊 Phạm vi tham số:")
        for param, range_info in self.param_ranges.items():
            print(f"   • {param}: [{range_info['min']}, {range_info['max']}]")
    
    def _evaluate_fitness(self, position):
        """Evaluate particle fitness using RMSE as the main metric for regression."""
        try:
            # Configure XGBoost with GPU acceleration
            model = xgb.XGBRegressor(
                n_estimators=int(position['n_estimators']),
                max_depth=int(position['max_depth']),
                learning_rate=position['learning_rate'],
                subsample=position['subsample'],
                colsample_bytree=position['colsample_bytree'],
                reg_alpha=position['reg_alpha'],
                reg_lambda=position['reg_lambda'],
                min_child_weight=int(position['min_child_weight']),
                tree_method='gpu_hist',  # GPU acceleration
                gpu_id=0,
                random_state=RANDOM_SEED,
                verbosity=0
            )
            
            # Fit model and predict
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate only essential metrics
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Store metrics history
            current_metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
            
            # Update best metrics tracking based on RMSE
            if self.best_metrics is None:
                self.best_metrics = current_metrics.copy()
            else:
                if current_metrics['rmse'] < self.best_metrics['rmse']:
                    self.best_metrics = current_metrics.copy()
            
            # Store metrics history
            self.metrics_history.append({
                'iteration': len(self.metrics_history),
                'params': position,
                'metrics': current_metrics
            })
            
            # Return negative RMSE as score (higher is better)
            return -current_metrics['rmse']
            
        except Exception as e:
            print(f"Error evaluating params: {str(e)}")
            return -np.inf
    
    def _update_particle(self, particle_idx):
        """Update particle velocity and position."""
        # Update velocity
        w = self.w - (self.w - self.w_min) * (particle_idx / self.n_particles)
        
        for param, range_info in self.param_ranges.items():
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
            self.positions[particle_idx][param] = np.clip(
                self.positions[particle_idx][param],
                range_info['min'],
                range_info['max']
            )
            
            # Round integer parameters
            if param in ['n_estimators', 'max_depth', 'min_child_weight']:
                self.positions[particle_idx][param] = int(self.positions[particle_idx][param])
    
    def optimize(self):
        """Execute PSO optimization algorithm."""
        print("Iter |  RMSE  |   R²   |  MAE   | Score |  Trend")
        print("-" * 90)
        
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
            
            # Print progress with more detailed information
            if len(self.metrics_history) > 0:
                latest_metrics = self.metrics_history[-1]['metrics']
                
                # Calculate trend
                if iteration > 0 and len(self.metrics_history) >= 2:
                    prev_rmse = self.metrics_history[-2]['metrics']['rmse']
                    current_rmse = latest_metrics['rmse']
                    change = current_rmse - prev_rmse
                    trend = "↑" if change > 0 else "↓" if change < 0 else "→"
                    trend_info = f" {trend}({change:+.4f})"
                else:
                    trend_info = ""
                
                print(f"{iteration+1:4d} | {latest_metrics['rmse']:6.4f} | {latest_metrics['r2']:6.4f} | "
                      f"{latest_metrics['mae']:6.4f} | {-self.global_best_score:6.4f}{trend_info}")
            
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
            
            # Display info every 10 iterations
            if (iteration + 1) % 10 == 0:
                recent_scores = current_scores[-10:]
                print(f"     Last 10 particles: Best={max(recent_scores):.4f}, Avg={np.mean(recent_scores):.4f}, Std={np.std(recent_scores):.4f}")
                print("-" * 90)
        
        optimization_time = time.time() - start_time
        
        print("\n" + "=" * 90)
        print(f"🏆 RMSE tốt nhất: {-self.global_best_score:.4f}")
        print(f"📋 Tham số tối ưu:")
        for param, value in self.global_best_position.items():
            if isinstance(value, float):
                print(f"     {param}: {value:.6f}")
            else:
                print(f"     {param}: {value}")
        
        # Export convergence data to CSV
        convergence_data = pd.DataFrame(self.optimization_history)
        convergence_data.to_csv('pso_xgb_iterations.csv', index=False)
        print(f"\n💾 Dữ liệu hội tụ đã lưu vào 'pso_xgb_iterations.csv'")
        
        return self.global_best_position, self.global_best_score
    
    def evaluate_test_performance(self):
        """Train final model and evaluate on test set."""
        if not self.global_best_position:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        # Train final model with GPU acceleration
        final_model = xgb.XGBRegressor(
            n_estimators=int(self.global_best_position['n_estimators']),
            max_depth=int(self.global_best_position['max_depth']),
            learning_rate=self.global_best_position['learning_rate'],
            subsample=self.global_best_position['subsample'],
            colsample_bytree=self.global_best_position['colsample_bytree'],
            reg_alpha=self.global_best_position['reg_alpha'],
            reg_lambda=self.global_best_position['reg_lambda'],
            min_child_weight=int(self.global_best_position['min_child_weight']),
            tree_method='gpu_hist',  # GPU acceleration
            gpu_id=0,
            random_state=RANDOM_SEED,
            verbosity=0
        )
        
        final_model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate on test set
        y_pred = final_model.predict(self.X_test_scaled)
        
        # Calculate regression metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        test_metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model': final_model,
            'best_params': self.global_best_position
        }
        
        print("\nTest Set Performance:")
        print(f"RMSE:      {test_metrics['rmse']:.4f}")
        print(f"MAE:       {test_metrics['mae']:.4f}")
        print(f"R²:        {test_metrics['r2']:.4f}")
        
        return test_metrics
    
    def plot_optimization_progress(self):
        """Plot optimization progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Best Score progression
        iterations = range(1, len(self.optimization_history) + 1)
        best_scores = [h['best_score'] for h in self.optimization_history]
        
        axes[0, 0].plot(iterations, best_scores, 'b-', label='Best Score (Negative RMSE)')
        axes[0, 0].plot(iterations, self.avg_scores_history, 'r--', label='Average Score')
        axes[0, 0].set_title('PSO Optimization Progress')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Score (Negative RMSE)')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Plot 2: RMSE progression
        rmse_values = [-score for score in best_scores]  # Convert back to positive RMSE
        axes[0, 1].plot(iterations, rmse_values, 'g-', linewidth=2)
        axes[0, 1].set_title('Best RMSE Progression')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].grid(True)
        
        # Plot 3: Population diversity
        pop_means = [h['population_mean_score'] for h in self.optimization_history]
        pop_mins = [h['population_min_score'] for h in self.optimization_history]
        pop_maxs = [h['population_max_score'] for h in self.optimization_history]
        
        axes[1, 0].fill_between(iterations, pop_mins, pop_maxs, alpha=0.3, label='Min-Max Range')
        axes[1, 0].plot(iterations, pop_means, 'r-', label='Population Mean')
        axes[1, 0].plot(iterations, best_scores, 'b-', label='Best Score')
        axes[1, 0].set_title('Population Diversity')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Plot 4: Metrics progression (if available)
        if self.metrics_history:
            metrics_rmse = [m['metrics']['rmse'] for m in self.metrics_history]
            metrics_r2 = [m['metrics']['r2'] for m in self.metrics_history]
            
            ax4 = axes[1, 1]
            ax4.plot(range(1, len(metrics_rmse) + 1), metrics_rmse, 'r-', label='RMSE')
            ax4.set_ylabel('RMSE', color='r')
            ax4.tick_params(axis='y', labelcolor='r')
            
            ax4_twin = ax4.twinx()
            ax4_twin.plot(range(1, len(metrics_r2) + 1), metrics_r2, 'b-', label='R²')
            ax4_twin.set_ylabel('R²', color='b')
            ax4_twin.tick_params(axis='y', labelcolor='b')
            
            ax4.set_title('Metrics Progression')
            ax4.set_xlabel('Evaluation')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def export_detailed_results_to_excel(self, filename='pso_xgb_detailed_results.xlsx'):
        """Export detailed results to Excel with multiple sheets"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Sheet 1: Optimization History
                history_data = []
                for i, hist in enumerate(self.optimization_history):
                    row = {
                        'Iteration': hist['iteration'],
                        'Best_Score': hist['best_score'],
                        'Population_Mean': hist['population_mean_score'],
                        'Population_Min': hist['population_min_score'],
                        'Population_Max': hist['population_max_score'],
                        'Inertia_Weight': hist['inertia_weight'],
                        'Cognitive_Param': hist['cognitive_param'],
                        'Social_Param': hist['social_param']
                    }
                    # Add best parameters
                    for k, v in hist['best_params'].items():
                        row[f'Best_{k}'] = v
                    history_data.append(row)
                
                history_df = pd.DataFrame(history_data)
                history_df.to_excel(writer, sheet_name='Optimization_History', index=False)
                
                # Sheet 2: Metrics History
                if self.metrics_history:
                    metrics_data = []
                    for i, hist in enumerate(self.metrics_history):
                        row = {
                            'Evaluation': i + 1,
                            'RMSE': hist['metrics']['rmse'],
                            'MAE': hist['metrics']['mae'],
                            'R²': hist['metrics']['r2'],
                            'Is_Best': hist['metrics']['rmse'] == min([h['metrics']['rmse'] for h in self.metrics_history])
                        }
                        # Add parameters
                        for k, v in hist['params'].items():
                            row[f'Param_{k}'] = v
                        metrics_data.append(row)
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df.to_excel(writer, sheet_name='Metrics_History', index=False)
                
                # Sheet 3: Final Best Parameters
                if self.global_best_position:
                    best_params_df = pd.DataFrame([self.global_best_position])
                    best_params_df.to_excel(writer, sheet_name='Final_Best_Parameters', index=False)
                
                # Sheet 4: Summary Statistics
                if self.metrics_history:
                    summary_stats = {
                        'Metric': ['RMSE', 'MAE', 'R²'],
                        'Best': [
                            min([h['metrics']['rmse'] for h in self.metrics_history]),
                            min([h['metrics']['mae'] for h in self.metrics_history]),
                            max([h['metrics']['r2'] for h in self.metrics_history])
                        ],
                        'Average': [
                            np.mean([h['metrics']['rmse'] for h in self.metrics_history]),
                            np.mean([h['metrics']['mae'] for h in self.metrics_history]),
                            np.mean([h['metrics']['r2'] for h in self.metrics_history])
                        ],
                        'Std_Dev': [
                            np.std([h['metrics']['rmse'] for h in self.metrics_history]),
                            np.std([h['metrics']['mae'] for h in self.metrics_history]),
                            np.std([h['metrics']['r2'] for h in self.metrics_history])
                        ]
                    }
                    summary_df = pd.DataFrame(summary_stats)
                    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
            print(f"✓ Detailed results exported to {filename}")
     
        except Exception as e:
            print(f"❌ Error exporting to Excel: {e}")
            print("Make sure you have openpyxl installed: pip install openpyxl")

def load_and_preprocess_data():
    # Load data
    df = pd.read_csv('/kaggle/input/flood-trainning/flood_training.csv', sep=';', na_values='<Null>')
    
    # Feature columns for flood prediction
    feature_columns = [
        'Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road',
        'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
        'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall'
    ]
    label_column = 'Nom'
    
    # Convert Yes/No to 1/0 for regression
    df[label_column] = (df[label_column] == 'Yes').astype(float)
    
    # Replace comma with dot in numeric columns and convert to float
    for col in feature_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)
    
    # Fill missing values with mean
    df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
    
    X = df[feature_columns].values
    y = df[label_column].values
    
    # Handle any remaining missing values
    if np.isnan(X).any():
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    return X, y, feature_columns


def main():
    """Main execution function for PSO-XGBoost optimization"""
    print("XGBoost PSO Optimization - Regression")
    print("=" * 50)
    
    # Load data
    X, y, feature_names = load_and_preprocess_data()
    
    # Initialize and run optimizer
    print("\nStarting XGBoost PSO Optimization...")
    optimizer = PSOXGBoostOptimizer(
        X=X, 
        y=y, 
        n_particles=25, 
        n_iterations=100
    )
    
    # Optimize hyperparameters
    best_params, best_score = optimizer.optimize()
    
    # Display final results
    print("\n" + "=" * 50)
    print("FINAL BEST RESULTS:")
    if optimizer.best_metrics:
        print(f"RMSE: {optimizer.best_metrics['rmse']:.4f}")
        print(f"MAE: {optimizer.best_metrics['mae']:.4f}")
        print(f"R²: {optimizer.best_metrics['r2']:.4f}")
    
    # Plot optimization progress
    optimizer.plot_optimization_progress()
    
    # Evaluate final model
    test_results = optimizer.evaluate_test_performance()
    
    # Save results
    print("\nSaving results...")
    
    # Save best parameters
    params_df = pd.DataFrame([best_params])
    params_df.to_csv('pso_xgb_best_params.csv', index=False)
    
    # Save final metrics
    if test_results:
        metrics_df = pd.DataFrame([test_results])
        metrics_df.to_csv('pso_xgb_final_metrics.csv', index=False)
    
    # Save optimization history
    history_data = []
    for i, hist in enumerate(optimizer.optimization_history):
        row = {
            'iteration': hist['iteration'],
            'best_score': hist['best_score'],
            'population_mean_score': hist['population_mean_score'],
            'population_min_score': hist['population_min_score'],
            'population_max_score': hist['population_max_score'],
            'inertia_weight': hist['inertia_weight'],
            'cognitive_param': hist['cognitive_param'],
            'social_param': hist['social_param']
        }
        # Add best parameters
        row.update({f'best_{k}': v for k, v in hist['best_params'].items()})
        history_data.append(row)
    
    history_df = pd.DataFrame(history_data)
    history_df.to_csv('pso_xgb_optimization_history.csv', index=False)
    
    # Export detailed results to Excel
    optimizer.export_detailed_results_to_excel('pso_xgb_detailed_results.xlsx')
    
    print("\nOptimization completed successfully!")


if __name__ == "__main__":
    main()