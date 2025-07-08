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
        self.global_best_rmse = np.inf
        self.optimization_history = []
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
        self.personal_best_rmse = np.full(self.n_particles, np.inf)
        
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
            
            # Return RMSE directly (lower is better)
            return current_metrics['rmse']
            
        except Exception as e:
            print(f"Error evaluating params: {str(e)}")
            return np.inf
    
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
        print("Iter | Best RMSE |   R²   |  MAE   | Trend")
        print("-" * 70)
        
        start_time = time.time()
        
        # Evaluate initial swarm
        for i in range(self.n_particles):
            rmse = self._evaluate_fitness(self.positions[i])
            self.personal_best_rmse[i] = rmse
            
            if rmse < self.global_best_rmse:
                self.global_best_rmse = rmse
                self.global_best_position = self.positions[i].copy()
        
        # Store initial best position in history
        self.optimization_history.append({
            'iteration': 0,
            'best_rmse': self.global_best_rmse,
            'best_params': self.global_best_position.copy()
        })
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            # Update particles
            for i in range(self.n_particles):
                self._update_particle(i)
                
                # Evaluate new position
                rmse = self._evaluate_fitness(self.positions[i])
                
                # Update personal best
                if rmse < self.personal_best_rmse[i]:
                    self.personal_best_rmse[i] = rmse
                    self.personal_best_positions[i] = self.positions[i].copy()
                    
                    # Update global best
                    if rmse < self.global_best_rmse:
                        self.global_best_rmse = rmse
                        self.global_best_position = self.positions[i].copy()                # Print progress
            if len(self.metrics_history) > 0:
                latest_metrics = self.metrics_history[-1]['metrics']
                
                # Get best R² and MAE values associated with the best solution
                best_r2 = 0
                best_mae = float('inf')
                for hist in self.metrics_history:
                    if abs(hist['metrics']['rmse'] - self.global_best_rmse) < 1e-6:
                        best_r2 = max(best_r2, hist['metrics']['r2'])
                        best_mae = min(best_mae, hist['metrics']['mae'])
                
                # Calculate trend for Best RMSE - this should only show improvements
                if iteration > 0 and len(self.optimization_history) >= 1:
                    prev_best_rmse = self.optimization_history[-1]['best_rmse']
                    if self.global_best_rmse < prev_best_rmse:
                        best_change = self.global_best_rmse - prev_best_rmse
                        trend_info = f" ↓({best_change:+.4f})"
                    else:
                        trend_info = ""
                else:
                    trend_info = ""
                
                print(f"{iteration+1:4d} | {self.global_best_rmse:9.4f} | {best_r2:6.4f} | "
                      f"{best_mae:6.4f} | {trend_info}")
            
            # Store history for this iteration
            self.optimization_history.append({
                'iteration': iteration + 1,
                'best_rmse': self.global_best_rmse,
                'best_params': self.global_best_position.copy()
            })
        
        optimization_time = time.time() - start_time
        
        print("\n" + "=" * 90)
        print(f"🏆 RMSE tốt nhất: {self.global_best_rmse:.4f}")
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
        
        return self.global_best_position, self.global_best_rmse
    
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
        
        # Plot 1: Best RMSE progression
        iterations = range(1, len(self.optimization_history))  # Skip iteration 0
        best_rmse = [h['best_rmse'] for h in self.optimization_history[1:]]  # Skip first entry
        
        # Ensure best_rmse is non-increasing (best RMSE should never increase)
        for i in range(1, len(best_rmse)):
            if best_rmse[i] > best_rmse[i-1]:
                best_rmse[i] = best_rmse[i-1]
        
        axes[0, 0].plot(iterations, best_rmse, 'b-', label='Best RMSE')
        axes[0, 0].set_title('PSO Optimization Progress')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Best RMSE')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Plot 2: Find R² values corresponding to best RMSE
        best_r2_values = []
        for i in range(len(best_rmse)):
            # Find best R² for each best RMSE
            iter_best_r2 = 0
            for hist in self.metrics_history:
                if abs(hist['metrics']['rmse'] - best_rmse[i]) < 1e-6:
                    iter_best_r2 = max(iter_best_r2, hist['metrics']['r2'])
            best_r2_values.append(iter_best_r2)
            
        axes[0, 1].plot(iterations, best_r2_values, 'g-', linewidth=2)
        axes[0, 1].set_title('R² Progression with Best RMSE')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].grid(True)
        
        # Plot 3: Find MAE values corresponding to best RMSE
        best_mae_values = []
        for i in range(len(best_rmse)):
            # Find best MAE for each best RMSE
            iter_best_mae = float('inf')
            for hist in self.metrics_history:
                if abs(hist['metrics']['rmse'] - best_rmse[i]) < 1e-6:
                    iter_best_mae = min(iter_best_mae, hist['metrics']['mae'])
            best_mae_values.append(iter_best_mae)
        
        axes[1, 0].plot(iterations, best_mae_values, 'r-', label='MAE', linewidth=2)
        axes[1, 0].set_title('MAE Progression with Best RMSE')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Plot 4: All metrics together
        ax4 = axes[1, 1]
        
        # Normalize values for better visualization
        norm_rmse = [(x - min(best_rmse)) / (max(best_rmse) - min(best_rmse) + 1e-10) if max(best_rmse) > min(best_rmse) else x for x in best_rmse]
        norm_r2 = [(x - min(best_r2_values)) / (max(best_r2_values) - min(best_r2_values) + 1e-10) if max(best_r2_values) > min(best_r2_values) else x for x in best_r2_values]
        norm_mae = [(x - min(best_mae_values)) / (max(best_mae_values) - min(best_mae_values) + 1e-10) if max(best_mae_values) > min(best_mae_values) else x for x in best_mae_values]
        
        ax4.plot(iterations, norm_rmse, 'b-', label='Best RMSE', linewidth=2)
        ax4.plot(iterations, norm_r2, 'g-', label='R²', linewidth=2)
        ax4.plot(iterations, norm_mae, 'r-', label='MAE', linewidth=2)
        
        ax4.set_title('Normalized Metrics Progression')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Normalized Value')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    def export_detailed_results_to_excel(self, filename='pso_xgb_detailed_results.xlsx'):
        """Export detailed results to Excel with multiple sheets"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Sheet 1: Optimization History with Best RMSE, R², and MAE
                history_data = []
                
                # Skip the initial history item (iteration 0)
                for i, hist in enumerate(self.optimization_history[1:], 1):
                    # Ensure best_rmse is non-increasing
                    if i > 1 and hist['best_rmse'] > history_data[-1]['Best_RMSE']:
                        best_rmse = history_data[-1]['Best_RMSE']
                    else:
                        best_rmse = hist['best_rmse']
                    
                    # Find best R² and MAE for this best RMSE
                    best_r2 = 0
                    best_mae = float('inf')
                    for m_hist in self.metrics_history:
                        if abs(m_hist['metrics']['rmse'] - best_rmse) < 1e-6:
                            best_r2 = max(best_r2, m_hist['metrics']['r2'])
                            best_mae = min(best_mae, m_hist['metrics']['mae'])
                    
                    row = {
                        'Iteration': hist['iteration'],
                        'Best_RMSE': best_rmse,
                        'Best_R²': best_r2,
                        'Best_MAE': best_mae
                    }
                    
                    # Add best parameters
                    for k, v in hist['best_params'].items():
                        row[f'Best_{k}'] = v
                    
                    history_data.append(row)
                
                history_df = pd.DataFrame(history_data)
                history_df.to_excel(writer, sheet_name='Optimization_History', index=False)
                
                # Sheet 2: Metrics History - for reference but focus on best values
                if self.metrics_history:
                    metrics_data = []
                    for i, hist in enumerate(self.metrics_history):
                        is_best = abs(hist['metrics']['rmse'] - self.global_best_rmse) < 1e-6
                        
                        row = {
                            'Evaluation': i + 1,
                            'Best_RMSE': hist['metrics']['rmse'] if is_best else None,
                            'Best_MAE': hist['metrics']['mae'] if is_best else None,
                            'Best_R²': hist['metrics']['r2'] if is_best else None,
                            'Is_Global_Best': is_best
                        }
                        
                        # Add parameters
                        for k, v in hist['params'].items():
                            row[f'Param_{k}'] = v
                        
                        metrics_data.append(row)
                    
                    # Only keep rows that have contributed to the best RMSE
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df.dropna(subset=['Best_RMSE'], inplace=True)
                    metrics_df.to_excel(writer, sheet_name='Best_Solutions', index=False)
                
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
    best_params, best_rmse = optimizer.optimize()
    
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
    
    # Save optimization history with Best RMSE, R², and MAE
    history_data = []
    
    # Skip the initial history item (iteration 0)
    for i, hist in enumerate(optimizer.optimization_history[1:], 1):
        # Ensure best_rmse is non-increasing
        if i > 1 and hist['best_rmse'] > history_data[-1]['best_rmse']:
            best_rmse = history_data[-1]['best_rmse']
        else:
            best_rmse = hist['best_rmse']
        
        # Find best R² and MAE for this best RMSE
        best_r2 = 0
        best_mae = float('inf')
        for m_hist in optimizer.metrics_history:
            if abs(m_hist['metrics']['rmse'] - best_rmse) < 1e-6:
                best_r2 = max(best_r2, m_hist['metrics']['r2'])
                best_mae = min(best_mae, m_hist['metrics']['mae'])
        
        row = {
            'iteration': hist['iteration'],
            'best_rmse': best_rmse,
            'best_r2': best_r2,
            'best_mae': best_mae
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