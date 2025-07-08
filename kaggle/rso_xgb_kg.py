import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Add constant random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class XGBRandomizedSearch:
    def __init__(self, X, y, n_iterations=200):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_iterations = n_iterations
        self.best_params = None
        self.best_score = -np.inf
        self.best_scores_history = []
        self.metrics_history = []
        self.best_metrics = None
        self.best_params_history = []  # New array to store best parameters for each iteration
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # Scale data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # XGBoost parameter ranges
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 1000},
            'max_depth': {'type': 'int', 'min': 3, 'max': 100},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.5},
            'subsample': {'type': 'float', 'min': 0.6, 'max': 1.0},
            'colsample_bytree': {'type': 'float', 'min': 0.6, 'max': 1.0},
            'reg_alpha': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'reg_lambda': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'min_child_weight': {'type': 'int', 'min': 1, 'max': 100}
        }
    
    def create_random_params(self):
        """Create random parameter set for randomized search"""
        params = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                params[param] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'float':
                params[param] = random.uniform(range_info['min'], range_info['max'])
        return params

    def evaluate_params(self, params):
        """Evaluate parameter set using RMSE as the main metric"""
        try:
            model = xgb.XGBRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                min_child_weight=params['min_child_weight'],
                random_state=RANDOM_SEED,
                verbosity=0
            )
            
            # Cross-validation with MSE scoring
            mse_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=3, scoring='neg_mean_squared_error')
            rmse_cv = np.sqrt(-np.mean(mse_scores))
            
            # Fit model to get additional metrics
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate regression metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            # Calculate additional metrics for monitoring
            n = len(self.y_test)
            p = self.X.shape[1]  # number of features
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            
            # Store metrics history
            current_metrics = {
                'r2': float(r2),
                'adjusted_r2': float(adjusted_r2),
                'rmse': float(rmse),
                'rmse_cv': float(rmse_cv),
                'mae': float(mae)
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
                'params': params,
                'metrics': current_metrics
            })
            
            # Return negative RMSE as score (higher is better)
            return -current_metrics['rmse']
            
        except Exception as e:
            print(f"Error evaluating params: {str(e)}")
            return -np.inf
    
    def search(self):
        """Main randomized search algorithm"""
        print("Starting XGBoost Randomized Search for Regression...")
        print(f"Data: {len(self.X)} points, {self.X.shape[1]} features")
        print(f"Number of iterations: {self.n_iterations}")
        print(f"Target range: [{np.min(self.y):.4f}, {np.max(self.y):.4f}]")
        print(f"Target mean: {np.mean(self.y):.4f}, std: {np.std(self.y):.4f}")
        
        print("-" * 90)
        print("Iter |  RMSE  |   R²   | Adj.R² |  MAE   | Score |  Trend")
        print("-" * 90)
        
        # Random search loop
        for iteration in range(self.n_iterations):
            # Generate random parameters
            params = self.create_random_params()
            
            # Evaluate parameters
            score = self.evaluate_params(params)
            
            # Update best if improved
            improved = False
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                improved = True
                print("*** NEW BEST FOUND! ***")
            
            # Store best parameters history (only update when better, otherwise keep previous)
            if len(self.best_params_history) == 0:
                # First iteration - store current params
                self.best_params_history.append(params.copy())
            else:
                # Check if current score is better than best so far
                if improved:
                    # Better score - update with current params
                    self.best_params_history.append(params.copy())
                else:
                    # Not better - keep previous best params
                    self.best_params_history.append(self.best_params_history[-1].copy())
            
            # Store best score history
            self.best_scores_history.append(max(self.best_score, score))
            
            # Print progress with more detailed information
            latest_metrics = self.metrics_history[-1]['metrics']
            
            # Tính toán xu hướng
            if iteration > 0:
                prev_rmse = self.metrics_history[-2]['metrics']['rmse']
                current_rmse = latest_metrics['rmse']
                change = current_rmse - prev_rmse
                trend = "↑" if change > 0 else "↓" if change < 0 else "→"
                trend_info = f" {trend}({change:+.4f})"
            else:
                trend_info = ""
            
            print(f"{iteration+1:4d} | {latest_metrics['rmse']:6.4f} | {latest_metrics['r2']:6.4f} | "
                  f"{latest_metrics['adjusted_r2']:6.4f} | {latest_metrics['mae']:6.4f} | {-score:6.4f}{trend_info}")
            
            # Hiển thị thông tin mỗi 25 vòng lặp
            if (iteration + 1) % 25 == 0:
                recent_rmse = [h['metrics']['rmse'] for h in self.metrics_history[-25:]]
                print(f"     Last 25 iterations: Best={min(recent_rmse):.4f}, Avg={np.mean(recent_rmse):.4f}, Std={np.std(recent_rmse):.4f}")
                print("-" * 90)
        
        print("\n" + "=" * 90)
        print("Randomized Search completed!")
        if self.best_params is not None:
            print(f"\nBest RMSE: {-self.best_score:.4f}")
            print("Best parameters:")
            for param, value in self.best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.4f}")
                else:
                    print(f"  {param}: {value}")
            
            # Return best parameters and final results from search
            return self.best_params, self.best_metrics
        
        return None, None
    
    def plot_optimization_progress(self):
        """Plot optimization progress including best parameters evolution"""
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Best RMSE progression (cumulative)
        plt.subplot(3, 3, 1)
        best_rmse_progression = []
        current_best_rmse = float('inf')
        
        for hist in self.metrics_history:
            rmse = hist['metrics']['rmse']
            if rmse < current_best_rmse:
                current_best_rmse = rmse
            best_rmse_progression.append(current_best_rmse)
        
        plt.plot(best_rmse_progression, 'b-', linewidth=2, label='Best RMSE')
        plt.title('Best RMSE Progression (Cumulative)')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: RMSE of each individual iteration
        plt.subplot(3, 3, 2)
        all_rmse = [m['metrics']['rmse'] for m in self.metrics_history]
        plt.plot(all_rmse, 'r-', alpha=0.7, label='RMSE per iteration')
        plt.plot(best_rmse_progression, 'b-', linewidth=2, label='Best RMSE')
        
        plt.title('RMSE Each Iteration vs Best RMSE')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: All metrics progression
        plt.subplot(3, 3, 3)
        metrics_to_plot = ['rmse', 'mae', 'r2']
        colors = ['r', 'g', 'b']
        
        for metric, color in zip(metrics_to_plot, colors):
            values = [m['metrics'][metric] for m in self.metrics_history]
            plt.plot(values, f'{color}-', label=metric.upper(), alpha=0.7)
        
        plt.title('All Metrics Progression')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4-9: Best parameters evolution for all parameters
        param_names = list(self.param_ranges.keys())
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, param_name in enumerate(param_names):
            plt.subplot(3, 3, i + 4)
            
            # Get best parameter values for each iteration
            best_param_values = [params[param_name] for params in self.best_params_history]
            
            plt.plot(best_param_values, color=colors[i % len(colors)], 
                    linewidth=2, label=f'Best {param_name}', marker='o', markersize=2)
            
            # Also plot current iteration values for comparison
            current_param_values = [m['params'][param_name] for m in self.metrics_history]
            plt.plot(current_param_values, color=colors[i % len(colors)], 
                    alpha=0.3, label=f'Current {param_name}', linewidth=1)
            
            plt.title(f'Best {param_name} Evolution')
            plt.xlabel('Iteration')
            plt.ylabel(param_name)
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            
            # Break if we've plotted 6 parameters (to fit in 3x3 grid)
            if i >= 5:
                break
        
        plt.tight_layout()
        plt.show()
        
        # Create a second figure for remaining parameters if there are more than 6
        if len(param_names) > 6:
            plt.figure(figsize=(12, 8))
            remaining_params = param_names[6:]
            
            for i, param_name in enumerate(remaining_params):
                plt.subplot(2, 2, i + 1)
                
                # Get best parameter values for each iteration
                best_param_values = [params[param_name] for params in self.best_params_history]
                
                plt.plot(best_param_values, color=colors[(i + 6) % len(colors)], 
                        linewidth=2, label=f'Best {param_name}', marker='o', markersize=2)
                
                # Also plot current iteration values for comparison
                current_param_values = [m['params'][param_name] for m in self.metrics_history]
                plt.plot(current_param_values, color=colors[(i + 6) % len(colors)], 
                        alpha=0.3, label=f'Current {param_name}', linewidth=1)
                
                plt.title(f'Best {param_name} Evolution')
                plt.xlabel('Iteration')
                plt.ylabel(param_name)
                plt.legend(fontsize=8)
                plt.grid(True, alpha=0.3)
                
                # Break if we've plotted 4 more parameters
                if i >= 3:
                    break
            
            plt.tight_layout()
            plt.show()
        
        # Create histogram plot for RMSE distribution
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        all_rmse = [m['metrics']['rmse'] for m in self.metrics_history]
        plt.hist(all_rmse, bins=min(30, len(all_rmse)//5), alpha=0.7, edgecolor='black', color='skyblue')
        best_rmse = min(all_rmse)
        plt.axvline(best_rmse, color='red', linestyle='--', linewidth=2, label=f'Best: {best_rmse:.4f}')
        plt.axvline(np.mean(all_rmse), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_rmse):.4f}')
        plt.title('RMSE Distribution (All Iterations)')
        plt.xlabel('RMSE')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        best_rmse_values = [params_hist for params_hist in best_rmse_progression]
        plt.plot(best_rmse_values, 'b-', linewidth=2, marker='o', markersize=3)
        plt.title('Best RMSE Evolution (Detailed)')
        plt.xlabel('Iteration')
        plt.ylabel('Best RMSE')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Analyze RMSE variation
        self.analyze_rmse_variation()
    
    def analyze_rmse_variation(self):
        """Phân tích sự biến động của RMSE"""
        if not self.metrics_history:
            return
            
        all_rmse = [m['metrics']['rmse'] for m in self.metrics_history]
        
        print("\n" + "=" * 60)
        print("PHÂN TÍCH SỰ BIẾN ĐỘNG RMSE")
        print("=" * 60)
        
        # Thống kê cơ bản
        print(f"RMSE tốt nhất: {min(all_rmse):.6f}")
        print(f"RMSE trung bình: {np.mean(all_rmse):.6f}")
        print(f"RMSE tệ nhất: {max(all_rmse):.6f}")
        print(f"Độ lệch chuẩn: {np.std(all_rmse):.6f}")
        print(f"Phương sai: {np.var(all_rmse):.6f}")
        
        # Phân tích xu hướng
        first_half = all_rmse[:len(all_rmse)//2]
        second_half = all_rmse[len(all_rmse)//2:]
        
        print(f"\nXU HƯỚNG:")
        print(f"RMSE trung bình nửa đầu: {np.mean(first_half):.6f}")
        print(f"RMSE trung bình nửa cuối: {np.mean(second_half):.6f}")
        
        if np.mean(second_half) < np.mean(first_half):
            print("✓ Xu hướng cải thiện theo thời gian")
        else:
            print("⚠ Không có xu hướng cải thiện rõ ràng")
        
        # Tìm những vòng lặp tốt nhất
        best_iterations = []
        for i, hist in enumerate(self.metrics_history):
            if hist['metrics']['rmse'] == min(all_rmse):
                best_iterations.append(i + 1)
        
        print(f"\nVòng lặp tốt nhất: {best_iterations}")
        
        # Phân tích sự dao động lớn
        rmse_changes = []
        for i in range(1, len(all_rmse)):
            change = abs(all_rmse[i] - all_rmse[i-1])
            rmse_changes.append(change)
        
        avg_change = np.mean(rmse_changes)
        print(f"\nĐộ dao động trung bình giữa các vòng lặp: {avg_change:.6f}")
        
        # Tìm những vòng lặp có dao động lớn
        large_changes = []
        for i, change in enumerate(rmse_changes):
            if change > 3 * avg_change:  # Dao động lớn hơn 3 lần trung bình
                large_changes.append((i + 2, change))  # i+2 vì bắt đầu từ vòng lặp thứ 2
        
        if large_changes:
            print(f"\nCác vòng lặp có dao động lớn:")
            for iter_num, change in large_changes[:10]:  # Hiển thị tối đa 10 vòng lặp
                print(f"  Vòng lặp {iter_num}: thay đổi {change:.6f}")
        else:
            print("\nKhông có dao động bất thường nào được phát hiện")
        
        print("=" * 60)
    
    def export_detailed_results_to_excel(self, filename='rso_xgb_detailed_results.xlsx'):
        """Export detailed results to Excel with multiple sheets including best parameters history"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Sheet 1: Iteration History
                history_data = []
                for i, hist in enumerate(self.metrics_history):
                    row = {
                        'Iteration': i + 1,
                        'RMSE': hist['metrics']['rmse'],
                        'R²': hist['metrics']['r2'],
                        'MAE': hist['metrics']['mae'],
                        'Adjusted_R²': hist['metrics']['adjusted_r2'],
                        'RMSE_CV': hist['metrics']['rmse_cv'],
                        'Is_Best': hist['metrics']['rmse'] == min([h['metrics']['rmse'] for h in self.metrics_history])
                    }
                    # Add current iteration parameters
                    for k, v in hist['params'].items():
                        row[f'Current_{k}'] = v
                    history_data.append(row)
                
                history_df = pd.DataFrame(history_data)
                history_df.to_excel(writer, sheet_name='Iteration_History', index=False)
                
                # Sheet 2: Best Parameters History
                best_params_history_data = []
                for i, best_params in enumerate(self.best_params_history):
                    row = {'Iteration': i + 1}
                    for k, v in best_params.items():
                        row[f'Best_{k}'] = v
                    # Add corresponding metrics for this iteration
                    if i < len(self.metrics_history):
                        row['RMSE'] = self.metrics_history[i]['metrics']['rmse']
                        row['R²'] = self.metrics_history[i]['metrics']['r2']
                        row['MAE'] = self.metrics_history[i]['metrics']['mae']
                    best_params_history_data.append(row)
                
                best_params_history_df = pd.DataFrame(best_params_history_data)
                best_params_history_df.to_excel(writer, sheet_name='Best_Parameters_History', index=False)
                
                # Sheet 3: Combined Parameters Comparison
                combined_data = []
                for i in range(len(self.metrics_history)):
                    row = {
                        'Iteration': i + 1,
                        'RMSE': self.metrics_history[i]['metrics']['rmse'],
                        'R²': self.metrics_history[i]['metrics']['r2'],
                        'MAE': self.metrics_history[i]['metrics']['mae'],
                        'Adjusted_R²': self.metrics_history[i]['metrics']['adjusted_r2'],
                        'RMSE_CV': self.metrics_history[i]['metrics']['rmse_cv'],
                    }
                    
                    # Add current iteration parameters
                    for k, v in self.metrics_history[i]['params'].items():
                        row[f'Current_{k}'] = v
                    
                    # Add best parameters for this iteration
                    if i < len(self.best_params_history):
                        for k, v in self.best_params_history[i].items():
                            row[f'Best_{k}'] = v
                    
                    combined_data.append(row)
                
                combined_df = pd.DataFrame(combined_data)
                combined_df.to_excel(writer, sheet_name='Combined_Parameters', index=False)
                
                # Sheet 4: Final Best Parameters
                if self.best_params:
                    best_params_df = pd.DataFrame([self.best_params])
                    best_params_df.to_excel(writer, sheet_name='Final_Best_Parameters', index=False)
                
                # Sheet 5: Summary Statistics
                summary_stats = {
                    'Metric': ['RMSE', 'R²', 'MAE', 'Adjusted_R²', 'RMSE_CV'],
                    'Best': [
                        min([h['metrics']['rmse'] for h in self.metrics_history]),
                        max([h['metrics']['r2'] for h in self.metrics_history]),
                        min([h['metrics']['mae'] for h in self.metrics_history]),
                        max([h['metrics']['adjusted_r2'] for h in self.metrics_history]),
                        min([h['metrics']['rmse_cv'] for h in self.metrics_history])
                    ],
                    'Average': [
                        np.mean([h['metrics']['rmse'] for h in self.metrics_history]),
                        np.mean([h['metrics']['r2'] for h in self.metrics_history]),
                        np.mean([h['metrics']['mae'] for h in self.metrics_history]),
                        np.mean([h['metrics']['adjusted_r2'] for h in self.metrics_history]),
                        np.mean([h['metrics']['rmse_cv'] for h in self.metrics_history])
                    ],
                    'Std_Dev': [
                        np.std([h['metrics']['rmse'] for h in self.metrics_history]),
                        np.std([h['metrics']['r2'] for h in self.metrics_history]),
                        np.std([h['metrics']['mae'] for h in self.metrics_history]),
                        np.std([h['metrics']['adjusted_r2'] for h in self.metrics_history]),
                        np.std([h['metrics']['rmse_cv'] for h in self.metrics_history])
                    ]
                }
                summary_df = pd.DataFrame(summary_stats)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Sheet 6: Parameter Analysis
                param_analysis_data = []
                for param_name in self.param_ranges.keys():
                    # Get all values for this parameter
                    current_values = [h['params'][param_name] for h in self.metrics_history]
                    best_values = [bp[param_name] for bp in self.best_params_history]
                    
                    param_analysis_data.append({
                        'Parameter': param_name,
                        'Current_Min': min(current_values),
                        'Current_Max': max(current_values),
                        'Current_Mean': np.mean(current_values),
                        'Current_Std': np.std(current_values),
                        'Best_Min': min(best_values),
                        'Best_Max': max(best_values),
                        'Best_Mean': np.mean(best_values),
                        'Best_Std': np.std(best_values),
                        'Best_Final': self.best_params[param_name] if self.best_params else None
                    })
                
                param_analysis_df = pd.DataFrame(param_analysis_data)
                param_analysis_df.to_excel(writer, sheet_name='Parameter_Analysis', index=False)
                
            print(f"✓ Detailed results exported to {filename}")
            print("  📊 Sheets included:")
            print("    • Iteration_History - All iterations with current parameters")
            print("    • Best_Parameters_History - Best parameters for each iteration")
            print("    • Combined_Parameters - Current vs Best parameters comparison")
            print("    • Final_Best_Parameters - Final best parameters")
            print("    • Summary_Statistics - Statistical summary of all metrics")
            print("    • Parameter_Analysis - Detailed parameter analysis")
            
        except Exception as e:
            print(f"❌ Error exporting to Excel: {e}")
            print("Make sure you have openpyxl installed: pip install openpyxl")

def main():
    """Main function for testing - adapted from po_rf_kg.py"""
    print("XGBoost Randomized Search - Regression")
    print("=" * 50)
    
    try:
        # Read flood data from Kaggle
        print("Reading flood data from Kaggle...")
        # For local testing, try to read from local file first
        try:
            df = pd.read_csv('flood_training.csv', sep=';', na_values='<Null>')
            print("✓ Successfully loaded flood training data from local file")
        except FileNotFoundError:
            df = pd.read_csv('/kaggle/input/flood-trainning/flood_training.csv', sep=';', na_values='<Null>')
            print("✓ Successfully loaded flood training data from Kaggle")
        
        # Expected flood data columns
        feature_columns = [
            'Aspect', 'Curvature', 'DEM', 'Density_river', 'Density_road',
            'Distance_river', 'Distance_road', 'Flow_direction', 'NDBI',
            'NDVI', 'NDWI', 'Slope', 'TWI_final', 'Rainfall'
        ]
        label_column = 'Nom'
        
        # Convert target column (Yes/No to 1/0)
        df[label_column] = (df[label_column] == 'Yes').astype(float)
        
        # Replace commas with dots and convert to float
        for col in feature_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
        
        # Handle missing values (from po_rf_kg.py)
        print("Checking missing values...")
        missing_summary = df[feature_columns].isnull().sum()
        print(missing_summary[missing_summary > 0])
        
        # Fill missing values with mean
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
        
        # Prepare data
        print(f"Features: {feature_columns}")
        print(f"Target: {label_column}")
        
        X = df[feature_columns].values
        y = df[label_column].values
        
        print(f"Data shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target range: [{np.min(y):.4f}, {np.max(y):.4f}]")
        print(f"Target mean: {np.mean(y):.4f}, std: {np.std(y):.4f}")
        
        # Check for any remaining missing values
        if np.isnan(X).any():
            print("⚠ Handling remaining missing values...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Initialize and run randomized search
        print("\n" + "=" * 50)
        print("Starting XGBoost Randomized Search...")
        searcher = XGBRandomizedSearch(X, y, n_iterations=200)
        
        import time
        start_time = time.time()
        best_params, best_metrics = searcher.search()
        end_time = time.time()
        
        print(f"\nSearch completed in {end_time - start_time:.2f} seconds")
        
        if best_params is not None and best_metrics is not None:
            # Display final results
            print("\n" + "=" * 50)
            print("FINAL BEST RESULTS:")
            print(f"R²: {best_metrics['r2']:.4f}")
            print(f"Adjusted R²: {best_metrics['adjusted_r2']:.4f}")
            print(f"RMSE: {best_metrics['rmse']:.4f}")
            print(f"RMSE (CV): {best_metrics['rmse_cv']:.4f}")
            print(f"MAE: {best_metrics['mae']:.4f}")
            
            # Plot optimization progress
            print("\nGenerating optimization plots...")
            searcher.plot_optimization_progress()
            
            # Save results to CSV and Excel
            print("\nSaving results...")
            
            # Save best parameters
            params_df = pd.DataFrame([best_params])
            params_df.to_csv('rso_xgb_best_params.csv', index=False)
            params_df.to_excel('rso_xgb_best_params.xlsx', index=False)
            
            # Save final metrics
            metrics_df = pd.DataFrame([best_metrics])
            metrics_df.to_csv('rso_xgb_final_metrics.csv', index=False)
            metrics_df.to_excel('rso_xgb_final_metrics.xlsx', index=False)
            
            # Save optimization history to Excel
            history_data = []
            for i, hist in enumerate(searcher.metrics_history):
                row = {
                    'iteration': i + 1,
                    'rmse': hist['metrics']['rmse'],
                    'r2': hist['metrics']['r2'],
                    'mae': hist['metrics']['mae'],
                    'adjusted_r2': hist['metrics']['adjusted_r2'],
                    'rmse_cv': hist['metrics']['rmse_cv']
                }
                # Add current iteration parameters
                row.update({f'current_{k}': v for k, v in hist['params'].items()})
                
                # Add best parameters for this iteration
                if i < len(searcher.best_params_history):
                    row.update({f'best_{k}': v for k, v in searcher.best_params_history[i].items()})
                
                history_data.append(row)
            
            history_df = pd.DataFrame(history_data)
            # Save to both CSV and Excel
            history_df.to_csv('rso_xgb_optimization_history.csv', index=False)
            history_df.to_excel('rso_xgb_optimization_history.xlsx', index=False)
            
            # Save best parameters history separately
            best_params_history_data = []
            for i, best_params in enumerate(searcher.best_params_history):
                row = {'iteration': i + 1}
                row.update({f'best_{k}': v for k, v in best_params.items()})
                if i < len(searcher.metrics_history):
                    row['rmse'] = searcher.metrics_history[i]['metrics']['rmse']
                    row['r2'] = searcher.metrics_history[i]['metrics']['r2']
                    row['mae'] = searcher.metrics_history[i]['metrics']['mae']
                best_params_history_data.append(row)
            
            best_params_history_df = pd.DataFrame(best_params_history_data)
            best_params_history_df.to_csv('rso_xgb_best_params_history.csv', index=False)
            best_params_history_df.to_excel('rso_xgb_best_params_history.xlsx', index=False)
            
            # Export detailed results to Excel
            searcher.export_detailed_results_to_excel('rso_xgb_detailed_results.xlsx')
            
            print("\n" + "=" * 60)
            print("ALL RESULTS HAVE BEEN SAVED")
            print("=" * 60)
            print("• rso_xgb_best_params.csv/.xlsx - Final best parameters")
            print("• rso_xgb_final_metrics.csv/.xlsx - Final metrics")
            print("• rso_xgb_optimization_history.csv/.xlsx - Complete optimization history (current + best params)")
            print("• rso_xgb_best_params_history.csv/.xlsx - Best parameters evolution history")
            print("• rso_xgb_detailed_results.xlsx - Comprehensive Excel report with 6 sheets")
            print("• Multiple optimization plots displayed showing parameter evolution")
            
        else:
            print("❌ Search failed to find valid parameters.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()