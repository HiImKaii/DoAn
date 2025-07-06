import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import time

class XGBRandomizedSearch:
    def __init__(self, X, y, n_iterations=50):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_iterations = n_iterations
        self.best_params = None
        self.best_score = -np.inf
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y
        )
        
        # Scale data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # XGBoost parameter ranges
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 300},
            'max_depth': {'type': 'int', 'min': 3, 'max': 10},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.3},
            'subsample': {'type': 'float', 'min': 0.6, 'max': 1.0},
            'colsample_bytree': {'type': 'float', 'min': 0.6, 'max': 1.0},
            'reg_alpha': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'reg_lambda': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'min_child_weight': {'type': 'int', 'min': 1, 'max': 10}
        }
    
    def create_random_params(self):
        """Create random parameter set"""
        params = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                params[param] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'float':
                params[param] = random.uniform(range_info['min'], range_info['max'])
        return params
    
    def evaluate_params(self, params):
        """Evaluate parameter set using cross-validation"""
        try:
            model = xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                min_child_weight=params['min_child_weight'],
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=3, scoring='f1')
            
            return float(np.mean(cv_scores))
            
        except Exception as e:
            print(f"Error evaluating params: {str(e)}")
            return -np.inf
    
    def search(self):
        """Main randomized search algorithm"""
        print("Starting XGBoost Randomized Search...")
        print(f"Data: {len(self.X)} points, {self.X.shape[1]} features")
        print(f"Number of iterations: {self.n_iterations}")
        
        # Class distribution
        unique_labels = np.unique(self.y)
        label_counts = np.bincount(self.y.astype(int))
        print("Class distribution:")
        for label, count in zip(unique_labels, label_counts):
            print(f"  Class {label}: {count}")
        print("-" * 50)
        
        # Random search loop
        for iteration in range(self.n_iterations):
            print(f"\nIteration {iteration + 1}/{self.n_iterations}")
            
            # Generate random parameters
            params = self.create_random_params()
            
            # Evaluate parameters
            score = self.evaluate_params(params)
            
            # Update best if improved
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                print("*** NEW BEST FOUND! ***")
            
            # Print results
            print(f"Current score: {score:.4f}")
            print(f"Best score so far: {self.best_score:.4f}")
            print("Current parameters:")
            for param, value in params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.4f}")
                else:
                    print(f"  {param}: {value}")
        
        print("\n" + "=" * 50)
        print("Randomized Search completed!")
        if self.best_params is not None:
            print(f"\nBest score: {self.best_score:.4f}")
            print("Best parameters:")
            for param, value in self.best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.4f}")
                else:
                    print(f"  {param}: {value}")
        
        return self.best_params, self.best_score
    
    def evaluate_final_model(self):
        """Evaluate final model on test set"""
        if self.best_params is None:
            print("No optimized model available!")
            return None
        
        # Train model with best parameters
        best_model = xgb.XGBClassifier(
            n_estimators=self.best_params['n_estimators'],
            max_depth=self.best_params['max_depth'],
            learning_rate=self.best_params['learning_rate'],
            subsample=self.best_params['subsample'],
            colsample_bytree=self.best_params['colsample_bytree'],
            reg_alpha=self.best_params['reg_alpha'],
            reg_lambda=self.best_params['reg_lambda'],
            min_child_weight=self.best_params['min_child_weight'],
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        best_model.fit(self.X_train_scaled, self.y_train)
        
        # Predict on test set
        y_pred = best_model.predict(self.X_test_scaled)
        y_prob = best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        test_f1 = f1_score(self.y_test, y_pred)
        test_auc = roc_auc_score(self.y_test, y_prob)
        test_acc = accuracy_score(self.y_test, y_pred)
        
        print("\nTest Set Results:")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"AUC-ROC: {test_auc:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        
        return {
            'model': best_model,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': self.best_params
        }

def main():
    """Main function"""
    print("Reading data from Excel file...")
    
    # Change this path to your data file
    file_path = "C:/Users/Admin/Downloads/prj/src/flood_data.xlsx"
    
    try:
        df = pd.read_excel(file_path)
        print(f"Read {len(df)} rows of data")
        
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
        
        # Initialize and run randomized search
        searcher = XGBRandomizedSearch(X, y, n_iterations=30)
        
        start_time = time.time()
        best_params, best_score = searcher.search()
        end_time = time.time()
        
        print(f"\nSearch time: {end_time - start_time:.2f} seconds")
        
        if best_params is not None:
            # Evaluate final model
            print("\nEvaluating final model on test set:")
            final_results = searcher.evaluate_final_model()
            
            if final_results:
                print(f"\nFinal Test Results:")
                print(f"F1-Score: {final_results['test_f1']:.4f}")
                print(f"AUC: {final_results['test_auc']:.4f}")
                print(f"Accuracy: {final_results['test_accuracy']:.4f}")
        else:
            print("\nSearch failed to find valid parameters.")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please ensure your Excel file exists at the specified path")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()