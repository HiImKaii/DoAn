import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import time
import joblib

class RandomSearchOptimizer:
    def __init__(self, X, y, n_iter=10, feature_names=None):
        """
        Random Search Optimizer for Random Forest
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training input samples
        y : array-like, shape (n_samples,)
            Target values
        n_iter : int, default=10
            Number of parameter settings that are sampled
        feature_names : list, optional
            Names of the features
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_iter = n_iter
        self.best_params = None
        self.best_score = -np.inf
        self.history = []
        self.feature_names = feature_names
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Random Forest parameter distributions
        self.param_distributions = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 300},
            'max_depth': {'type': 'int', 'min': 3, 'max': 50},
            'min_samples_split': {'type': 'int', 'min': 2, 'max': 30},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 20},
            'max_features': ['sqrt', 'log2', None, 0.5, 0.7, 0.9],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }
    
    def sample_parameters(self):
        """
        Sample random parameters from the distributions
        
        Returns:
        --------
        dict : Sampled parameters
        """
        params = {}
        for param_name, param_config in self.param_distributions.items():
            if isinstance(param_config, dict):
                if param_config['type'] == 'int':
                    params[param_name] = random.randint(param_config['min'], param_config['max'])
                elif param_config['type'] == 'float':
                    params[param_name] = random.uniform(param_config['min'], param_config['max'])
            else:  # List of choices
                params[param_name] = random.choice(param_config)
        
        return params
    
    def evaluate_parameters(self, params):
        """
        Evaluate a parameter set using cross-validation
        
        Parameters:
        -----------
        params : dict
            Parameter set to evaluate
            
        Returns:
        --------
        float : Mean cross-validation score
        """
        try:
            # Create RandomForest with sampled parameters
            rf = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                bootstrap=params['bootstrap'],
                criterion=params['criterion'],
                random_state=42,
                class_weight='balanced',
                n_jobs=-1  # Use all available cores
            )
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                rf, self.X_train_scaled, self.y_train, 
                cv=3, scoring='f1', n_jobs=-1
            )
            
            return float(np.mean(cv_scores))
            
        except Exception as e:
            print(f"Error evaluating parameters: {str(e)}")
            print("Parameters:", params)
            return -np.inf
    
    def optimize(self):
        """
        Main Random Search optimization
        
        Returns:
        --------
        tuple : (best_params, best_score)
        """
        print("Starting Random Search optimization...")
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
        
        for iteration in range(self.n_iter):
            try:
                print(f"\nIteration {iteration + 1}/{self.n_iter}")
                
                # Sample random parameters
                params = self.sample_parameters()
                
                # Evaluate parameters
                score = self.evaluate_parameters(params)
                
                # Update best if better
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    
                    print("\nNew best solution found!")
                    print(f"Parameters:")
                    for param, value in params.items():
                        print(f"  {param}: {value}")
                
                print(f"\nBest score in iteration {iteration + 1}: {self.best_score:.4f}")
                print(f"Current score: {score:.4f}")
                print(f"Best parameters so far:")
                if self.best_params:
                    for param, value in self.best_params.items():
                        print(f"  {param}: {value}")
                
                # Store history
                self.history.append({
                    'iteration': iteration + 1,
                    'score': score,
                    'params': params.copy(),
                    'is_best': score == self.best_score,
                    'best_score': self.best_score
                })
                    
            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {str(e)}")
                continue
        
        print("\n" + "=" * 50)
        print("Optimization completed!")
        
        if self.best_params is not None:
            print("\nBest solution found:")
            print(f"Score: {self.best_score:.4f}")
            print("Parameters:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
        else:
            print("No valid parameters found!")
            
        return self.best_params, self.best_score
    
    def evaluate_final_model(self):
        """
        Train and evaluate the final model on test set
        
        Returns:
        --------
        dict : Final model results
        """
        if self.best_params is None:
            print("No optimized model available!")
            return None
        
        # Create and train final model
        final_rf = RandomForestClassifier(
            n_estimators=self.best_params['n_estimators'],
            max_depth=self.best_params['max_depth'],
            min_samples_split=self.best_params['min_samples_split'],
            min_samples_leaf=self.best_params['min_samples_leaf'],
            max_features=self.best_params['max_features'],
            bootstrap=self.best_params['bootstrap'],
            criterion=self.best_params['criterion'],
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        final_rf.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_pred = final_rf.predict(self.X_test_scaled)
        y_prob = final_rf.predict_proba(self.X_test_scaled)
        
        # Get probabilities for class 1
        if isinstance(y_prob, np.ndarray) and y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
        
        # Calculate metrics
        test_f1 = f1_score(self.y_test, y_pred)
        test_auc = roc_auc_score(self.y_test, y_prob)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        
        print("\nTest Set Metrics:")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"AUC-ROC: {test_auc:.4f}")
        print(f"Accuracy: {test_accuracy:.4f}")
        
        # Get feature names
        feature_names = getattr(self, 'feature_names', None)
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
        
        return {
            'model': final_rf,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_accuracy,
            'best_params': self.best_params,
            'feature_importances': dict(zip(feature_names, final_rf.feature_importances_))
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
        
        # Initialize Random Search optimizer with feature names
        optimizer = RandomSearchOptimizer(X, y, n_iter=10, feature_names=feature_columns)
        
        # Run optimization
        start_time = time.time()
        best_params, best_score = optimizer.optimize()
        end_time = time.time()
        
        print(f"\nOptimization time: {end_time - start_time:.2f} seconds")
        
        if best_params is not None:
            print("\nBest parameters:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            print(f"\nBest score: {best_score:.4f}")
            
            # Evaluate final model
            print("\nEvaluating model on test set:")
            final_results = optimizer.evaluate_final_model()
            
            if final_results:
                print(f"Test F1-Score: {final_results['test_f1']:.4f}")
                print(f"Test AUC: {final_results['test_auc']:.4f}")
                print(f"Test Accuracy: {final_results['test_accuracy']:.4f}")
                
                # Save model
                joblib.dump(final_results['model'], 'best_flood_rf_randomsearch.pkl')
                print("\nModel saved to 'best_flood_rf_randomsearch.pkl'")
        else:
            print("\nOptimization failed to find valid parameters.")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please ensure your Excel file exists at the specified path")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()