import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import time

class SVMRandomizedSearch:
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
        
        # Scale data (rất quan trọng cho SVM)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # SVM parameter ranges
        self.param_ranges = {
            'C': {'type': 'log_uniform', 'min': 0.001, 'max': 1000},  # Regularization parameter
            'gamma': {'type': 'log_uniform', 'min': 0.0001, 'max': 10},  # Kernel coefficient
            'kernel': {'type': 'choice', 'options': ['linear', 'poly', 'rbf', 'sigmoid']},
            'degree': {'type': 'int', 'min': 2, 'max': 5},  # Degree for poly kernel
            'coef0': {'type': 'float', 'min': 0.0, 'max': 10.0},  # Independent term for poly/sigmoid
            'class_weight': {'type': 'choice', 'options': [None, 'balanced']},
            'tol': {'type': 'log_uniform', 'min': 1e-5, 'max': 1e-2}  # Tolerance for stopping criterion
        }
    
    def create_random_params(self):
        """Create random parameter set"""
        params = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                params[param] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'float':
                params[param] = random.uniform(range_info['min'], range_info['max'])
            elif range_info['type'] == 'log_uniform':
                # Log-uniform distribution for parameters like C and gamma
                log_min = np.log10(range_info['min'])
                log_max = np.log10(range_info['max'])
                params[param] = 10 ** random.uniform(log_min, log_max)
            elif range_info['type'] == 'choice':
                params[param] = random.choice(range_info['options'])
        
        # Adjust parameters based on kernel choice
        kernel = params['kernel']
        if kernel == 'linear':
            # Linear kernel doesn't use gamma, degree, coef0
            params.pop('gamma', None)
            params.pop('degree', None)
            params.pop('coef0', None)
        elif kernel == 'poly':
            # Polynomial kernel uses all parameters
            pass
        elif kernel == 'rbf':
            # RBF kernel doesn't use degree, coef0
            params.pop('degree', None)
            params.pop('coef0', None)
        elif kernel == 'sigmoid':
            # Sigmoid kernel doesn't use degree
            params.pop('degree', None)
        
        return params
    
    def evaluate_params(self, params):
        """Evaluate parameter set using cross-validation"""
        try:
            # Create SVM model with parameters
            model_params = params.copy()
            
            model = SVC(
                C=model_params.get('C', 1.0),
                gamma=model_params.get('gamma', 'scale'),
                kernel=model_params.get('kernel', 'rbf'),
                degree=model_params.get('degree', 3),
                coef0=model_params.get('coef0', 0.0),
                class_weight=model_params.get('class_weight', None),
                tol=model_params.get('tol', 1e-3),
                probability=True,  # Enable probability estimates for AUC calculation
                max_iter=10000  # Increase max iterations to ensure convergence
            )
            
            # Cross-validation với timeout protection
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=3, scoring='f1', n_jobs=1)
            
            return float(np.mean(cv_scores))
            
        except Exception as e:
            print(f"Error evaluating params: {str(e)}")
            return -np.inf
    
    def search(self):
        """Main randomized search algorithm"""
        print("Starting SVM Randomized Search...")
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
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")
        
        print("\n" + "=" * 50)
        print("Randomized Search completed!")
        if self.best_params is not None:
            print(f"\nBest score: {self.best_score:.4f}")
            print("Best parameters:")
            for param, value in self.best_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")
        
        return self.best_params, self.best_score
    
    def evaluate_final_model(self):
        """Evaluate final model on test set"""
        if self.best_params is None:
            print("No optimized model available!")
            return None
        
        # Train model with best parameters
        best_model = SVC(
            C=self.best_params.get('C', 1.0),
            gamma=self.best_params.get('gamma', 'scale'),
            kernel=self.best_params.get('kernel', 'rbf'),
            degree=self.best_params.get('degree', 3),
            coef0=self.best_params.get('coef0', 0.0),
            class_weight=self.best_params.get('class_weight', None),
            tol=self.best_params.get('tol', 1e-3),
            probability=True,
            max_iter=10000
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
        
        # Print support vectors info
        print(f"Number of support vectors: {best_model.n_support_}")
        print(f"Support vectors per class: {dict(zip(best_model.classes_, best_model.n_support_))}")
        
        return {
            'model': best_model,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'best_params': {
                'C': self.best_params['C'],
                'kernel': self.best_params['kernel'],
                'gamma': self.best_params['gamma']
            },
            'n_support_vectors': best_model.n_support_
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
        
        # Check data size for SVM efficiency
        if len(X) > 10000:
            print(f"WARNING: Large dataset ({len(X)} samples). SVM may be slow.")
            print("Consider using a subset or switching to linear kernel.")
        
        # Initialize and run randomized search
        searcher = SVMRandomizedSearch(X, y, n_iterations=30)
        
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
                
                # Additional SVM-specific information
                print(f"\nModel Information:")
                print(f"Best kernel: {final_results['best_params']['kernel']}")
                print(f"Total support vectors: {sum(final_results['n_support_vectors'])}")
        else:
            print("\nSearch failed to find valid parameters.")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please ensure your Excel file exists at the specified path")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()