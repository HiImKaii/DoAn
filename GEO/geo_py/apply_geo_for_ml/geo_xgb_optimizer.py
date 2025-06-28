import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import sys
sys.path.append('..')
from geo import GEO

class GEO_XGB_Optimizer:
    def __init__(self, X_train, y_train, cv=5):
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        
        # Define parameter bounds
        self.param_bounds = {
            'max_depth': (3, 15),           # Maximum tree depth
            'learning_rate': (0.01, 0.3),   # Learning rate
            'n_estimators': (50, 500),      # Number of trees
            'subsample': (0.5, 1.0),        # Subsample ratio
            'colsample_bytree': (0.5, 1.0), # Column sample ratio
            'min_child_weight': (1, 7)      # Minimum sum of instance weight
        }
        
    def objective_function(self, x):
        """
        Objective function to minimize (negative cross-validation score)
        x contains: [max_depth, learning_rate, n_estimators, subsample, colsample_bytree, min_child_weight]
        """
        # Convert parameters to appropriate types
        params = {
            'max_depth': int(x[0]),
            'learning_rate': x[1],
            'n_estimators': int(x[2]),
            'subsample': x[3],
            'colsample_bytree': x[4],
            'min_child_weight': int(x[5])
        }
        
        # Create and evaluate XGBoost model
        xgb = XGBClassifier(
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            min_child_weight=params['min_child_weight'],
            random_state=42
        )
        
        try:
            # Use cross-validation to evaluate model
            scores = cross_val_score(xgb, self.X_train, self.y_train, cv=self.cv)
            return -np.mean(scores)  # Return negative score since GEO minimizes
        except Exception as e:
            # Return a large value if there's an error
            print(f"Error in evaluation: {e}")
            return 1000
    
    def optimize(self, population_size=30, max_iterations=50):
        """
        Run GEO optimization to find best XGBoost parameters
        """
        # Setup optimization problem
        nvars = len(self.param_bounds)  # Number of parameters to optimize
        
        # Define bounds
        lb = np.array([bounds[0] for bounds in self.param_bounds.values()])
        ub = np.array([bounds[1] for bounds in self.param_bounds.values()])
        
        # Define GEO options
        options = {
            'PopulationSize': population_size,
            'MaxIterations': max_iterations,
            'AttackPropensity': [0.5, 2],
            'CruisePropensity': [1, 0.5]
        }
        
        # Create wrapper for objective function
        def obj_wrapper(x):
            return np.array([self.objective_function(xi) for xi in x])
        
        # Run optimization
        geo = GEO(obj_wrapper, nvars, lb, ub, options)
        best_params, best_score, convergence_curve = geo.optimize()
        
        # Convert parameters to appropriate types
        optimized_params = {
            'max_depth': int(best_params[0]),
            'learning_rate': best_params[1],
            'n_estimators': int(best_params[2]),
            'subsample': best_params[3],
            'colsample_bytree': best_params[4],
            'min_child_weight': int(best_params[5])
        }
        
        return optimized_params, -best_score, convergence_curve

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate sample dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                             n_redundant=5, random_state=42)
    
    # Create and run optimizer
    optimizer = GEO_XGB_Optimizer(X, y)
    best_params, best_score, convergence = optimizer.optimize()
    
    print("Best Parameters:", best_params)
    print("Best CV Score:", best_score) 