import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sys
sys.path.append('..')
from geo import GEO

class GEO_RF_Optimizer:
    def __init__(self, X_train, y_train, cv=5):
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        
        # Define parameter bounds
        self.param_bounds = {
            'n_estimators': (10, 200),      # Number of trees
            'max_depth': (3, 20),           # Maximum depth of trees
            'min_samples_split': (2, 20),   # Minimum samples required to split
            'min_samples_leaf': (1, 10)     # Minimum samples required at leaf node
        }
        
    def objective_function(self, x):
        """
        Objective function to minimize (negative cross-validation score)
        x contains: [n_estimators, max_depth, min_samples_split, min_samples_leaf]
        """
        # Convert parameters to appropriate types
        params = {
            'n_estimators': int(x[0]),
            'max_depth': int(x[1]),
            'min_samples_split': int(x[2]),
            'min_samples_leaf': int(x[3])
        }
        
        # Create and evaluate Random Forest model
        rf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=42
        )
        
        # Use cross-validation to evaluate model
        scores = cross_val_score(rf, self.X_train, self.y_train, cv=self.cv)
        
        # Return negative mean score (since GEO minimizes)
        return -np.mean(scores)
    
    def optimize(self, population_size=30, max_iterations=50):
        """
        Run GEO optimization to find best Random Forest parameters
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
            'n_estimators': int(best_params[0]),
            'max_depth': int(best_params[1]),
            'min_samples_split': int(best_params[2]),
            'min_samples_leaf': int(best_params[3])
        }
        
        return optimized_params, -best_score, convergence_curve

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate sample dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                             n_redundant=5, random_state=42)
    
    # Create and run optimizer
    optimizer = GEO_RF_Optimizer(X, y)
    best_params, best_score, convergence = optimizer.optimize()
    
    print("Best Parameters:", best_params)
    print("Best CV Score:", best_score) 