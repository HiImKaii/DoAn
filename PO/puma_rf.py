import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, Optional
from po import puma

class PumaRF:
    def __init__(self, 
                 task: str = 'classification',
                 n_sol: int = 30,
                 max_iter: int = 100,
                 cv: int = 5,
                 scoring: Optional[str] = None):
        """
        Initialize PumaRF optimizer
        
        Parameters:
        -----------
        task : str
            'classification' or 'regression'
        n_sol : int
            Number of solutions (population size) for Puma Optimizer
        max_iter : int
            Maximum number of iterations
        cv : int
            Number of cross-validation folds
        scoring : str, optional
            Scoring metric for cross validation
            For classification: 'accuracy' (default), 'f1', 'precision', 'recall', etc.
            For regression: 'neg_mean_squared_error' (default), 'r2', etc.
        """
        self.task = task.lower()
        self.n_sol = n_sol
        self.max_iter = max_iter
        self.cv = cv
        
        if scoring is None:
            self.scoring = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        else:
            self.scoring = scoring
            
        # Define parameter bounds
        self.param_bounds = {
            'n_estimators': (10, 200),           # Number of trees
            'max_depth': (3, 20),                # Maximum depth of trees
            'min_samples_split': (2, 20),        # Minimum samples required to split
            'min_samples_leaf': (1, 10),         # Minimum samples required at leaf node
            'max_features_ratio': (0.1, 1.0)     # Ratio of features to consider for best split
        }
        
        self.dim = len(self.param_bounds)
        self.lb = np.array([bound[0] for bound in self.param_bounds.values()])
        self.ub = np.array([bound[1] for bound in self.param_bounds.values()])
        
        self.best_params_ = None
        self.best_score_ = None
        self.convergence_curve_ = None
        
    def _decode_solution(self, solution: np.ndarray) -> dict:
        """Convert continuous solution to discrete parameters"""
        params = {
            'n_estimators': int(solution[0]),
            'max_depth': int(solution[1]),
            'min_samples_split': int(solution[2]),
            'min_samples_leaf': int(solution[3]),
            'max_features': max(0.1, min(1.0, solution[4]))  # Ensure between 0.1 and 1.0
        }
        return params
    
    def _objective_function(self, solution: np.ndarray, X, y) -> float:
        """Objective function to minimize"""
        params = self._decode_solution(solution)
        
        if self.task == 'classification':
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                n_jobs=-1,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                n_jobs=-1,
                random_state=42
            )
            
        scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1)
        
        # Convert to minimization problem
        if 'neg_' in self.scoring:
            return -np.mean(scores)  # Already negative, just take mean
        else:
            return 1 - np.mean(scores)  # Convert to minimization
    
    def fit(self, X, y) -> 'PumaRF':
        """
        Find the best parameters for Random Forest using Puma Optimizer
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define the wrapper for objective function
        def objective(solution):
            return self._objective_function(solution, X_scaled, y)
        
        # Run Puma Optimizer
        best_solution, best_score, convergence = puma(
            n_sol=self.n_sol,
            max_iter=self.max_iter,
            lb=self.lb,
            ub=self.ub,
            dim=self.dim,
            cost_function=objective
        )
        
        # Store results
        self.best_params_ = self._decode_solution(best_solution)
        self.best_score_ = 1 - best_score if 'neg_' not in self.scoring else -best_score
        self.convergence_curve_ = convergence
        
        return self
    
    def get_best_model(self) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """
        Return the best Random Forest model with optimized parameters
        
        Returns:
        --------
        model : RandomForestClassifier or RandomForestRegressor
            The best model with optimized parameters
        """
        if self.best_params_ is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
            
        if self.task == 'classification':
            return RandomForestClassifier(**self.best_params_, random_state=42)
        else:
            return RandomForestRegressor(**self.best_params_, random_state=42)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                             n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                       random_state=42)
    
    # Initialize and run optimizer
    optimizer = PumaRF(task='classification', n_sol=20, max_iter=50, 
                      scoring='accuracy')
    optimizer.fit(X_train, y_train)
    
    # Get best model and evaluate
    best_model = optimizer.get_best_model()
    best_model.fit(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    print("Best parameters:", optimizer.best_params_)
    print("Best cross-validation score:", optimizer.best_score_)
    print("Test set score:", test_score) 