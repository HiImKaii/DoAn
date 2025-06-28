import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Union
import pandas as pd
import time
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error, r2_score

class GridSearchOptimizer:
    def __init__(self,
                 param_grid: Dict[str, Any],
                 task: str = 'classification',
                 cv: int = 5,
                 scoring: Optional[str] = None,
                 n_jobs: int = -1,
                 verbose: int = 1):
        """
        Base class for Grid Search Optimization
        
        Parameters:
        -----------
        param_grid : dict
            Dictionary with parameters names (string) as keys and lists of parameter settings to try
        task : str
            'classification' or 'regression'
        cv : int
            Number of cross-validation folds
        scoring : str, optional
            Scoring metric for cross validation
            For classification: 'accuracy' (default), 'f1', 'precision', 'recall', etc.
            For regression: 'neg_mean_squared_error' (default), 'r2', etc.
        n_jobs : int
            Number of jobs to run in parallel (-1 means using all processors)
        verbose : int
            Controls the verbosity: the higher, the more messages
        """
        self.param_grid = param_grid
        self.task = task.lower()
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Set default scoring if not provided
        if scoring is None:
            self.scoring = 'accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        else:
            self.scoring = scoring
            
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.best_model_ = None
        self.search_time_ = None
        
    def _get_scorer(self):
        """Get the scorer function based on the scoring metric"""
        if self.task == 'classification':
            if self.scoring == 'accuracy':
                return make_scorer(accuracy_score)
        else:
            if self.scoring == 'neg_mean_squared_error':
                return make_scorer(mean_squared_error, greater_is_better=False)
            elif self.scoring == 'r2':
                return make_scorer(r2_score)
        return self.scoring
    
    def fit(self, X, y, model):
        """
        Perform grid search to find the best parameters
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        model : estimator object
            This is assumed to implement the scikit-learn estimator interface
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create grid search object
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self._get_scorer(),
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        # Perform grid search
        start_time = time.time()
        grid_search.fit(X_scaled, y)
        self.search_time_ = time.time() - start_time
        
        # Store results
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.cv_results_ = pd.DataFrame(grid_search.cv_results_)
        self.best_model_ = grid_search.best_estimator_
        
        return self
    
    def get_best_model(self):
        """
        Return the best model with optimized parameters
        
        Returns:
        --------
        model : estimator object
            The best model with optimized parameters
        """
        if self.best_model_ is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        return self.best_model_
    
    def print_results(self):
        """Print the optimization results"""
        if self.best_params_ is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
            
        print("\nGrid Search Results:")
        print("-" * 50)
        print(f"Best parameters: {self.best_params_}")
        print(f"Best score: {self.best_score_:.4f}")
        print(f"Search time: {self.search_time_:.2f} seconds")
        
        if self.cv_results_ is not None:
            print("\nTop 5 parameter combinations:")
            top_results = self.cv_results_.nlargest(5, 'mean_test_score')
            print(top_results[['params', 'mean_test_score', 'std_test_score']]) 