from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from grid_search_optimizer import GridSearchOptimizer
from typing import Optional

class RFGridSearch(GridSearchOptimizer):
    def __init__(self,
                 task: str = 'classification',
                 cv: int = 5,
                 scoring: Optional[str] = None,
                 n_jobs: int = -1,
                 verbose: int = 1):
        """
        Grid Search Optimizer for Random Forest
        
        Parameters:
        -----------
        task : str
            'classification' or 'regression'
        cv : int
            Number of cross-validation folds
        scoring : str, optional
            Scoring metric for cross validation
        n_jobs : int
            Number of jobs to run in parallel
        verbose : int
            Controls the verbosity
        """
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        super().__init__(
            param_grid=param_grid,
            task=task,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
    
    def fit(self, X, y):
        """
        Find the best parameters for Random Forest
        
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
        # Initialize appropriate model
        if self.task == 'classification':
            model = RandomForestClassifier(random_state=42)
        else:
            model = RandomForestRegressor(random_state=42)
            
        return super().fit(X, y, model)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20,
                             n_informative=15, n_redundant=5,
                             random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=0.2,
                                                       random_state=42)
    
    # Initialize and run optimizer
    rf_optimizer = RFGridSearch(task='classification',
                              cv=3,  # Using 3 folds for faster execution
                              scoring='accuracy',
                              verbose=2)
    
    # Find best parameters
    rf_optimizer.fit(X_train, y_train)
    
    # Get best model and evaluate
    best_model = rf_optimizer.get_best_model()
    test_score = best_model.score(X_test, y_test)
    
    # Print results
    rf_optimizer.print_results()
    print(f"\nTest set score: {test_score:.4f}") 