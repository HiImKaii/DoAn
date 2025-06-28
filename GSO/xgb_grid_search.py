import xgboost as xgb
from grid_search_optimizer import GridSearchOptimizer
from typing import Optional

class XGBGridSearch(GridSearchOptimizer):
    def __init__(self,
                 task: str = 'classification',
                 cv: int = 5,
                 scoring: Optional[str] = None,
                 n_jobs: int = -1,
                 verbose: int = 1):
        """
        Grid Search Optimizer for XGBoost
        
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
        # Define parameter grid for XGBoost
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0.1, 1.0, 5.0]
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
        Find the best parameters for XGBoost
        
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
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42
            )
            
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
    xgb_optimizer = XGBGridSearch(task='classification',
                                cv=3,  # Using 3 folds for faster execution
                                scoring='accuracy',
                                verbose=2)
    
    # Find best parameters
    xgb_optimizer.fit(X_train, y_train)
    
    # Get best model and evaluate
    best_model = xgb_optimizer.get_best_model()
    test_score = best_model.score(X_test, y_test)
    
    # Print results
    xgb_optimizer.print_results()
    print(f"\nTest set score: {test_score:.4f}") 