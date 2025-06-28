import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matlab.engine

class PSOMLOptimizer:
    def __init__(self, model_type='rf', task='classification', n_particles=30, max_iter=100, cv=5):
        """
        Initialize PSO ML Optimizer
        
        Parameters:
        -----------
        model_type : str
            'rf' for Random Forest
            'xgb' for XGBoost
            'cnn' for Convolutional Neural Network
        task : str
            'classification' or 'regression'
        n_particles : int
            Number of particles
        max_iter : int
            Maximum number of iterations
        cv : int
            Number of cross-validation folds
        """
        self.model_type = model_type.lower()
        self.task = task.lower()
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.cv = cv
        
        # Define parameter bounds for each model
        if self.model_type == 'rf':
            self.param_bounds = {
                'n_estimators': (10, 200),
                'max_depth': (3, 20),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10)
            }
        elif self.model_type == 'xgb':
            self.param_bounds = {
                'n_estimators': (50, 500),
                'max_depth': (3, 15),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0)
            }
        else:  # CNN
            self.param_bounds = {
                'filters1': (16, 64),
                'filters2': (32, 128),
                'dense_units': (64, 512),
                'dropout_rate': (0.1, 0.5),
                'learning_rate': (0.0001, 0.01)
            }
            
        self.dim = len(self.param_bounds)
        self.lb = np.array([bound[0] for bound in self.param_bounds.values()])
        self.ub = np.array([bound[1] for bound in self.param_bounds.values()])
        
        # Initialize MATLAB engine
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath('src/pso')
        
        self.best_params_ = None
        self.best_score_ = None
        
    def _decode_solution(self, solution):
        """Convert continuous solution to proper parameter format"""
        if self.model_type == 'rf':
            return {
                'n_estimators': int(solution[0]),
                'max_depth': int(solution[1]),
                'min_samples_split': int(solution[2]),
                'min_samples_leaf': int(solution[3])
            }
        elif self.model_type == 'xgb':
            return {
                'n_estimators': int(solution[0]),
                'max_depth': int(solution[1]),
                'learning_rate': solution[2],
                'subsample': solution[3],
                'colsample_bytree': solution[4]
            }
        else:  # CNN
            return {
                'filters1': int(solution[0]),
                'filters2': int(solution[1]),
                'dense_units': int(solution[2]),
                'dropout_rate': solution[3],
                'learning_rate': solution[4]
            }
            
    def _create_model(self, params):
        """Create model with given parameters"""
        if self.model_type == 'rf':
            if self.task == 'classification':
                return RandomForestClassifier(**params, random_state=42)
            else:
                return RandomForestRegressor(**params, random_state=42)
                
        elif self.model_type == 'xgb':
            if self.task == 'classification':
                return xgb.XGBClassifier(**params, random_state=42)
            else:
                return xgb.XGBRegressor(**params, random_state=42)
                
        else:  # CNN
            def create_cnn():
                model = Sequential([
                    Conv2D(params['filters1'], (3, 3), activation='relu', padding='same',
                          input_shape=self.input_shape),
                    MaxPooling2D(),
                    Conv2D(params['filters2'], (3, 3), activation='relu', padding='same'),
                    MaxPooling2D(),
                    Flatten(),
                    Dense(params['dense_units'], activation='relu'),
                    Dropout(params['dropout_rate']),
                    Dense(self.num_classes, activation='softmax')
                ])
                model.compile(optimizer='adam', loss='categorical_crossentropy',
                            metrics=['accuracy'])
                return model
                
            return KerasClassifier(model=create_cnn, epochs=30, batch_size=32, verbose=0)
            
    def _objective_function(self, solution, X, y):
        """Objective function to minimize"""
        params = self._decode_solution(solution)
        model = self._create_model(params)
        
        if self.task == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'neg_mean_squared_error'
            
        scores = cross_val_score(model, X, y, cv=self.cv, scoring=scoring)
        
        if scoring == 'neg_mean_squared_error':
            return -np.mean(scores)  # Convert to minimization
        else:
            return 1 - np.mean(scores)  # Convert to minimization
            
    def fit(self, X, y):
        """
        Find the best parameters using PSO algorithm
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Target values
        """
        if self.model_type == 'cnn':
            if len(X.shape) != 4:
                raise ValueError("For CNN, input X must be 4D: (samples, height, width, channels)")
            self.input_shape = X.shape[1:]
            self.num_classes = len(np.unique(y))
            
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        
        # Convert bounds to MATLAB format
        lb_m = matlab.double(self.lb.tolist())
        ub_m = matlab.double(self.ub.tolist())
        
        # Define objective function wrapper for MATLAB
        def obj_fun_wrapper(x):
            return float(self._objective_function(np.array(x), X_scaled, y))
            
        # Run PSO algorithm
        best_position, best_fitness = self.eng.pso(
            obj_fun_wrapper,
            self.dim,
            lb_m,
            ub_m,
            self.max_iter,
            self.n_particles,
            nargout=2
        )
        
        # Store results
        self.best_params_ = self._decode_solution(np.array(best_position))
        self.best_score_ = 1 - best_fitness if self.task == 'classification' else -best_fitness
        
        return self
        
    def get_best_model(self):
        """Return the best model with optimized parameters"""
        if self.best_params_ is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        return self._create_model(self.best_params_)

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
    optimizer = PSOMLOptimizer(
        model_type='rf',
        task='classification',
        n_particles=20,
        max_iter=50,
        cv=3
    )
    
    # Find best parameters
    optimizer.fit(X_train, y_train)
    
    # Get best model and evaluate
    best_model = optimizer.get_best_model()
    best_model.fit(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    print("Best parameters:", optimizer.best_params_)
    print("Best cross-validation score:", optimizer.best_score_)
    print("Test set score:", test_score) 