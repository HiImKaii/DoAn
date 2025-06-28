try:
    import tensorflow.keras as keras
    from tensorflow.keras import layers, models
    from scikeras.wrappers import KerasClassifier, KerasRegressor

except ImportError:
    raise ImportError("Please install tensorflow: pip install tensorflow")

from grid_search_optimizer import GridSearchOptimizer
from typing import Optional, Tuple, List, Union, Any
import numpy as np

class CNNGridSearch(GridSearchOptimizer):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 num_classes: int,
                 task: str = 'classification',
                 cv: int = 5,
                 scoring: Optional[str] = None,
                 n_jobs: int = -1,
                 verbose: int = 1):
        """
        Grid Search Optimizer for CNN
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input images (height, width, channels)
        num_classes : int
            Number of classes for classification or 1 for regression
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
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Define parameter grid for CNN
        param_grid = {
            'conv_layers': [[32, 64], [32, 64, 128], [64, 128, 256]],
            'dense_layers': [[128], [256], [512], [256, 128]],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'optimizer': ['adam', 'rmsprop'],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [32, 64, 128],
            'epochs': [10, 20, 30]
        }
        
        super().__init__(
            param_grid=param_grid,
            task=task,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
    
    def _create_model(self, conv_layers: List[int], dense_layers: List[int],
                     dropout_rate: float, optimizer: str, learning_rate: float) -> models.Sequential:
        """Create CNN model with given parameters"""
        model = models.Sequential()
        
        # Add convolutional layers
        for i, filters in enumerate(conv_layers):
            if i == 0:
                model.add(layers.Conv2D(filters, (3, 3), activation='relu',
                               padding='same', input_shape=self.input_shape))
            else:
                model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))
        
        # Flatten layer
        model.add(layers.Flatten())
        
        # Add dense layers
        for units in dense_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        if self.task == 'classification':
            if self.num_classes == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
            else:
                model.add(layers.Dense(self.num_classes, activation='softmax'))
        else:
            model.add(layers.Dense(1))
        
        # Compile model
        optimizer_map = {
            'adam': keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            'rmsprop': keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
        }
        
        if self.task == 'classification':
            loss = 'binary_crossentropy' if self.num_classes == 2 else 'categorical_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
            
        model.compile(optimizer=optimizer_map[optimizer],
                     loss=loss,
                     metrics=metrics)
        
        return model
    
    def _create_default_model(self) -> models.Sequential:
        """Create a model with default parameters"""
        return self._create_model(
            conv_layers=[32, 64],
            dense_layers=[128],
            dropout_rate=0.3,
            optimizer='adam',
            learning_rate=0.001
        )
    
    def fit(self, X, y):
        """
        Find the best parameters for CNN
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, height, width, channels)
            Training images
        y : array-like of shape (n_samples,) or (n_samples, num_classes)
            Target values
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        # Ensure X is numpy array
        X = np.asarray(X)
        
        # Normalize input images
        X = X.astype('float32') / 255.0
        
        # Convert labels for classification
        if self.task == 'classification':
            if self.num_classes > 2:
                y = keras.utils.to_categorical(y, self.num_classes)
        
        # Create scikit-learn compatible model
        if self.task == 'classification':
            model = KerasClassifier(
                model=self._create_default_model,
                verbose=0
            )
        else:
            model = KerasRegressor(
                model=self._create_default_model,
                verbose=0
            )
            
        return super().fit(X, y, model)

# Example usage
if __name__ == "__main__":
    # Generate sample image data
    num_samples = 1000
    img_height, img_width = 28, 28
    num_channels = 1
    num_classes = 10
    
    # Create random image data and labels
    X = np.random.rand(num_samples, img_height, img_width, num_channels)
    y = np.random.randint(0, num_classes, num_samples)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=0.2,
                                                       random_state=42)
    
    # Initialize and run optimizer
    cnn_optimizer = CNNGridSearch(
        input_shape=(img_height, img_width, num_channels),
        num_classes=num_classes,
        task='classification',
        cv=3,  # Using 3 folds for faster execution
        scoring='accuracy',
        verbose=2
    )
    
    # Find best parameters
    cnn_optimizer.fit(X_train, y_train)
    
    # Get best model and evaluate
    best_model = cnn_optimizer.get_best_model()
    test_score = best_model.score(X_test, y_test)
    
    # Print results
    cnn_optimizer.print_results()
    print(f"\nTest set score: {test_score:.4f}") 