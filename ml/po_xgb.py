import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import random
import warnings
warnings.filterwarnings('ignore')

class PUMAOptimizer:
    def __init__(self, X, y, population_size=30, generations=500):
        self.X = np.array(X)
        self.y = np.array(y)
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_score = -np.inf
        
        # Split and scale data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # XGBoost parameter ranges
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 500},
            'max_depth': {'type': 'int', 'min': 3, 'max': 15},
            'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.3},
            'subsample': {'type': 'float', 'min': 0.5, 'max': 1.0},
            'colsample_bytree': {'type': 'float', 'min': 0.5, 'max': 1.0},
            'min_child_weight': {'type': 'int', 'min': 1, 'max': 7},
            'gamma': {'type': 'float', 'min': 0, 'max': 5}
        }
        
    def create_individual(self):
        """Create a random individual (parameter set)"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                individual[param] = random.randint(range_info['min'], range_info['max'])
            else:
                individual[param] = random.uniform(range_info['min'], range_info['max'])
        return individual

    def evaluate_individual(self, individual):
        """Evaluate fitness of an individual using cross-validation"""
        try:
            model = xgb.XGBClassifier(
                n_estimators=individual['n_estimators'],
                max_depth=individual['max_depth'],
                learning_rate=individual['learning_rate'],
                subsample=individual['subsample'],
                colsample_bytree=individual['colsample_bytree'],
                min_child_weight=individual['min_child_weight'],
                gamma=individual['gamma'],
                objective='binary:logistic',
                use_label_encoder=False,
                random_state=42
            )
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=3, scoring='f1')
            return float(np.mean(cv_scores))
        except:
            return -np.inf

    def exploration_phase(self, population, fitness_values):
        """PUMA Exploration Phase"""
        new_population = []
        new_fitness = []
        
        for i in range(self.population_size):
            current = population[i]
            
            # Select 6 different solutions randomly
            available_indices = list(range(self.population_size))
            available_indices.remove(i)
            selected_indices = random.sample(available_indices, 6)
            a, b, c, d, e, f = [population[idx] for idx in selected_indices]
            
            # Create new solution
            new_individual = {}
            for param, range_info in self.param_ranges.items():
                # Exploration formula
                if random.random() < 0.5:
                    if range_info['type'] == 'int':
                        new_individual[param] = random.randint(range_info['min'], range_info['max'])
                    else:
                        new_individual[param] = random.uniform(range_info['min'], range_info['max'])
                else:
                    G = 2 * random.random() - 1
                    term1 = a[param] + G * (a[param] - b[param])
                    term2 = G * (((a[param] - b[param]) - (c[param] - d[param])) + 
                                ((c[param] - d[param]) - (e[param] - f[param])))
                    new_val = term1 + term2
                    
                    if range_info['type'] == 'int':
                        new_val = int(round(np.clip(new_val, range_info['min'], range_info['max'])))
                    else:
                        new_val = np.clip(new_val, range_info['min'], range_info['max'])
                    new_individual[param] = new_val
            
            # Evaluate and update
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val > fitness_values[i]:
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
        
        return new_population, new_fitness

    def exploitation_phase(self, population, fitness_values):
        """PUMA Exploitation Phase"""
        new_population = []
        new_fitness = []
        
        # Get best solution
        best_idx = np.argmax(fitness_values)
        best_solution = population[best_idx]
        
        for i in range(self.population_size):
            current = population[i]
            new_individual = {}
            
            for param, range_info in self.param_ranges.items():
                if random.random() < 0.5:
                    # Move towards best solution
                    beta = 2 * random.random()
                    random_sol = random.choice(population)
                    new_val = best_solution[param] + beta * (random_sol[param] - current[param])
                else:
                    # Alternative strategy
                    mean_val = np.mean([ind[param] for ind in population])
                    random_val = random.choice(population)[param]
                    new_val = (mean_val * random_val - current[param]) / (1 + 2 * random.random())
                
                if range_info['type'] == 'int':
                    new_val = int(round(np.clip(new_val, range_info['min'], range_info['max'])))
                else:
                    new_val = np.clip(new_val, range_info['min'], range_info['max'])
                new_individual[param] = new_val
            
            # Evaluate and update
            new_fitness_val = self.evaluate_individual(new_individual)
            if new_fitness_val > fitness_values[i]:
                new_population.append(new_individual)
                new_fitness.append(new_fitness_val)
            else:
                new_population.append(current)
                new_fitness.append(fitness_values[i])
        
        return new_population, new_fitness

    def optimize(self):
        """Main PUMA optimization algorithm"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        fitness_values = [self.evaluate_individual(ind) for ind in population]
        
        # Find initial best
        best_idx = np.argmax(fitness_values)
        self.best_individual = population[best_idx].copy()
        self.best_score = fitness_values[best_idx]
        
        print(f"Initial best score: {self.best_score:.4f}")
        
        # Main optimization loop
        for iteration in range(self.generations):
            # Alternate between exploration and exploitation
            if iteration % 2 == 0:
                population, fitness_values = self.exploration_phase(population, fitness_values)
                phase = "Exploration"
            else:
                population, fitness_values = self.exploitation_phase(population, fitness_values)
                phase = "Exploitation"
            
            # Update best solution
            current_best_idx = np.argmax(fitness_values)
            if fitness_values[current_best_idx] > self.best_score:
                self.best_score = fitness_values[current_best_idx]
                self.best_individual = population[current_best_idx].copy()
                print(f"Iteration {iteration + 1}: {phase} - New best score: {self.best_score:.4f}")
            
            # Early stopping
            if iteration > 10:
                if max(fitness_values) - min(fitness_values) < 0.001:
                    print("Early stopping: No improvement")
                    break
        
        return self.best_individual, self.best_score

def main():
    # Example usage
    X = np.random.rand(100, 10)  # Replace with your data
    y = np.random.randint(0, 2, 100)  # Replace with your labels
    
    optimizer = PUMAOptimizer(X, y, population_size=20, generations=50)
    best_params, best_score = optimizer.optimize()
    
    print("\nOptimization completed!")
    print(f"Best score: {best_score:.4f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

if __name__ == "__main__":
    main()
