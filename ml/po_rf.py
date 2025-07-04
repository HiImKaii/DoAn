import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import random
import warnings
warnings.filterwarnings('ignore')

class PUMAOptimizer:
    def __init__(self, X, y, population_size=30, generations=50):
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
        
        # RF parameter ranges
        self.param_ranges = {
            'n_estimators': {'type': 'int', 'min': 50, 'max': 500},
            'max_depth': {'type': 'int', 'min': 5, 'max': 50},
            'min_samples_split': {'type': 'int', 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'min': 1, 'max': 10},
            'max_features': {'type': 'categorical', 'values': ['sqrt', 'log2', None]}
        }
    
    def create_individual(self):
        """Create a random individual (parameter set)"""
        individual = {}
        for param, range_info in self.param_ranges.items():
            if range_info['type'] == 'int':
                individual[param] = random.randint(range_info['min'], range_info['max'])
            elif range_info['type'] == 'categorical':
                individual[param] = random.choice(range_info['values'])
        return individual
    
    def evaluate_individual(self, individual):
        """Evaluate fitness of an individual using cross-validation"""
        try:
            model = RandomForestClassifier(
                n_estimators=individual['n_estimators'],
                max_depth=individual['max_depth'],
                min_samples_split=individual['min_samples_split'],
                min_samples_leaf=individual['min_samples_leaf'],
                max_features=individual['max_features'],
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
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
                if range_info['type'] == 'int':
                    if random.random() < 0.5:
                        new_individual[param] = random.randint(range_info['min'], range_info['max'])
                    else:
                        G = 2 * random.random() - 1
                        term1 = a[param] + G * (a[param] - b[param])
                        term2 = G * (((a[param] - b[param]) - (c[param] - d[param])) + 
                                   ((c[param] - d[param]) - (e[param] - f[param])))
                        new_val = int(round(np.clip(term1 + term2, range_info['min'], range_info['max'])))
                        new_individual[param] = new_val
                else:  # categorical
                    values = [ind[param] for ind in [a, b, c, d, e, f]]
                    new_individual[param] = random.choice(values)
            
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
        Q = 0.67  # Exploitation constant
        Beta = 2  # Beta constant
        new_population = []
        new_fitness = []
        
        # Get best solution
        best_idx = np.argmax(fitness_values)
        best_solution = population[best_idx]
        
        # Calculate mean position
        mbest = {}
        for param in self.param_ranges:
            if self.param_ranges[param]['type'] == 'int':
                mbest[param] = int(np.mean([p[param] for p in population]))
            else:
                mbest[param] = np.mean([p[param] for p in population])
        
        for i in range(self.population_size):
            current = population[i]
            new_individual = {}
            
            beta1 = 2 * random.random()
            beta2 = np.random.randn(len(self.param_ranges))
            
            w = np.random.randn(len(self.param_ranges))  # Eq 37
            v = np.random.randn(len(self.param_ranges))  # Eq 38
            
            # Eq 35
            F1 = np.random.randn(len(self.param_ranges)) * np.exp(2 - i * (2/self.generations))
            # Eq 36
            F2 = w * np.power(v, 2) * np.cos((2 * random.random()) * w)
            
            R_1 = 2 * random.random() - 1  # Eq 34
            
            if random.random() <= 0.5:
                # Calculate S1 and S2
                S1 = 2 * random.random() - 1 + np.random.randn(len(self.param_ranges))
                
                # Convert to arrays for vector operations
                current_array = np.array([current[param] for param in self.param_ranges])
                best_array = np.array([best_solution[param] for param in self.param_ranges])
                
                S2 = F1 * R_1 * current_array + F2 * (1 - R_1) * best_array
                VEC = S2 / S1
                
                if random.random() > Q:
                    # Eq 32 first part
                    random_sol = random.choice(population)
                    random_array = np.array([random_sol[param] for param in self.param_ranges])
                    new_pos = best_array + beta1 * (np.exp(beta2)) * (random_array - current_array)
                else:
                    # Eq 32 second part
                    new_pos = beta1 * VEC - best_array
            else:
                # Eq 33
                r1 = random.randint(0, self.population_size-1)
                r1_sol = population[r1]
                r1_array = np.array([r1_sol[param] for param in self.param_ranges])
                mbest_array = np.array([mbest[param] for param in self.param_ranges])
                current_array = np.array([current[param] for param in self.param_ranges])
                
                sign = 1 if random.random() > 0.5 else -1
                new_pos = (mbest_array * r1_array - sign * current_array) / (1 + (Beta * random.random()))
            
            # Convert back to dictionary and clip values
            for j, param in enumerate(self.param_ranges):
                if self.param_ranges[param]['type'] == 'int':
                    new_individual[param] = int(round(np.clip(new_pos[j], 
                                                            self.param_ranges[param]['min'], 
                                                            self.param_ranges[param]['max'])))
                else:
                    new_individual[param] = np.clip(new_pos[j], 
                                                  self.param_ranges[param]['min'], 
                                                  self.param_ranges[param]['max'])
            
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
        
        # Initial best
        best_idx = np.argmax(fitness_values)
        self.best_individual = population[best_idx].copy()
        self.best_score = fitness_values[best_idx]
        initial_best_score = self.best_score
        
        # Parameters for phase selection
        unselected = [1, 1]  # [Exploration, Exploitation]
        seq_time_explore = [1, 1, 1]
        seq_time_exploit = [1, 1, 1]
        seq_cost_explore = [0.0, 0.0, 0.0]
        seq_cost_exploit = [0.0, 0.0, 0.0]
        pf = [0.5, 0.5, 0.3]  # Weights for F1, F2, F3
        mega_explor = 0.99
        mega_exploit = 0.99
        f3_explore = 0
        f3_exploit = 0
        pf_f3 = []
        flag_change = 1
        
        # Unexperienced Phase (first 3 iterations)
        for iteration in range(3):
            # Exploration
            pop_explore, fit_explore = self.exploration_phase(population, fitness_values)
            cost_explore = max(fit_explore)
            
            # Exploitation
            pop_exploit, fit_exploit = self.exploitation_phase(population, fitness_values)
            cost_exploit = max(fit_exploit)
            
            # Combine and select best solutions
            population = population + pop_explore + pop_exploit
            fitness_values = fitness_values + fit_explore + fit_exploit
            indices = np.argsort(fitness_values)[::-1][:self.population_size]
            population = [population[i] for i in indices]
            fitness_values = [fitness_values[i] for i in indices]
            
            # Update best
            if fitness_values[0] > self.best_score:
                self.best_score = fitness_values[0]
                self.best_individual = population[0].copy()
        
        # Initialize sequence costs
        seq_cost_explore[0] = abs(initial_best_score - cost_explore)
        seq_cost_exploit[0] = abs(initial_best_score - cost_exploit)
        
        # Add non-zero costs to PF_F3
        for cost in seq_cost_explore + seq_cost_exploit:
            if cost != 0:
                pf_f3.append(cost)
        
        # Calculate initial scores
        f1_explore = pf[0] * (seq_cost_explore[0] / seq_time_explore[0])
        f1_exploit = pf[0] * (seq_cost_exploit[0] / seq_time_exploit[0])
        f2_explore = pf[1] * sum(seq_cost_explore) / sum(seq_time_explore)
        f2_exploit = pf[1] * sum(seq_cost_exploit) / sum(seq_time_exploit)
        score_explore = pf[0] * f1_explore + pf[1] * f2_explore
        score_exploit = pf[0] * f1_exploit + pf[1] * f2_exploit
        
        # Experienced Phase
        for iteration in range(3, self.generations):
            if score_explore > score_exploit:
                # Exploration
                population, fitness_values = self.exploration_phase(population, fitness_values)
                count_select = unselected.copy()
                unselected[1] += 1
                unselected[0] = 1
                f3_explore = pf[2]
                f3_exploit += pf[2]
                
                # Update sequence costs
                if fitness_values[0] > self.best_score:
                    seq_cost_explore = [abs(self.best_score - fitness_values[0])] + seq_cost_explore[:2]
                    if seq_cost_explore[0] != 0:
                        pf_f3.append(seq_cost_explore[0])
            else:
                # Exploitation
                population, fitness_values = self.exploitation_phase(population, fitness_values)
                count_select = unselected.copy()
                unselected[0] += 1
                unselected[1] = 1
                f3_explore += pf[2]
                f3_exploit = pf[2]
                
                # Update sequence costs
                if fitness_values[0] > self.best_score:
                    seq_cost_exploit = [abs(self.best_score - fitness_values[0])] + seq_cost_exploit[:2]
                    if seq_cost_exploit[0] != 0:
                        pf_f3.append(seq_cost_exploit[0])
            
            # Update best solution
            if fitness_values[0] > self.best_score:
                self.best_score = fitness_values[0]
                self.best_individual = population[0].copy()
            
            # Update time sequences if phase changed
            if flag_change != (1 if score_explore > score_exploit else 2):
                flag_change = 1 if score_explore > score_exploit else 2
                seq_time_explore = [count_select[0]] + seq_time_explore[:2]
                seq_time_exploit = [count_select[1]] + seq_time_exploit[:2]
            
            # Update scores
            if score_explore < score_exploit:
                mega_explor = max(mega_explor - 0.01, 0.01)
                mega_exploit = 0.99
            elif score_explore > score_exploit:
                mega_explor = 0.99
                mega_exploit = max(mega_exploit - 0.01, 0.01)
            
            lmn_explore = 1 - mega_explor
            lmn_exploit = 1 - mega_exploit
            
            f1_explore = pf[0] * (seq_cost_explore[0] / seq_time_explore[0])
            f1_exploit = pf[0] * (seq_cost_exploit[0] / seq_time_exploit[0])
            f2_explore = pf[1] * sum(seq_cost_explore) / sum(seq_time_explore)
            f2_exploit = pf[1] * sum(seq_cost_exploit) / sum(seq_time_exploit)
            
            min_pf_f3 = min(pf_f3) if pf_f3 else 0
            score_explore = (mega_explor * f1_explore) + (mega_explor * f2_explore) + (lmn_explore * min_pf_f3 * f3_explore)
            score_exploit = (mega_exploit * f1_exploit) + (mega_exploit * f2_exploit) + (lmn_exploit * min_pf_f3 * f3_exploit)
        
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