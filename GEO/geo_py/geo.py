import numpy as np

class GEO:
    """
    Golden Eagle Optimizer (GEO) implementation in Python
    Based on the paper: Golden Eagle Optimizer: A nature-inspired metaheuristic algorithm
    by Abdolkarim Mohammadi-Balani, et al.
    """
    
    def __init__(self, fun, nvars, lb, ub, options):
        """
        Initialize the GEO optimizer
        
        Parameters:
        -----------
        fun : callable
            The objective function to minimize
        nvars : int
            Number of variables
        lb : array_like
            Lower bounds for variables
        ub : array_like
            Upper bounds for variables
        options : dict
            Algorithm options including:
            - PopulationSize: int
            - MaxIterations: int
            - AttackPropensity: list [start, end]
            - CruisePropensity: list [start, end]
        """
        self.fun = fun
        self.nvars = nvars
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.options = options
        
    @staticmethod
    def vec_norm(A, p, axis=1):
        """Compute the p-norm of vectors along specified axis"""
        return np.power(np.sum(np.power(np.abs(A), p), axis=axis), 1/p)
    
    def optimize(self):
        """Run the optimization algorithm"""
        
        # Initialization
        pop_size = self.options['PopulationSize']
        max_iter = self.options['MaxIterations']
        
        convergence_curve = np.zeros(max_iter)
        
        # Initialize population
        x = self.lb + np.random.rand(pop_size, self.nvars) * (self.ub - self.lb)
        
        # Evaluate initial population
        fitness_scores = np.array([self.fun(xi.reshape(1, -1)) for xi in x])
        
        # Initialize memory
        flock_memory_f = fitness_scores.copy()
        flock_memory_x = x.copy()
        
        # Initialize propensity parameters
        attack_propensity = np.linspace(
            self.options['AttackPropensity'][0],
            self.options['AttackPropensity'][1],
            max_iter
        )
        cruise_propensity = np.linspace(
            self.options['CruisePropensity'][0],
            self.options['CruisePropensity'][1],
            max_iter
        )
        
        # Main loop
        for current_iter in range(max_iter):
            # Prey selection (one-to-one mapping)
            destination_eagle = np.random.permutation(pop_size)
            
            # Calculate AttackVectorInitial
            attack_vector_initial = flock_memory_x[destination_eagle] - x
            
            # Calculate Radius
            radius = self.vec_norm(attack_vector_initial, 2)
            radius = radius.reshape(-1, 1)
            
            # Determine converged and unconverged eagles
            converged_eagles = (np.sum(radius, axis=1) == 0)
            unconverged_eagles = ~converged_eagles
            
            # Initialize CruiseVectorInitial
            cruise_vector_initial = 2 * np.random.rand(pop_size, self.nvars) - 1  # [-1,1]
            
            # Correct vectors for converged eagles
            attack_vector_initial[converged_eagles] = 0
            cruise_vector_initial[converged_eagles] = 0
            
            # Determine constrained and free variables
            for i in range(pop_size):
                if unconverged_eagles[i]:
                    v_constrained = np.zeros(self.nvars, dtype=bool)
                    idx = np.random.choice(np.where(attack_vector_initial[i] != 0)[0], 1)[0]
                    v_constrained[idx] = True
                    v_free = ~v_constrained
                    
                    # Calculate constrained components
                    if np.any(v_constrained):
                        cruise_vector_initial[i, v_constrained] = (
                            -np.sum(attack_vector_initial[i, v_free] * cruise_vector_initial[i, v_free])
                            / attack_vector_initial[i, v_constrained]
                        )
            
            # Calculate unit vectors
            attack_vector_unit = attack_vector_initial / (self.vec_norm(attack_vector_initial, 2).reshape(-1, 1) + 1e-10)
            cruise_vector_unit = cruise_vector_initial / (self.vec_norm(cruise_vector_initial, 2).reshape(-1, 1) + 1e-10)
            
            # Correct vectors for converged eagles
            attack_vector_unit[converged_eagles] = 0
            cruise_vector_unit[converged_eagles] = 0
            
            # Calculate movement vectors
            attack_vector = (np.random.rand(pop_size, 1) * 
                           attack_propensity[current_iter] * 
                           radius * 
                           attack_vector_unit)
            
            cruise_vector = (np.random.rand(pop_size, 1) * 
                           cruise_propensity[current_iter] * 
                           radius * 
                           cruise_vector_unit)
            
            step_vector = attack_vector + cruise_vector
            
            # Update positions
            x = x + step_vector
            
            # Enforce bounds
            x = np.clip(x, self.lb, self.ub)
            
            # Calculate fitness
            fitness_scores = np.array([self.fun(xi.reshape(1, -1)) for xi in x])
            
            # Update memory
            update_mask = fitness_scores < flock_memory_f
            flock_memory_f[update_mask] = fitness_scores[update_mask]
            flock_memory_x[update_mask] = x[update_mask]
            
            # Update convergence curve
            convergence_curve[current_iter] = np.min(flock_memory_f)
        
        # Return best solution
        best_idx = np.argmin(flock_memory_f)
        best_x = flock_memory_x[best_idx]
        best_f = flock_memory_f[best_idx]
        
        print(f'Best solution obtained by GEO: {best_x}')
        print(f'Best objective function value obtained by GEO: {best_f}')
        
        return best_x, best_f, convergence_curve 