import numpy as np
from scipy.optimize import minimize

# ... existing code ...

def wso_optimization(objective_func, variables, lb, ub, max_iter=1000, w=0.7, c1=0.3, c2=0.6, k=2, p=2):
    # Diversified population initialization
    population = np.random.uniform(lb, ub, (10, variables.shape[0]))

    for i in range(max_iter):
        # Calculate fitness for each candidate
        fitness = np.array([objective_func(c) for c in population])

        # Update best and worst candidates
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)
        best_candidate = population[best_idx]
        worst_candidate = population[worst_idx]

        # Calculate new step sizes
        delta = (ub - lb) * 0.1
        step_sizes = delta * np.exp(-i / k)

        # Update candidates using WSO logic
        for j in range(10):
            if j != best_idx:
                r1, r2, r3 = np.random.rand(3)
                a = 2 * r1 - 1
                c = 2 * r2
                d = 2 * r3 - 1
                population[j] = best_candidate + a * (best_candidate - population[j]) + c * (best_candidate - worst_candidate) + d * (best_candidate - np.random.uniform(lb, ub))

                # Adjust step size using learning rate decay
                step_sizes[j] *= 0.995

                # Apply lower and upper bounds
                population[j] = np.clip(population[j], lb, ub)

        print(f"Iteration {i+1}: Best fitness -> {fitness[best_idx]:.4f}")

    return best_candidate

# ... rest of your code ...

# Now you can call the improved WSO function like this:
result = wso_optimization(objective_function, variables, lb, ub)