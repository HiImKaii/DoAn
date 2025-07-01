import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from geo_visualizer import GEOVisualizer

def plot_convergence(fun, nvars, lb, ub, options):
    """Plot convergence before animation"""
    # Create instance without visualization
    visualizer = GEOVisualizer(
        fun=fun,
        nvars=nvars,
        lb=lb,
        ub=ub,
        options=options
    )
    
    # Run optimization and collect convergence data
    best_costs = []
    mean_costs = []
    
    # Initialize population
    x = visualizer.x
    fitness = visualizer.fitness
    
    for iter in range(options['MaxIterations']):
        # Calculate vectors towards prey
        attack_vector_initial = visualizer.prey_position - x
        radius = np.sqrt(np.sum(attack_vector_initial**2, axis=1)).reshape(-1, 1)
        
        # Calculate cruise vectors
        cruise_vector_initial = 2 * np.random.rand(visualizer.pop_size, nvars) - 1
        
        # Update positions
        attack_vector = (np.random.rand(visualizer.pop_size, 1) * 
                        visualizer.attack_propensity[iter] * 
                        radius * 
                        attack_vector_initial / (radius + 1e-10))
        
        cruise_vector = (np.random.rand(visualizer.pop_size, 1) * 
                        visualizer.cruise_propensity[iter] * 
                        radius * 
                        cruise_vector_initial / (np.sqrt(np.sum(cruise_vector_initial**2, axis=1)).reshape(-1, 1) + 1e-10))
        
        # Update positions
        x = x + attack_vector + cruise_vector
        x = np.clip(x, lb, ub)
        
        # Update fitness
        fitness = np.array([fun(xi.reshape(1, -1)) for xi in x])
        
        # Store convergence data
        best_costs.append(np.min(fitness))
        mean_costs.append(np.mean(fitness))
        
        # Print progress
        print(f'Iteration: {iter + 1} Best Cost = {np.min(fitness):.6f}')
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_costs) + 1), best_costs, 'b-', label='Best Fitness')
    plt.plot(range(1, len(mean_costs) + 1), mean_costs, 'r--', label='Mean Fitness')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.title('Convergence History')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
    
    return np.min(best_costs)

def main():
    # Define optimization options
    options = {
        'PopulationSize': 25,  # 5x5 grid of eagles
        'MaxIterations': 100,
        'AttackPropensity': [0.1, 0.9],  # Attack rate increases over iterations
        'CruisePropensity': [0.1, 0.1]   # Constant small cruise rate
    }

    # Define a simple objective function (for demonstration)
    def objective_function(x):
        return np.sum(x**2)  # Simple sphere function
    
    # First show convergence plot
    best_cost = plot_convergence(
        fun=objective_function,
        nvars=2,
        lb=[-5, -5],
        ub=[5, 5],
        options=options
    )
    print(f"\nOptimization completed with best cost: {best_cost:.6f}")
    
    # Block until the convergence plot is closed
    plt.show(block=True)
    
    # Then show animation
    print("\nStarting animation...")
    visualizer = GEOVisualizer(
        fun=objective_function,
        nvars=2,              # 2D visualization
        lb=[-5, -5],         # Lower bounds
        ub=[5, 5],          # Upper bounds
        options=options
    )

    # Run the animation
    anim = visualizer.animate(interval=200)  # 200ms between frames
    plt.show()  # Make sure the plot window appears
    
if __name__ == "__main__":
    main() 