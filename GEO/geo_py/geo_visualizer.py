import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arrow
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

class GEOVisualizer:
    """Visualizer for Golden Eagle Optimizer hunting process"""
    
    def __init__(self, fun, nvars, lb, ub, options):
        self.fun = fun
        self.nvars = nvars
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.pop_size = options['PopulationSize']
        self.max_iter = options['MaxIterations']
        
        # Initialize attack and cruise propensities
        attack_start, attack_end = options['AttackPropensity']
        cruise_start, cruise_end = options['CruisePropensity']
        
        self.attack_propensity = np.linspace(attack_start, attack_end, self.max_iter)
        self.cruise_propensity = np.linspace(cruise_start, cruise_end, self.max_iter)
        
        # Initialize population
        self.x = np.random.uniform(lb, ub, (self.pop_size, nvars))
        self.fitness = np.array([self.fun(xi.reshape(1, -1)) for xi in self.x])
        
        # Initialize prey position (best solution)
        self.prey_position = self.x[np.argmin(self.fitness)]
        
        # Setup the figure
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(lb[0], ub[0])
        self.ax.set_ylim(lb[1], ub[1])
        self.ax.set_title('Golden Eagle Optimizer - Hunting Process')
        
        # Initialize scatter plots
        self.eagles_scatter = self.ax.scatter([], [], c='black', marker='o', label='Eagles')
        self.prey_scatter = self.ax.scatter([], [], c='red', marker='*', s=200, label='Prey')
        self.ax.legend()
        
        # Store animation frames
        self.frames = []
    
    def animate(self, interval=200):
        def update(frame):
            # Remove previous scatter plots
            for collection in self.ax.collections:
                collection.remove()
            
            # Calculate vectors towards prey
            attack_vector_initial = self.prey_position - self.x
            radius = np.sqrt(np.sum(attack_vector_initial**2, axis=1)).reshape(-1, 1)
            
            # Calculate cruise vectors
            cruise_vector_initial = 2 * np.random.rand(self.pop_size, self.nvars) - 1
            
            # Update positions
            attack_vector = (np.random.rand(self.pop_size, 1) * 
                           self.attack_propensity[frame] * 
                           radius * 
                           attack_vector_initial / (radius + 1e-10))
            
            cruise_vector = (np.random.rand(self.pop_size, 1) * 
                           self.cruise_propensity[frame] * 
                           radius * 
                           cruise_vector_initial / (np.sqrt(np.sum(cruise_vector_initial**2, axis=1)).reshape(-1, 1) + 1e-10))
            
            # Update positions
            self.x = self.x + attack_vector + cruise_vector
            self.x = np.clip(self.x, self.lb, self.ub)
            
            # Update fitness
            self.fitness = np.array([self.fun(xi.reshape(1, -1)) for xi in self.x])
            
            # Update prey position
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.fun(self.prey_position.reshape(1, -1)):
                self.prey_position = self.x[best_idx]
            
            # Update plot
            self.eagles_scatter = self.ax.scatter(self.x[:, 0], self.x[:, 1], c='black', marker='o', label='Eagles')
            self.prey_scatter = self.ax.scatter(self.prey_position[0], self.prey_position[1], c='red', marker='*', s=200, label='Prey')
            
            if frame == 0:
                self.ax.legend()
            
            return self.eagles_scatter, self.prey_scatter
        
        anim = FuncAnimation(
            self.fig, 
            update, 
            frames=self.max_iter,
            interval=interval, 
            blit=True
        )
        
        return anim 