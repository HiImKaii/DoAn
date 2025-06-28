import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arrow
import matplotlib.colors as mcolors

class GEOVisualizer:
    """Visualizer for Golden Eagle Optimizer hunting process"""
    
    def __init__(self, fun, lb, ub, nvars, options):
        if nvars != 2:
            raise ValueError("Visualization is only supported for 2D functions")
            
        self.fun = fun
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.nvars = nvars
        self.options = options
        
        # Initialize population
        self.pop_size = options['PopulationSize']
        self.max_iter = options['MaxIterations']
        self.attack_propensity = np.linspace(options['AttackPropensity'][0], 
                                           options['AttackPropensity'][1], 
                                           self.max_iter)
        self.cruise_propensity = np.linspace(options['CruisePropensity'][0], 
                                           options['CruisePropensity'][1], 
                                           self.max_iter)
        
        # Create grid-based initial positions
        grid_size = int(np.sqrt(self.pop_size))
        x_coords = np.linspace(self.lb[0], self.ub[0], grid_size)
        y_coords = np.linspace(self.lb[1], self.ub[1], grid_size)
        X, Y = np.meshgrid(x_coords, y_coords)
        self.x = np.column_stack((X.ravel(), Y.ravel()))[:self.pop_size]
        
        # Set prey position at center
        self.prey_position = np.mean([self.lb, self.ub], axis=0)
        
        # Initialize fitness and memory
        self.fitness = np.array([self.fun(xi.reshape(1, -1)) for xi in self.x])
        self.flock_memory_x = self.x.copy()
        self.flock_memory_f = self.fitness.copy()
        
        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.setup_plot()
        self.setup_animation()
        
    def setup_plot(self):
        """Setup the basic plot"""
        self.ax.set_xlim(self.lb[0], self.ub[0])
        self.ax.set_ylim(self.lb[1], self.ub[1])
        self.ax.set_xlabel('x1')
        self.ax.set_ylabel('x2')
        self.ax.set_title('Golden Eagle Optimizer - Hunting Process')
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
    def setup_animation(self):
        """Setup the animation elements"""
        # Eagles (scatter plot)
        self.eagles = self.ax.scatter([], [], c='black', s=50, label='Eagles', zorder=5)
        
        # Prey position (target)
        self.prey = self.ax.scatter([], [], c='red', s=100, marker='*', 
                                  label='Prey', zorder=6)
        
        # Attack vectors (arrows)
        self.attack_arrows = []
        self.cruise_arrows = []
        
        # Add legend
        self.ax.legend()
        
        # Add iteration counter text
        self.iter_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                    verticalalignment='top')
        
    def update(self, frame):
        """Update function for animation"""
        # Clear previous arrows
        for arrow in self.attack_arrows + self.cruise_arrows:
            if arrow in self.ax.patches:
                arrow.remove()
        self.attack_arrows.clear()
        self.cruise_arrows.clear()
        
        # Update iteration counter
        self.iter_text.set_text(f'Iteration #{frame + 1}')
        
        # Calculate vectors towards prey
        attack_vector_initial = self.prey_position - self.x
        radius = np.sqrt(np.sum(attack_vector_initial**2, axis=1)).reshape(-1, 1)
        
        # Determine converged eagles
        converged_eagles = np.sum(radius, axis=1) == 0
        unconverged_eagles = ~converged_eagles
        
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
        
        # Draw arrows for unconverged eagles (optional - can be commented out for cleaner visualization)
        """
        for i in range(self.pop_size):
            if unconverged_eagles[i]:
                # Attack vector arrow
                attack_arrow = Arrow(self.x[i, 0], self.x[i, 1],
                                   attack_vector[i, 0], attack_vector[i, 1],
                                   color='red', width=0.1, alpha=0.3)
                self.ax.add_patch(attack_arrow)
                self.attack_arrows.append(attack_arrow)
        """
        
        # Update positions
        self.x = self.x + attack_vector + cruise_vector
        
        # Enforce bounds
        self.x = np.clip(self.x, self.lb, self.ub)
        
        # Update plot
        self.eagles.set_offsets(self.x)
        self.prey.set_offsets([self.prey_position])
        
        return (self.eagles, self.prey, *self.attack_arrows)
    
    def animate(self, interval=200):
        """Create and display the animation"""
        anim = FuncAnimation(self.fig, self.update, frames=self.max_iter,
                           interval=interval, blit=True)
        plt.show()
        return anim 