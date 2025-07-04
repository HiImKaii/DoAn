import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib import colormaps

class Solution:
    def __init__(self, x: np.ndarray, cost: float):
        self.X = x
        self.Cost = cost

class PumaOptimizer:
    def __init__(self, n_pumas: int = 100, max_iter: int = 100, 
                 lb: float = -100, ub: float = 100, dim: int = 30):
        """
        Puma Optimizer Algorithm implementation
        Args:
            n_pumas: Number of pumas in population
            max_iter: Maximum number of iterations
            lb: Lower bound of search space
            ub: Upper bound of search space
            dim: Dimension of search space
        """
        self.n_pumas = n_pumas
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.dim = dim
        
        # Algorithm parameters
        self.PF = np.array([0.1, 0.1, 0.05])  # Parameters for F1, F2, F3
        self.mega_explor = 0.99  #alpha
        self.mega_exploit = 0.99
        
        # For visualization
        self.history = []
        
        # Prey capture threshold
        self.capture_threshold = 0  # Distance threshold to consider prey captured
        self.captured_pumas = set()  # Track which pumas have captured prey
        
        # Movement parameters
        self.min_step = 0.1  # Minimum step size
        self.max_step = 5.0  # Maximum step size
        self.social_factor = 0.3  # Weight for social influence

    def initialize_population(self, cost_func: Callable) -> list:
        """Initialize puma population"""
        population = []
        for _ in range(self.n_pumas):
            x = np.random.uniform(self.lb, self.ub, self.dim)
            cost = cost_func(x)
            population.append(Solution(x, cost))
        return population

    def boundary_check(self, x: np.ndarray) -> np.ndarray:
        """Ensure solution stays within bounds"""
        return np.clip(x, self.lb, self.ub)

    def exploration(self, population: list, cost_func: Callable) -> list:
        """Exploration phase - searching for prey"""
        new_population = []
        
        # Calculate center of mass of uncaptured pumas
        uncaptured_positions = np.array([p.X for i, p in enumerate(population) if i not in self.captured_pumas])
        if len(uncaptured_positions) > 0:
            center_of_mass = np.mean(uncaptured_positions, axis=0)
        else:
            center_of_mass = np.zeros(self.dim)
        
        for i in range(self.n_pumas):
            if i in self.captured_pumas:
                new_population.append(population[i])
                continue
            # nếu puma đã bắt được con mồi thì không cần di chuyển nữa
            
            current_pos = population[i].X
            
            # Calculate direction to prey (origin)
            direction_to_prey = -current_pos  # Vector pointing to origin
            distance_to_prey = np.linalg.norm(direction_to_prey)
            if distance_to_prey > 0:
                direction_to_prey = direction_to_prey / distance_to_prey  # Normalize
            
            # Calculate social influence
            direction_to_center = center_of_mass - current_pos
            distance_to_center = np.linalg.norm(direction_to_center)
            if distance_to_center > 0:
                direction_to_center = direction_to_center / distance_to_center  # Normalize
            
            # Combine directions with random exploration
            r1, r2, r3 = np.random.rand(3)
            
            # Adaptive step size based on distance to prey
            step_size = self.max_step * (1 - np.exp(-distance_to_prey/50)) + self.min_step
            #xác định khoảng cách di chuyển của puma đến con mồi, puma càng xa con mồi thì khoảng cách di chuyển càng lớn

            # Calculate movement
            movement = (
                step_size * (
                    (1 - self.social_factor) * direction_to_prey * r1 +  # Move toward prey
                    self.social_factor * direction_to_center * r2 +      # Social influence
                    0.1 * (2 * np.random.rand(self.dim) - 1) * r3       # Random movement
                )
            )
            
            # Update position
            new_x = current_pos + movement
            new_x = self.boundary_check(new_x)
            new_cost = cost_func(new_x)
            new_population.append(Solution(new_x, new_cost))
            
            # Check if prey is captured
            if self.dim >= 2:
                distance_to_prey = np.sqrt(np.sum(new_x[:2]**2))
                if distance_to_prey <= self.capture_threshold:
                    self.captured_pumas.add(i)
                    print(f"Puma {i+1} has captured the prey!")
        
        return new_population

    def exploitation(self, population: list, best: Solution, iter: int, cost_func: Callable) -> list:
        """Exploitation phase - hunting prey"""
        new_population = []
        
        # Calculate center of successful pumas (captured or close to prey)
        successful_positions = []
        for i, p in enumerate(population):
            if i in self.captured_pumas or p.Cost < 10:  # Consider pumas close to prey
                successful_positions.append(p.X)
        
        if successful_positions:
            success_center = np.mean(successful_positions, axis=0)
        else:
            success_center = np.zeros(self.dim)
        
        for i in range(self.n_pumas):
            if i in self.captured_pumas:
                new_population.append(population[i])
                continue
            
            current_pos = population[i].X
            
            # Calculate direction to prey
            direction_to_prey = -current_pos
            distance_to_prey = np.linalg.norm(direction_to_prey)
            if distance_to_prey > 0:
                direction_to_prey = direction_to_prey / distance_to_prey
            
            # Calculate direction to successful pumas
            direction_to_success = success_center - current_pos
            distance_to_success = np.linalg.norm(direction_to_success)
            if distance_to_success > 0:
                direction_to_success = direction_to_success / distance_to_success
            
            # Adaptive step size
            a = self.max_step * (1 - iter/self.max_iter) + self.min_step
            
            # Combine movements
            r = np.random.rand()
            if r < 0.7:  # Higher probability of directed movement
                movement = a * (
                    0.7 * direction_to_prey +  # Strong attraction to prey
                    0.3 * direction_to_success  # Influence of successful pumas
                )
            else:
                # Local search around current position
                movement = a * 0.5 * (2 * np.random.rand(self.dim) - 1)
            
            # Update position
            new_x = current_pos + movement
            new_x = self.boundary_check(new_x)
            new_cost = cost_func(new_x)
            new_population.append(Solution(new_x, new_cost))
            
            # Check if prey is captured
            if self.dim >= 2:
                distance_to_prey = np.sqrt(np.sum(new_x[:2]**2))
                if distance_to_prey <= self.capture_threshold:
                    self.captured_pumas.add(i)
                    print(f"Puma {i+1} has captured the prey!")
        
        return new_population

    def print_iteration_params(self, iter: int, score_explore: float, score_exploit: float, 
                             f1_explor: float = 0, f1_exploit: float = 0,
                             f2_explor: float = 0, f2_exploit: float = 0,
                             f3_explore: float = 0, f3_exploit: float = 0,
                             best_cost: float = 0,
                             current_phase: str = ""):
        """Print parameters for current iteration"""
        print("\n" + "="*80)
        print(f"Iteration {iter+1}/{self.max_iter}")
        print("="*80)
        
        # Print phase information
        if iter < 3:
            print("Stage: Unexperienced Phase (Initial Learning)")
        else:
            print("Stage: Experienced Phase (Adaptive Learning)")
        print(f"Current Phase: {current_phase}")
        
        print(f"\nBest Cost: {best_cost:.6f}")
        print(f"Pumas that caught prey: {len(self.captured_pumas)}/{self.n_pumas}")
        
        print("\nScores:")
        print(f"Exploration Score: {score_explore:.6f}")
        print(f"Exploitation Score: {score_exploit:.6f}")
        
        print("\nMega Parameters:")
        print(f"Mega Exploration: {self.mega_explor:.6f}")
        print(f"Mega Exploitation: {self.mega_exploit:.6f}")
        
        print("\nFitness Components:")
        print(f"F1 Exploration: {f1_explor:.6f}")
        print(f"F1 Exploitation: {f1_exploit:.6f}")
        print(f"F2 Exploration: {f2_explor:.6f}")
        print(f"F2 Exploitation: {f2_exploit:.6f}")
        print(f"F3 Exploration: {f3_explore:.6f}")
        print(f"F3 Exploitation: {f3_exploit:.6f}")
        
        print("-"*80 + "\n")

    def optimize(self, cost_func: Callable) -> tuple:
        """Main optimization loop"""
        # Reset captured pumas for new optimization
        self.captured_pumas = set()
        
        # Initialize population
        population = self.initialize_population(cost_func)
        best = min(population, key=lambda x: x.Cost)
        initial_best = best
        
        # For tracking convergence
        convergence = np.zeros(self.max_iter)
        
        # For visualization
        self.history = []
        
        # Algorithm parameters
        unselected = np.ones(2)  # [exploration, exploitation]
        f3_explore = f3_exploit = 0
        seq_time_explore = np.ones(3)
        seq_time_exploit = np.ones(3)
        seq_cost_explore = np.ones(3)
        seq_cost_exploit = np.ones(3)
        pf_f3 = []
        flag_change = 1
        
        # Initialize variables for first iterations
        f1_explor = f1_exploit = f2_explor = f2_exploit = 0
        score_explore = score_exploit = 0
        current_phase = "Initialization"
        
        # Main loop
        for iter in range(self.max_iter):
            # Store current positions for visualization
            if self.dim >= 2:  # Only store if we have at least 2D
                current_positions = np.array([p.X[:2] for p in population])  # Take first 2 dimensions
                self.history.append(current_positions)
            
            # Calculate scores for iter >= 3
            if iter >= 3:
                score_explore = (self.mega_explor * f1_explor + 
                               self.mega_explor * f2_explor + 
                               (1 - self.mega_explor) * min(pf_f3) * f3_explore 
                               if pf_f3 else 0)
                               
                score_exploit = (self.mega_exploit * f1_exploit + 
                               self.mega_exploit * f2_exploit + 
                               (1 - self.mega_exploit) * min(pf_f3) * f3_exploit 
                               if pf_f3 else 0)
            
            # Select and execute phase
            if iter < 3:
                # In unexperienced phase, alternate between exploration and exploitation
                if iter % 2 == 0:
                    current_phase = "Exploration (Unexperienced)"
                    new_population = self.exploration(population, cost_func)
                    f3_explore = self.PF[2]
                    f3_exploit += self.PF[2]
                else:
                    current_phase = "Exploitation (Unexperienced)"
                    new_population = self.exploitation(population, best, iter, cost_func)
                    f3_explore += self.PF[2]
                    f3_exploit = self.PF[2]
            else:
                # In experienced phase, select based on scores
                if score_explore > score_exploit:
                    current_phase = "Exploration (Experienced)"
                    new_population = self.exploration(population, cost_func)
                    unselected[1] += 1
                    unselected[0] = 1
                    f3_explore = self.PF[2]
                    f3_exploit += self.PF[2]
                else:
                    current_phase = "Exploitation (Experienced)"
                    new_population = self.exploitation(population, best, iter, cost_func)
                    unselected[0] += 1
                    unselected[1] = 1
                    f3_explore += self.PF[2]
                    f3_exploit = self.PF[2]
            
            # Update population
            population = new_population
            current_best = min(population, key=lambda x: x.Cost)
            
            if current_best.Cost < best.Cost:
                best = current_best
            
            # Update parameters
            if iter >= 2:
                # Update sequence costs and times
                seq_cost_explore = np.roll(seq_cost_explore, 1)
                seq_cost_exploit = np.roll(seq_cost_exploit, 1)
                seq_time_explore = np.roll(seq_time_explore, 1)
                seq_time_exploit = np.roll(seq_time_exploit, 1)
                
                seq_cost_explore[0] = abs(best.Cost - current_best.Cost)
                seq_cost_exploit[0] = abs(best.Cost - current_best.Cost)
                
                if seq_cost_explore[0] != 0:
                    pf_f3.append(seq_cost_explore[0])
                if seq_cost_exploit[0] != 0:
                    pf_f3.append(seq_cost_exploit[0])
                
                # Calculate F1 and F2
                f1_explor = self.PF[0] * (seq_cost_explore[0] / seq_time_explore[0])
                f1_exploit = self.PF[0] * (seq_cost_exploit[0] / seq_time_exploit[0])
                
                f2_explor = self.PF[1] * (sum(seq_cost_explore) / sum(seq_time_explore))
                f2_exploit = self.PF[1] * (sum(seq_cost_exploit) / sum(seq_time_exploit))
                
                # Update mega parameters
                if score_explore < score_exploit:
                    self.mega_explor = max(self.mega_explor - 0.01, 0.01)
                    self.mega_exploit = 0.99
                elif score_explore > score_exploit:
                    self.mega_explor = 0.99
                    self.mega_exploit = max(self.mega_exploit - 0.01, 0.01)
            
            convergence[iter] = best.Cost
            
            # Print iteration parameters
            self.print_iteration_params(
                iter=iter,
                score_explore=score_explore,
                score_exploit=score_exploit,
                f1_explor=f1_explor,
                f1_exploit=f1_exploit,
                f2_explor=f2_explor,
                f2_exploit=f2_exploit,
                f3_explore=f3_explore,
                f3_exploit=f3_exploit,
                best_cost=best.Cost,
                current_phase=current_phase
            )
            
        return best.X, best.Cost, convergence

    def visualize_hunting(self, interval: int = 100):
        """
        Visualize the hunting process with 5x5 puma formation
        Args:
            interval: Animation interval in milliseconds
        """
        if len(self.history) == 0:
            print("No hunting history available. Run optimize() first.")
            return
        
        # Setup the figure
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xlim(self.lb, self.ub)
        ax.set_ylim(self.lb, self.ub)
        
        # Initialize puma markers
        pumas = []
        colors = colormaps['viridis'](np.linspace(0, 1, self.n_pumas))
        
        # Create 5x5 grid positions for initial formation
        grid_x = np.linspace(-80, 80, 5)
        grid_y = np.linspace(-80, 80, 5)
        X, Y = np.meshgrid(grid_x, grid_y)
        grid_positions = np.column_stack((X.ravel(), Y.ravel()))
        
        # Initialize pumas with triangular markers in grid formation
        for i in range(self.n_pumas):
            puma = ax.scatter([], [], marker='^', s=100, color=colors[i], alpha=0.6)
            pumas.append(puma)
        
        # Add target (prey) at origin
        target = patches.Circle((0, 0), 3, color='red', fill=True)
        ax.add_patch(target)
        
        # Add hunting range visualization
        hunting_range = patches.Circle((0, 0), 90, color='green', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(hunting_range)
        
        # Title and labels
        ax.set_title('Puma Hunting Simulation (5x5 Formation)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        def init():
            for puma in pumas:
                puma.set_offsets(np.c_[[], []])
            return pumas
        
        def animate(frame):
            positions = self.history[frame]
            for i, puma in enumerate(pumas):
                puma.set_offsets(positions[i])
            
            # Update title with frame information
            ax.set_title(f'Puma Hunting Simulation - Iteration {frame+1}/{len(self.history)}')
            return pumas
        
        anim = FuncAnimation(
            fig, animate, init_func=init, frames=len(self.history),
            interval=interval, blit=True
        )
        
        plt.grid(True)
        plt.show()
        
        return anim

# Example usage with a test function
def hunting_cost(x: np.ndarray) -> float:
    """Cost function for hunting simulation - based on distance to prey"""
    if len(x.shape) == 1:
        return float(np.sqrt(np.sum(x**2)))  # Distance to origin (prey)
    return float(np.sqrt(np.sum(x**2, axis=1)))  # For multiple points

if __name__ == "__main__":
    # Initialize optimizer with 25 pumas and 2D space for visualization
    optimizer = PumaOptimizer(n_pumas=100, max_iter=100, lb=-100, ub=100, dim=2)
    
    # Run optimization with hunting cost function
    best_x, best_cost, convergence = optimizer.optimize(hunting_cost)
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(convergence)
    plt.title('Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Distance to Prey')
    plt.grid(True)
    plt.show()
    
    # Visualize hunting process
    optimizer.visualize_hunting(interval=100)
    
    print(f"\nHunting Results:")
    print(f"Final Distance to Prey: {best_cost:.6f}")
    print(f"Best Position: {best_x}")
    print(f"Total Pumas that caught prey: {len(optimizer.captured_pumas)}") 