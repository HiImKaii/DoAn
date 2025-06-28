import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def sphere(x: np.ndarray) -> float:
    """Sphere function as the cost/fitness function"""
    return float(np.sum(x**2))

class Particle:
    """Lớp đại diện cho một hạt trong thuật toán PSO"""
    
    def __init__(self, dim: int, var_min: float, var_max: float):
        self.position = np.random.uniform(var_min, var_max, dim)
        self.velocity = np.zeros(dim)
        self.cost = float('inf')
        self.best_position = self.position.copy()
        self.best_cost = float('inf')
        # Lưu lại lịch sử vị trí để vẽ
        self.position_history = [self.position.copy()]

def visualize_pso_2d(particles_history: List[List[np.ndarray]], 
                    var_min: float, var_max: float,
                    cost_func: Callable[[np.ndarray], float],
                    title: str = "PSO Optimization Process"):
    """
    Tạo animation cho quá trình PSO trong không gian 2D
    
    Parameters:
    -----------
    particles_history : List[List[np.ndarray]]
        Lịch sử vị trí của tất cả các hạt
    var_min, var_max : float
        Giới hạn của không gian tìm kiếm
    cost_func : Callable
        Hàm mục tiêu để vẽ contour
    title : str
        Tiêu đề của đồ thị
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Tạo lưới điểm để vẽ contour
    x = np.linspace(var_min, var_max, 100)
    y = np.linspace(var_min, var_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Tính giá trị hàm mục tiêu tại mỗi điểm
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = cost_func(np.array([X[i,j], Y[i,j]]))
    
    # Vẽ contour của hàm mục tiêu
    contour = ax.contour(X, Y, Z, levels=20)
    fig.colorbar(contour)
    
    # Khởi tạo scatter plot cho các hạt
    scatter = ax.scatter([], [], c='red', marker='o')
    
    # Thiết lập giới hạn trục
    ax.set_xlim(var_min, var_max)
    ax.set_ylim(var_min, var_max)
    ax.set_title(title)
    
    def update(frame):
        # Cập nhật vị trí của các hạt
        positions = np.array([history[frame] for history in particles_history])
        scatter.set_offsets(positions)
        ax.set_title(f"{title} - Iteration {frame}")
        return scatter,
    
    # Tạo animation
    anim = FuncAnimation(
        fig, update,
        frames=len(particles_history[0]),
        interval=100,
        blit=True
    )
    
    plt.show()

def pso(cost_func: Callable[[np.ndarray], float] = sphere, 
        n_var: int = 10, 
        var_min: float = -10, 
        var_max: float = 10, 
        max_it: int = 1000, 
        n_pop: int = 100,
        w: float = 1, 
        w_damp: float = 0.99, 
        c1: float = 1.5, 
        c2: float = 2.0,
        visualize: bool = False) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Particle Swarm Optimization
    
    Parameters:
    -----------
    cost_func : function
        The cost/fitness function to minimize
    n_var : int
        Number of decision variables
    var_min : float
        Lower bound of variables
    var_max : float
        Upper bound of variables
    max_it : int
        Maximum number of iterations
    n_pop : int
        Population size (swarm size)
    w : float
        Inertia weight
    w_damp : float
        Inertia weight damping ratio
    c1 : float
        Personal learning coefficient
    c2 : float
        Global learning coefficient
    visualize : bool
        Nếu True và n_var=2, sẽ tạo animation cho quá trình tối ưu
        
    Returns:
    --------
    Tuple[np.ndarray, float, np.ndarray]
        Returns (best_position, best_cost, best_costs_history)
    """
    
    # Initialize velocity limits
    vel_max = 0.1 * (var_max - var_min)
    vel_min = -vel_max
    
    # Initialize population
    particles = [Particle(n_var, var_min, var_max) for _ in range(n_pop)]
    
    # Initialize global best
    global_best_cost = float('inf')
    global_best_position = np.zeros(n_var)
    
    # Evaluate initial population
    for particle in particles:
        particle.cost = float(cost_func(particle.position))
        particle.best_cost = particle.cost
        
        if particle.best_cost < global_best_cost:
            global_best_cost = particle.best_cost
            global_best_position = particle.position.copy()
    
    # Array to hold best costs
    best_costs = np.zeros(max_it, dtype=float)
    
    # PSO Main Loop
    for it in range(max_it):
        for particle in particles:
            # Update Velocity
            particle.velocity = (w * particle.velocity +
                               c1 * np.random.rand(n_var) * (particle.best_position - particle.position) +
                               c2 * np.random.rand(n_var) * (global_best_position - particle.position))
            
            # Apply Velocity Limits
            particle.velocity = np.clip(particle.velocity, vel_min, vel_max)
            
            # Update Position
            particle.position = particle.position + particle.velocity
            
            # Velocity Mirror Effect
            is_outside = (particle.position < var_min) | (particle.position > var_max)
            particle.velocity[is_outside] = -particle.velocity[is_outside]
            
            # Apply Position Limits
            particle.position = np.clip(particle.position, var_min, var_max)
            
            # Lưu lại vị trí mới nếu cần visualize
            if visualize and n_var == 2:
                particle.position_history.append(particle.position.copy())
            
            # Evaluation
            particle.cost = float(cost_func(particle.position))
            
            # Update Personal Best
            if particle.cost < particle.best_cost:
                particle.best_position = particle.position.copy()
                particle.best_cost = particle.cost
                
                # Update Global Best
                if particle.best_cost < global_best_cost:
                    global_best_cost = particle.best_cost
                    global_best_position = particle.position.copy()
        
        # Store Best Cost
        best_costs[it] = float(global_best_cost)
        
        # Show Iteration Information
        print(f'Iteration {it+1}: Best Cost = {best_costs[it]:.10f}')
        
        # Update Inertia Weight
        w *= w_damp
    
    # Visualization for 2D problems
    if visualize and n_var == 2:
        particles_history = [p.position_history for p in particles]
        visualize_pso_2d(particles_history, var_min, var_max, cost_func)
    
    # Results
    plt.figure()
    plt.semilogy(range(1, max_it + 1), best_costs, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Cost')
    plt.grid(True)
    plt.title('Convergence Curve')
    plt.show()
    
    return global_best_position, global_best_cost, best_costs

# Định nghĩa các hàm test bổ sung
def rastrigin(x: np.ndarray) -> float:
    """Hàm Rastrigin - phức tạp hơn với nhiều local minima"""
    A = 10
    n = len(x)
    return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

def rosenbrock(x: np.ndarray) -> float:
    """Hàm Rosenbrock - hàm "banana" nổi tiếng"""
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

if __name__ == '__main__':
    print("=== THUẬT TOÁN PSO CHO BÀI TOÁN TỐI ƯU ===\n")
    
    # Test với hàm Sphere (2D)
    print("1. Tối ưu hàm Sphere (2D):")
    print("   f(x,y) = x² + y²")
    print("   Minimum lý thuyết: f(0,0) = 0")
    best_position, best_cost, _ = pso(sphere, n_var=2, max_it=50, n_pop=30, visualize=True)
    print(f"\nKết quả:")
    print(f"   Vị trí tốt nhất: ({best_position[0]:.6f}, {best_position[1]:.6f})")
    print(f"   Giá trị tốt nhất: {best_cost:.6f}")
    
    # Test với hàm Rastrigin (2D)
    print("\n" + "="*50)
    print("2. Tối ưu hàm Rastrigin (2D):")
    print("   f(x,y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy))")
    print("   Minimum lý thuyết: f(0,0) = 0")
    best_position, best_cost, _ = pso(rastrigin, n_var=2, var_min=-5.12, var_max=5.12, 
                                    max_it=100, n_pop=50, visualize=True)
    print(f"\nKết quả:")
    print(f"   Vị trí tốt nhất: ({best_position[0]:.6f}, {best_position[1]:.6f})")
    print(f"   Giá trị tốt nhất: {best_cost:.6f}")
    
    # Test với hàm Rosenbrock (2D)
    print("\n" + "="*50)
    print("3. Tối ưu hàm Rosenbrock (2D):")
    print("   Minimum lý thuyết: f(1,1) = 0")
    best_position, best_cost, _ = pso(rosenbrock, n_var=2, var_min=-5, var_max=10, 
                                    max_it=100, n_pop=50, visualize=True)
    print(f"\nKết quả:")
    print(f"   Vị trí tốt nhất: ({best_position[0]:.6f}, {best_position[1]:.6f})")
    print(f"   Giá trị tốt nhất: {best_cost:.6f}")