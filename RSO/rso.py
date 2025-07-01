import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import warnings
warnings.filterwarnings('ignore')

class RandomSearchOptimizer:
    def __init__(self, objective_function, bounds, maximize=False):
        """
        Khởi tạo Random Search Optimizer
        
        Args:
            objective_function: Hàm mục tiêu cần tối ưu
            bounds: List các tuple (min, max) cho mỗi chiều
            maximize: True nếu muốn tối đa hóa, False nếu muốn tối thiểu hóa
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.maximize = maximize
        self.dimension = len(bounds)
        
        # Lưu trữ lịch sử tìm kiếm
        self.history = []
        self.best_solution = None
        self.best_value = None
        
    def generate_random_solution(self):
        """Tạo một solution ngẫu nhiên trong không gian tìm kiếm"""
        solution = []
        for i in range(self.dimension):
            min_val, max_val = self.bounds[i]
            solution.append(random.uniform(min_val, max_val))
        return np.array(solution)
    
    def evaluate(self, solution):
        """Đánh giá solution"""
        return self.objective_function(solution)
    
    def optimize(self, num_iterations=1000, verbose=True):
        """
        Thực hiện tối ưu hóa Random Search
        
        Args:
            num_iterations: Số lần lặp
            verbose: In thông tin quá trình tìm kiếm
        """
        if verbose:
            print(f"Bắt đầu Random Search với {num_iterations} iterations...")
            print(f"Mục tiêu: {'Maximize' if self.maximize else 'Minimize'}")
            print(f"Số chiều: {self.dimension}")
            print(f"Bounds: {self.bounds}")
            print("-" * 50)
        
        for i in range(num_iterations):
            # Tạo solution ngẫu nhiên
            solution = self.generate_random_solution()
            value = self.evaluate(solution)
            
            # Lưu vào lịch sử
            self.history.append({
                'iteration': i + 1,
                'solution': solution.copy(),
                'value': value
            })
            
            # Cập nhật best solution
            if self.best_solution is None:
                self.best_solution = solution.copy()
                self.best_value = value
            else:
                if self.maximize:
                    if value > self.best_value:
                        self.best_solution = solution.copy()
                        self.best_value = value
                else:
                    if value < self.best_value:
                        self.best_solution = solution.copy()
                        self.best_value = value
            
            # In thông tin mỗi 100 iterations
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}: Best value = {self.best_value:.6f}")
        
        if verbose:
            print("-" * 50)
            print(f"Kết thúc! Best solution: {self.best_solution}")
            print(f"Best value: {self.best_value:.6f}")
        
        return self.best_solution, self.best_value
    
    def plot_convergence(self):
        """Vẽ đồ thị hội tụ"""
        if not self.history:
            print("Chưa có dữ liệu để vẽ đồ thị!")
            return
        
        try:
            iterations = [h['iteration'] for h in self.history]
            values = [h['value'] for h in self.history]
            
            # Tính best value tại mỗi iteration
            best_values = []
            current_best = values[0]
            
            for value in values:
                if self.maximize:
                    if value > current_best:
                        current_best = value
                else:
                    if value < current_best:
                        current_best = value
                best_values.append(current_best)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Subplot 1: Tất cả các giá trị
            ax1.scatter(iterations, values, alpha=0.6, s=10)
            ax1.plot(iterations, best_values, 'r-', linewidth=2, label='Best so far')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Objective Value')
            ax1.set_title('Random Search - Tất cả các điểm')
            ax1.legend()
            ax1.grid(True)
            
            # Subplot 2: Đường hội tụ
            ax2.plot(iterations, best_values, 'r-', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Best Objective Value')
            ax2.set_title('Đường hội tụ')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Lỗi khi vẽ đồ thị convergence: {e}")
            print("Thông tin cơ bản:")
            print(f"Best value: {self.best_value}")
            print(f"Best solution: {self.best_solution}")
            print(f"Số iterations: {len(self.history)}")
    
    def plot_search_space_2d(self, num_points=100):
        """Vẽ không gian tìm kiếm 2D (chỉ hoạt động với bài toán 2 chiều)"""
        if self.dimension != 2:
            print("Visualization 2D chỉ hoạt động với bài toán 2 chiều!")
            return
        
        if not self.history:
            print("Chưa có dữ liệu để vẽ đồ thị!")
            return
        
        try:
            # Tạo grid để vẽ contour
            x_min, x_max = self.bounds[0]
            y_min, y_max = self.bounds[1]
            
            x = np.linspace(x_min, x_max, 50)
            y = np.linspace(y_min, y_max, 50)
            X, Y = np.meshgrid(x, y)
            
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.objective_function(np.array([X[i, j], Y[i, j]]))
            
            fig = plt.figure(figsize=(12, 5))
            
            # Subplot 1: Contour với search points
            ax1 = fig.add_subplot(1, 2, 1)
            contour = ax1.contour(X, Y, Z, levels=20, alpha=0.6)
            plt.colorbar(contour, ax=ax1)
            
            # Vẽ các điểm đã search
            search_x = [h['solution'][0] for h in self.history]
            search_y = [h['solution'][1] for h in self.history]
            ax1.scatter(search_x, search_y, c='red', s=10, alpha=0.6, label='Search points')
            
            # Vẽ best solution nếu có
            if self.best_solution is not None and self.best_value is not None:
                ax1.scatter(self.best_solution[0], self.best_solution[1], 
                           c='yellow', s=100, marker='*', edgecolor='black', 
                           label=f'Best solution ({self.best_value:.4f})')
            
            ax1.set_xlabel('x1')
            ax1.set_ylabel('x2')
            ax1.set_title('Random Search trong không gian 2D')
            ax1.legend()
            
            # Subplot 2: Surface plot
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            surf = ax2.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
            
            # Vẽ search points trên surface
            search_z = [h['value'] for h in self.history]
            ax2.scatter(search_x, search_y, search_z, color='red', marker='o', alpha=0.8)
            
            # Vẽ best point nếu có
            if self.best_solution is not None and self.best_value is not None:
                ax2.scatter(self.best_solution[0], self.best_solution[1], self.best_value,
                          color='yellow', marker='*')
            
            ax2.set_xlabel('x1')
            ax2.set_ylabel('x2')
            ax2.set_zlabel('f(x1, x2)')
            ax2.set_title('Surface Plot với Search Points')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Lỗi khi vẽ đồ thị 2D: {e}")
            print("Chỉ hiển thị kết quả số:")
            print(f"Best solution: {self.best_solution}")
            print(f"Best value: {self.best_value}")
            print(f"Tổng số điểm đã search: {len(self.history)}")

# Định nghĩa một số hàm test phổ biến
def sphere_function(x):
    """Hàm Sphere: f(x) = sum(x_i^2)"""
    return np.sum(x**2)

def rosenbrock_function(x):
    """Hàm Rosenbrock: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)"""
    result = 0
    for i in range(len(x) - 1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return result

def rastrigin_function(x):
    """Hàm Rastrigin: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def himmelblau_function(x):
    """Hàm Himmelblau: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2"""
    if len(x) != 2:
        raise ValueError("Himmelblau function chỉ dành cho 2 biến")
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

# Demo chương trình
if __name__ == "__main__":
    print("=== RANDOM SEARCH OPTIMIZATION DEMO ===\n")
    
    try:
        # Test với hàm Sphere (2D)
        print("1. Test với Sphere Function (2D)")
        print("f(x,y) = x^2 + y^2")
        print("Global minimum: f(0,0) = 0")
        
        bounds = [(-5, 5), (-5, 5)]
        optimizer = RandomSearchOptimizer(sphere_function, bounds, maximize=False)
        best_sol, best_val = optimizer.optimize(num_iterations=500)
        
        # Visualization
        optimizer.plot_convergence()
        optimizer.plot_search_space_2d()
        
        print("\n" + "="*60 + "\n")
        
        # Test với hàm Himmelblau (2D)
        print("2. Test với Himmelblau Function (2D)")
        print("f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2")
        print("Global minima: f(3,2)=0, f(-2.8,3.1)=0, f(-3.8,-3.3)=0, f(3.6,-1.8)=0")
        
        bounds = [(-5, 5), (-5, 5)]
        optimizer2 = RandomSearchOptimizer(himmelblau_function, bounds, maximize=False)
        best_sol2, best_val2 = optimizer2.optimize(num_iterations=1000)
        
        # Visualization
        optimizer2.plot_convergence()
        optimizer2.plot_search_space_2d()
        
        print("\n" + "="*60 + "\n")
        
        # Test với hàm Rastrigin (nhiều chiều)
        print("3. Test với Rastrigin Function (5D)")
        print("f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i)), A=10")
        print("Global minimum: f(0,0,0,0,0) = 0")
        
        bounds = [(-5.12, 5.12) for _ in range(5)]
        optimizer3 = RandomSearchOptimizer(rastrigin_function, bounds, maximize=False)
        best_sol3, best_val3 = optimizer3.optimize(num_iterations=2000)
        
        # Chỉ vẽ convergence cho bài toán nhiều chiều
        optimizer3.plot_convergence()
        
        print("\nDemo hoàn thành!")
        
    except Exception as e:
        print(f"Lỗi trong quá trình chạy demo: {e}")
        print("Thử chạy từng phần riêng biệt để debug.")
        
        # Tạo một test đơn giản không cần matplotlib
        print("\n=== TEST ĐơN GIẢN KHÔNG CẦN VISUALIZATION ===")
        bounds = [(-5, 5), (-5, 5)]
        simple_optimizer = RandomSearchOptimizer(sphere_function, bounds, maximize=False)
        best_sol, best_val = simple_optimizer.optimize(num_iterations=100, verbose=True)
        print(f"\nKết quả cuối cùng:")
        print(f"Best solution: {best_sol}")
        print(f"Best value: {best_val}")
        print(f"Lý thuyết (global minimum): [0, 0] với value = 0")