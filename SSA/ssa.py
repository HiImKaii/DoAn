import numpy as np
import random
import matplotlib.pyplot as plt

class SparrowSearchAlgorithm:
    def __init__(self, N=30, max_iter=100, pd=0.1, sd=0.1, ST=0.8, 
                 dim=2, bounds=(-10, 10), objective_function=None):
        """
        Khởi tạo thuật toán Sparrow Search Algorithm
        
        Args:
            N: số lượng chim sẻ trong quần thể
            max_iter: số lần lặp tối đa
            pd: tỷ lệ producers (kẻ khám phá) trong quần thể
            sd: tỷ lệ scroungers nhận biết nguy hiểm
            ST: ngưỡng an toàn (Safety Threshold)
            dim: số chiều của bài toán
            bounds: giới hạn không gian tìm kiếm (tuple: (min, max))
            objective_function: hàm mục tiêu cần tối ưu
        """
        self.N = N
        self.max_iter = max_iter
        self.pd = pd  # Tỷ lệ producers (10%)
        self.sd = sd  # Tỷ lệ scroungers nhận biết nguy hiểm (10%)
        self.ST = ST  # Safety threshold
        self.dim = dim
        self.bounds = bounds
        self.objective_function = objective_function or self.sphere_function
        
        # Số lượng từng loại chim sẻ
        self.num_producers = int(self.pd * N)  # 10% producers
        self.num_scroungers = int(0.8 * N)     # 80% scroungers
        self.num_scouts = int(self.sd * N)     # 10% scouts
        
        # Khởi tạo quần thể
        self.population = np.random.uniform(bounds[0], bounds[1], (N, dim))
        self.fitness = np.zeros(N)
        self.best_position = np.zeros(dim)
        self.best_fitness = float('inf')
        self.worst_fitness = float('-inf')
        self.fitness_history = []
        
        # Sắp xếp quần thể theo fitness
        self.evaluate_fitness()
        self.sort_population()
        
    def sphere_function(self, x):
        """Hàm mục tiêu mặc định - Sphere function"""
        return np.sum(x**2)
    
    def evaluate_fitness(self):
        """Đánh giá fitness cho tất cả cá thể"""
        for i in range(self.N):
            self.fitness[i] = self.objective_function(self.population[i])
        
        # Tìm cá thể tốt nhất và tệ nhất
        best_idx = np.argmin(self.fitness)
        worst_idx = np.argmax(self.fitness)
        
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_position = self.population[best_idx].copy()
        
        self.worst_fitness = self.fitness[worst_idx]
    
    def sort_population(self):
        """Sắp xếp quần thể theo fitness (tốt nhất đến tệ nhất)"""
        sorted_indices = np.argsort(self.fitness)
        self.population = self.population[sorted_indices]
        self.fitness = self.fitness[sorted_indices]
    
    def bound_check(self, position):
        """Kiểm tra và điều chỉnh vị trí trong giới hạn"""
        return np.clip(position, self.bounds[0], self.bounds[1])
    
    def update_producers(self, iter_count):
        """Cập nhật vị trí của producers (kẻ tìm kiếm thức ăn)"""
        R2 = np.random.random()  # Cảnh báo giá trị [0,1]
        
        for i in range(self.num_producers):
            if R2 < self.ST:
                # Không có kẻ thù, tìm kiếm thức ăn bình thường
                r1 = np.random.random()
                self.population[i] = (self.population[i] * 
                                    np.exp(-i / (r1 * self.max_iter)))
            else:
                # Có kẻ thù, bay đến vị trí an toàn
                Q = np.random.normal(0, 1)
                L = np.ones(self.dim)  # Vector đơn vị
                self.population[i] = (np.random.uniform(self.bounds[0], self.bounds[1], self.dim) + 
                                    Q * L)
            
            # Kiểm tra giới hạn
            self.population[i] = self.bound_check(self.population[i])
    
    def update_scroungers(self):
        """Cập nhật vị trí của scroungers (kẻ ăn theo)"""
        for i in range(self.num_producers, self.N):
            if i < self.N // 2:
                # Nửa đầu của scroungers
                Q = np.random.normal(0, 1)
                A = np.random.choice([-1, 1], size=self.dim) * np.random.randint(1, 3, size=self.dim)
                
                self.population[i] = (Q * np.exp((self.population[-1] - self.population[i]) / 
                                               (i**2)))
            else:
                # Nửa sau của scroungers
                A = np.random.choice([-1, 1], size=self.dim) * np.random.randint(1, 3, size=self.dim)
                L = np.ones(self.dim)  # Vector đơn vị cho scroungers
                
                self.population[i] = (self.population[0] + 
                                    np.abs(self.population[i] - self.population[0]) * A * L)
            
            # Kiểm tra giới hạn
            self.population[i] = self.bound_check(self.population[i])
    
    def update_scouts(self):
        """Cập nhật vị trí của scouts (kẻ trinh sát cảnh báo nguy hiểm)"""
        # Chọn ngẫu nhiên scouts từ quần thể
        scout_indices = np.random.choice(self.N, self.num_scouts, replace=False)
        
        for i in scout_indices:
            if self.fitness[i] > self.best_fitness:
                # Chim sẻ ở vị trí nguy hiểm
                beta = 2 * np.random.random() - 1  # [-1, 1]
                K = np.random.choice([-1, 1]) * 2 * np.random.random()
                
                self.population[i] = (self.best_position + 
                                    beta * np.abs(self.population[i] - self.best_position))
            else:
                # Chim sẻ ở vị trí an toàn, tìm kiếm thức ăn xung quanh
                e = np.random.normal(0, 1, self.dim)
                self.population[i] = (self.population[i] + 
                                    np.random.random() * e)
            
            # Kiểm tra giới hạn  
            self.population[i] = self.bound_check(self.population[i])
    
    def optimize(self):
        """Thực hiện tối ưu hóa"""
        for iter_count in range(self.max_iter):
            # 1. Cập nhật producers
            self.update_producers(iter_count)
            
            # 2. Cập nhật scroungers
            self.update_scroungers()
            
            # 3. Cập nhật scouts
            self.update_scouts()
            
            # 4. Đánh giá fitness và sắp xếp
            self.evaluate_fitness()
            self.sort_population()
            
            # Lưu lại fitness tốt nhất
            self.fitness_history.append(self.best_fitness)
            
            # In tiến trình
            if iter_count % 10 == 0:
                print(f"Iteration {iter_count}: Best fitness = {self.best_fitness:.6f}")
        
        return self.best_position, self.best_fitness
    
    def plot_convergence(self):
        """Vẽ đồ thị hội tụ"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, linewidth=2)
        plt.title('SSA Convergence Curve', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Fitness', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Một số hàm test phổ biến
def sphere_function(x):
    """Sphere function - minimum at (0,0,...,0)"""
    return np.sum(x**2)

def rastrigin_function(x):
    """Rastrigin function - multimodal function"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    """Rosenbrock function - banana function"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley_function(x):
    """Ackley function - multimodal function"""
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - 
            np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)

# Ví dụ sử dụng
if __name__ == "__main__":
    print("=== Sparrow Search Algorithm (SSA) ===")
    
    # Test với các hàm mục tiêu khác nhau
    test_functions = [
        ("Sphere Function", sphere_function, (-10, 10), 10),
        ("Rastrigin Function", rastrigin_function, (-5.12, 5.12), 5),
        ("Rosenbrock Function", rosenbrock_function, (-5, 5), 5),
        ("Ackley Function", ackley_function, (-32, 32), 5)
    ]
    
    for name, func, bounds, dim in test_functions:
        print(f"\n{'='*50}")
        print(f"Testing with {name}")
        print(f"Dimension: {dim}, Bounds: {bounds}")
        print(f"{'='*50}")
        
        # Khởi tạo SSA
        ssa = SparrowSearchAlgorithm(
            N=30,               # Số lượng chim sẻ
            max_iter=100,       # Số lần lặp tối đa
            pd=0.1,             # Tỷ lệ producers (10%)
            sd=0.1,             # Tỷ lệ scouts (10%)
            ST=0.8,             # Safety threshold
            dim=dim,            # Số chiều
            bounds=bounds,      # Giới hạn
            objective_function=func
        )
        
        # Chạy tối ưu hóa
        best_pos, best_fit = ssa.optimize()
        
        print(f"\nResults:")
        print(f"Best position: {best_pos}")
        print(f"Best fitness: {best_fit:.8f}")
        
        # Vẽ đồ thị hội tụ cho hàm đầu tiên
        if name == "Sphere Function":
            print("\nPlotting convergence curve...")
            ssa.plot_convergence()
    
    print(f"\n{'='*50}")
    print("All tests completed!")