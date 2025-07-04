import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class MothSearchAlgorithm:
    def __init__(self, 
                 objective_function: Callable[[np.ndarray], float],
                 dimension: int,
                 bounds: Tuple[float, float],
                 population_size: int = 30,
                 max_generations: int = 100,
                 max_walk_step: float = 1.0,
                 beta: float = 1.5,
                 acceleration_factor: float = 0.1) -> None:
        """
        Khởi tạo thuật toán Moth Search Algorithm
        
        Args:
            objective_function: Hàm mục tiêu cần tối ưu
            dimension: Số chiều của không gian tìm kiếm
            bounds: Giới hạn của không gian tìm kiếm (min, max)
            population_size: Kích thước quần thể (NP)
            max_generations: Số thế hệ tối đa (MaxGen)
            max_walk_step: Bước đi tối đa (Smax)
            beta: Chỉ số β
            acceleration_factor: Hệ số gia tốc
        """
        self.objective_function = objective_function
        self.dimension = dimension
        self.bounds = bounds
        self.population_size = population_size
        self.max_generations = max_generations
        self.max_walk_step = max_walk_step
        self.beta = beta
        self.acceleration_factor = acceleration_factor
        
        # Khởi tạo quần thể
        self.population: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None
        self.best_position: Optional[np.ndarray] = None
        self.best_fitness: float = float('inf')
        self.convergence_curve: List[float] = []
        
    def initialize_population(self) -> None:
        """Bước 1: Khởi tạo quần thể ngẫu nhiên"""
        self.population = np.random.uniform(
            self.bounds[0], 
            self.bounds[1], 
            (self.population_size, self.dimension)
        )
        
    def evaluate_fitness(self) -> None:
        """Bước 2: Đánh giá fitness cho từng cá thể"""
        if self.population is None:
            raise ValueError("Population chưa được khởi tạo")
            
        self.fitness = np.array([self.objective_function(individual) 
                               for individual in self.population])
        
        # Cập nhật giải pháp tốt nhất
        min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.best_fitness:
            self.best_fitness = self.fitness[min_idx]
            self.best_position = self.population[min_idx].copy()
    
    def levy_flight(self, dimension: int) -> np.ndarray:
        """
        Tạo vector Lévy flight
        Lévy flight được sử dụng trong phương trình (5)
        """
        # Tính toán sigma cho phân phối Lévy
        try:
            from math import gamma, sin, pi
            num = gamma(1 + self.beta) * sin(pi * self.beta / 2)
            den = gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
            sigma = (num / den) ** (1 / self.beta)
        except (ValueError, ZeroDivisionError, OverflowError):
            # Fallback nếu có lỗi trong tính toán gamma
            sigma = 1.0
        
        # Tạo các số ngẫu nhiên
        u = np.random.normal(0, sigma, dimension)
        v = np.random.normal(0, 1, dimension)
        
        # Tính toán bước Lévy với xử lý chia cho 0
        v_abs = np.abs(v)
        v_abs[v_abs < 1e-10] = 1e-10  # Tránh chia cho 0
        step = u / (v_abs ** (1 / self.beta))
        
        # Giới hạn step để tránh giá trị quá lớn
        step = np.clip(step, -10, 10)
        
        return step
    
    def update_position_eq5(self, moth_position: np.ndarray, t: int) -> np.ndarray:
        """
        Phương trình (5): Cập nhật vị trí bằng Lévy flight
        Được sử dụng khi r > 0.5
        """
        levy_step = self.levy_flight(self.dimension)
        scale_factor = self.max_walk_step / (t + 1)  # Giảm dần theo thế hệ
        
        new_position = moth_position + scale_factor * levy_step
        
        # Đảm bảo vị trí nằm trong giới hạn
        new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
        
        return new_position
    
    def update_position_eq6(self, moth_position: np.ndarray, t: int) -> np.ndarray:
        """
        Phương trình (6): Cập nhật vị trí hướng về nguồn sáng (best position)
        Được sử dụng khi r <= 0.5
        """
        if self.best_position is None:
            return moth_position
        
        # Tính toán hướng về nguồn sáng tốt nhất
        direction = self.best_position - moth_position
        
        # Thêm yếu tố ngẫu nhiên
        random_factor = np.random.uniform(-1, 1, self.dimension)
        
        # Hệ số giảm dần theo thế hệ
        scale_factor = self.acceleration_factor * (1 - t / self.max_generations)
        
        new_position = moth_position + scale_factor * direction + 0.01 * random_factor
        
        # Đảm bảo vị trí nằm trong giới hạn
        new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
        
        return new_position
    
    def update_population(self, t: int) -> None:
        """Cập nhật vị trí của toàn bộ quần thể"""
        if self.population is None:
            raise ValueError("Population chưa được khởi tạo")
            
        new_population = np.zeros_like(self.population)
        
        for i in range(self.population_size):
            r = random.random()
            
            if r > 0.5:
                # Sử dụng phương trình (5) - Lévy flight
                new_population[i] = self.update_position_eq5(self.population[i], t)
            else:
                # Sử dụng phương trình (6) - Hướng về nguồn sáng
                new_population[i] = self.update_position_eq6(self.population[i], t)
        
        self.population = new_population
    
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Thực hiện thuật toán tối ưu hóa
        
        Returns:
            Tuple chứa: (best_position, best_fitness, convergence_curve)
        """
        print("Bắt đầu thuật toán Moth Search Algorithm...")
        
        # Bước 1: Khởi tạo
        t = 0  # Bắt đầu từ 0 thay vì 1
        self.initialize_population()
        
        # Bước 2: Đánh giá fitness ban đầu
        self.evaluate_fitness()
        self.convergence_curve.append(self.best_fitness)
        
        print(f"Thế hệ {t + 1}: Best fitness = {self.best_fitness:.6f}")
        
        # Bước 3: Vòng lặp chính
        while t < self.max_generations:
            # Cập nhật vị trí quần thể
            self.update_population(t)
            
            # Đánh giá fitness mới
            self.evaluate_fitness()
            
            # Lưu kết quả convergence
            self.convergence_curve.append(self.best_fitness)
            
            # In tiến trình
            if (t + 1) % 10 == 0 or t == self.max_generations - 1:
                print(f"Thế hệ {t + 1}: Best fitness = {self.best_fitness:.6f}")
            
            t += 1
        
        print(f"Thuật toán hoàn thành!")
        print(f"Giải pháp tốt nhất: {self.best_position}")
        print(f"Giá trị tốt nhất: {self.best_fitness:.6f}")
        
        return self.best_position, self.best_fitness, self.convergence_curve
    
    def plot_convergence(self):
        """Vẽ đồ thị convergence"""
        if not self.convergence_curve:
            print("Chưa có dữ liệu convergence để vẽ đồ thị.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.convergence_curve)), self.convergence_curve, 'b-', linewidth=2)
        plt.title('Moth Search Algorithm - Convergence Curve')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Sử dụng log scale cho trục y
        plt.show()


# Các hàm test benchmark
def sphere_function(x):
    """Hàm Sphere - f(x) = sum(x_i^2)"""
    return np.sum(x**2)

def rosenbrock_function(x):
    """Hàm Rosenbrock"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin_function(x):
    """Hàm Rastrigin"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley_function(x):
    """Hàm Ackley"""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


# Ví dụ sử dụng
if __name__ == "__main__":
    # Thiết lập random seed để có kết quả reproducible
    np.random.seed(42)
    random.seed(42)
    
    # Test với hàm Sphere
    print("=== Test với hàm Sphere ===")
    msa_sphere = MothSearchAlgorithm(
        objective_function=sphere_function,
        dimension=10,
        bounds=(-5, 5),
        population_size=30,
        max_generations=100,
        max_walk_step=1.0,
        beta=1.5,
        acceleration_factor=0.1
    )
    
    try:
        best_pos, best_fit, convergence = msa_sphere.optimize()
        msa_sphere.plot_convergence()
    except Exception as e:
        print(f"Lỗi trong quá trình tối ưu hóa: {e}")
    
    print("\n" + "="*50)
    
    # Test với hàm Rosenbrock
    print("=== Test với hàm Rosenbrock ===")
    msa_rosenbrock = MothSearchAlgorithm(
        objective_function=rosenbrock_function,
        dimension=5,
        bounds=(-2, 2),
        population_size=40,
        max_generations=200,
        max_walk_step=0.5,
        beta=1.5,
        acceleration_factor=0.2
    )
    
    try:
        best_pos, best_fit, convergence = msa_rosenbrock.optimize()
        msa_rosenbrock.plot_convergence()
    except Exception as e:
        print(f"Lỗi trong quá trình tối ưu hóa: {e}")
    
    print("\n=== Test với hàm Rastrigin ===")
    msa_rastrigin = MothSearchAlgorithm(
        objective_function=rastrigin_function,
        dimension=8,
        bounds=(-5.12, 5.12),
        population_size=50,
        max_generations=150,
        max_walk_step=0.8,
        beta=1.2,
        acceleration_factor=0.15
    )
    
    try:
        best_pos, best_fit, convergence = msa_rastrigin.optimize()
        msa_rastrigin.plot_convergence()
    except Exception as e:
        print(f"Lỗi trong quá trình tối ưu hóa: {e}")