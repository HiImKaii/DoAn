import numpy as np
import random
import matplotlib.pyplot as plt

class BirdSwarmAlgorithm:
    def __init__(self, N=30, M=100, FQ=10, P=0.8, C=1.5, S=1.0, a1=2.0, a2=2.0, FL=0.5, 
                 dim=2, bounds=(-10, 10), objective_function=None):
        """
        Khởi tạo thuật toán Bird Swarm Algorithm
        
        Args:
            N: số lượng cá thể (chim) trong quần thể
            M: số lần lặp tối đa
            FQ: tần suất hành vi bay của chim
            P: xác suất tìm kiếm thức ăn
            C, S, a1, a2, FL: các tham số hằng số
            dim: số chiều của bài toán
            bounds: giới hạn không gian tìm kiếm
            objective_function: hàm mục tiêu cần tối ưu
        """
        self.N = N
        self.M = M
        self.FQ = FQ
        self.P = P
        self.C = C
        self.S = S
        self.a1 = a1
        self.a2 = a2
        self.FL = FL
        self.dim = dim
        self.bounds = bounds
        self.objective_function = objective_function or self.sphere_function
        
        # Khởi tạo quần thể
        self.population = np.random.uniform(bounds[0], bounds[1], (N, dim))
        self.fitness = np.zeros(N)
        self.best_position = np.zeros(dim)  # Khởi tạo với mảng zeros thay vì None
        self.best_fitness = float('inf')
        self.fitness_history = []
        
    def sphere_function(self, x):
        """Hàm mục tiêu mặc định - Sphere function"""
        return np.sum(x**2)
    
    def evaluate_fitness(self):
        """Đánh giá fitness cho tất cả cá thể"""
        for i in range(self.N):
            self.fitness[i] = self.objective_function(self.population[i])
            
        # Tìm cá thể tốt nhất
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_position = self.population[best_idx].copy()
    
    def foraging_behavior(self, i):
        """Hành vi tìm kiếm thức ăn (Equation 1)"""
        # Tìm mean position của swarm
        mean_position = np.mean(self.population, axis=0)
        
        # Cập nhật vị trí
        r1 = np.random.random()
        r2 = np.random.random()
        
        new_position = (self.population[i] + 
                       self.C * r1 * (self.best_position - self.population[i]) +
                       self.S * r2 * (mean_position - self.population[i]))
        
        return new_position
    
    def vigilance_behavior(self, i):
        """Hành vi cảnh giác (Equation 2)"""
        # Chọn ngẫu nhiên một cá thể khác
        k = random.choice([j for j in range(self.N) if j != i])
        
        # Tính toán vị trí mới
        A = np.random.uniform(-self.a1, self.a1, self.dim)
        
        if self.fitness[i] < self.fitness[k]:
            new_position = self.population[i] + A * (self.population[i] - self.population[k])
        else:
            new_position = self.population[i] + A * (self.population[k] - self.population[i])
            
        return new_position
    
    def producing_behavior(self, i):
        """Hành vi sản xuất (Equation 5)"""
        r1 = np.random.random()
        r2 = np.random.random()
        
        new_position = (self.population[i] + 
                       r1 * (self.best_position - self.population[i]) +
                       r2 * (np.random.uniform(self.bounds[0], self.bounds[1], self.dim) - self.population[i]))
        
        return new_position
    
    def scrounging_behavior(self, i):
        """Hành vi ăn theo (Equation 6)"""
        # Chọn ngẫu nhiên một producer
        producers_count = int(self.N * 0.8)  # 80% là producers
        producer_idx = random.randint(0, producers_count - 1)
        
        r3 = np.random.random()
        new_position = (self.population[i] + 
                       r3 * (self.population[producer_idx] - self.population[i]))
        
        return new_position
    
    def bound_check(self, position):
        """Kiểm tra và điều chỉnh vị trí trong giới hạn"""
        return np.clip(position, self.bounds[0], self.bounds[1])
    
    def optimize(self):
        """Thực hiện tối ưu hóa"""
        # Đánh giá fitness ban đầu và tìm best solution ngay từ đầu
        self.evaluate_fitness()
        
        for t in range(self.M):
            new_population = self.population.copy()
            
            if t % self.FQ == 0:
                # Hành vi bay (foraging hoặc vigilance)
                for i in range(self.N):
                    if random.random() < self.P:
                        # Birds forage for food
                        new_position = self.foraging_behavior(i)
                    else:
                        # Birds keep vigilance
                        new_position = self.vigilance_behavior(i)
                    
                    new_population[i] = self.bound_check(new_position)
            else:
                # Chia swarm thành producers và scroungers
                producers_count = int(self.N * 0.8)  # 80% là producers
                
                for i in range(self.N):
                    if i < producers_count:
                        # Producer
                        new_position = self.producing_behavior(i)
                    else:
                        # Scrounger
                        new_position = self.scrounging_behavior(i)
                    
                    new_population[i] = self.bound_check(new_position)
            
            # Đánh giá các giải pháp mới
            new_fitness = np.zeros(self.N)
            for i in range(self.N):
                new_fitness[i] = self.objective_function(new_population[i])
            
            # Cập nhật nếu tốt hơn
            for i in range(self.N):
                if new_fitness[i] < self.fitness[i]:
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]
            
            # Tìm giải pháp tốt nhất hiện tại
            self.evaluate_fitness()
            self.fitness_history.append(self.best_fitness)
            
            # In tiến trình
            if t % 10 == 0:
                print(f"Iteration {t}: Best fitness = {self.best_fitness:.6f}")
        
        return self.best_position, self.best_fitness
    
    def plot_convergence(self):
        """Vẽ đồ thị hội tụ"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title('BSA Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()

# Ví dụ sử dụng
if __name__ == "__main__":
    # Định nghĩa hàm mục tiêu (ví dụ: Sphere function)
    def sphere_function(x):
        return np.sum(x**2)
    
    # Định nghĩa hàm mục tiêu phức tạp hơn (ví dụ: Rastrigin function)
    def rastrigin_function(x):
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    # Khởi tạo và chạy BSA
    print("=== Bird Swarm Algorithm ===")
    
    # Test với Sphere function
    print("\n1. Testing with Sphere function:")
    bsa1 = BirdSwarmAlgorithm(
        N=30,           # Số lượng chim
        M=100,          # Số lần lặp
        FQ=10,          # Tần suất bay
        P=0.8,          # Xác suất tìm kiếm thức ăn
        C=1.5,          # Tham số C
        S=1.0,          # Tham số S
        a1=2.0,         # Tham số a1
        a2=2.0,         # Tham số a2
        FL=0.5,         # Tham số FL
        dim=10,         # Số chiều
        bounds=(-10, 10),
        objective_function=sphere_function
    )
    
    best_pos1, best_fit1 = bsa1.optimize()
    print(f"Best position: {best_pos1}")
    print(f"Best fitness: {best_fit1}")
    
    # Test với Rastrigin function
    print("\n2. Testing with Rastrigin function:")
    bsa2 = BirdSwarmAlgorithm(
        N=30,
        M=100,
        FQ=10,
        P=0.8,
        C=1.5,
        S=1.0,
        a1=2.0,
        a2=2.0,
        FL=0.5,
        dim=5,
        bounds=(-5.12, 5.12),
        objective_function=rastrigin_function
    )
    
    best_pos2, best_fit2 = bsa2.optimize()
    print(f"Best position: {best_pos2}")
    print(f"Best fitness: {best_fit2}")
    
    # Vẽ đồ thị hội tụ
    print("\n3. Plotting convergence curves...")
    bsa1.plot_convergence()