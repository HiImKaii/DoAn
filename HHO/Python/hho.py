import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Cấu hình font để hiển thị tiếng Việt
rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class HarrisHawksOptimization:
    def __init__(self, pop_size=30, max_iter=500, dim=2, lb=-10, ub=10):
        """
        Khởi tạo thuật toán Harris Hawks Optimization
        
        Args:
            pop_size: Kích thước quần thể
            max_iter: Số lần lặp tối đa
            dim: Số chiều của bài toán
            lb: Cận dưới
            ub: Cận trên
        """
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = dim
        self.lb = lb
        self.ub = ub
        
        # Khởi tạo quần thể ngẫu nhiên
        self.population = np.random.uniform(lb, ub, (pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_position = np.zeros(dim)
        self.best_fitness = float('inf')
        self.convergence_curve = []
    
    def objective_function(self, x):
        """
        Hàm mục tiêu - Sphere function
        f(x) = sum(x_i^2)
        """
        return np.sum(x**2)
    
    def evaluate_fitness(self):
        """Đánh giá fitness cho toàn bộ quần thể"""
        for i in range(self.pop_size):
            self.fitness[i] = self.objective_function(self.population[i])
            
            # Cập nhật vị trí tốt nhất (rabbit)
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.population[i].copy()
    
    def levy_flight(self, dim):
        """Tính toán Levy flight"""
        beta = 1.5
        num = np.random.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = np.random.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma_u = (num / den) ** (1 / beta)
        
        u = np.random.normal(0, sigma_u, dim)
        v = np.random.normal(0, 1, dim)
        
        return u / (np.abs(v) ** (1 / beta))
    
    def optimize(self):
        """Thuật toán tối ưu hóa HHO chính"""
        
        # Đánh giá fitness ban đầu
        self.evaluate_fitness()
        
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # Cập nhật năng lượng ban đầu E0 và jump strength J
                E0 = 2 * np.random.rand() - 1  # [-1, 1]
                J = 2 * (1 - np.random.rand())  # [0, 2]
                
                # Cập nhật E sử dụng phương trình (3)
                E = 2 * E0 * (1 - (t / self.max_iter))
                
                # Tham số ngẫu nhiên
                r = np.random.rand()
                
                if abs(E) >= 1:
                    # Pha thăm dò (Exploration phase)
                    rand_hawk_idx = np.random.randint(0, self.pop_size)
                    
                    if np.random.rand() < 0.5:
                        # Phương trình (1) - Random perching
                        X1 = self.population[rand_hawk_idx] - np.random.rand() * np.abs(
                            self.population[rand_hawk_idx] - 2 * np.random.rand() * self.population[i])
                        X2 = (self.best_position - np.mean(self.population, axis=0)) - \
                             np.random.rand() * (self.lb + np.random.rand() * (self.ub - self.lb))
                        
                        if self.objective_function(X1) < self.objective_function(X2):
                            self.population[i] = X1
                        else:
                            self.population[i] = X2
                    else:
                        # Random perching based on two random hawks
                        X1 = self.best_position - np.random.rand() * np.abs(
                            self.best_position - 2 * np.random.rand() * self.population[i])
                        X2 = self.best_position - np.random.rand() * np.abs(
                            self.best_position - 2 * np.random.rand() * self.population[rand_hawk_idx])
                        
                        if self.objective_function(X1) < self.objective_function(X2):
                            self.population[i] = X1
                        else:
                            self.population[i] = X2
                
                else:
                    # Pha khai thác (Exploitation phase)
                    if r >= 0.5 and abs(E) >= 0.5:
                        # Soft besiege - Phương trình (4)
                        delta_X = self.best_position - self.population[i]
                        self.population[i] = delta_X - E * np.abs(J * self.best_position - self.population[i])
                    
                    elif r >= 0.5 and abs(E) < 0.5:
                        # Hard besiege - Phương trình (6)
                        self.population[i] = self.best_position - E * np.abs(delta_X)
                    
                    elif r < 0.5 and abs(E) >= 0.5:
                        # Soft besiege with progressive rapid dives - Phương trình (10)
                        delta_X = self.best_position - self.population[i]
                        S = np.random.rand(self.dim) * self.levy_flight(self.dim)
                        
                        Y = self.best_position - E * np.abs(J * self.best_position - self.population[i])
                        Z = Y + S
                        
                        if self.objective_function(Y) < self.objective_function(self.population[i]):
                            self.population[i] = Y
                        elif self.objective_function(Z) < self.objective_function(self.population[i]):
                            self.population[i] = Z
                    
                    else:
                        # Hard besiege with progressive rapid dives - Phương trình (11)
                        delta_X = self.best_position - self.population[i]
                        S = np.random.rand(self.dim) * self.levy_flight(self.dim)
                        
                        Y = self.best_position - E * np.abs(J * self.best_position - self.mean_position())
                        Z = Y + S
                        
                        if self.objective_function(Y) < self.objective_function(self.population[i]):
                            self.population[i] = Y
                        elif self.objective_function(Z) < self.objective_function(self.population[i]):
                            self.population[i] = Z
                
                # Giới hạn biên
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
            
            # Đánh giá lại fitness
            self.evaluate_fitness()
            
            # Lưu kết quả hội tụ
            self.convergence_curve.append(self.best_fitness)
            
            # In kết quả mỗi 50 lần lặp
            if (t + 1) % 50 == 0:
                print(f"Lần lặp {t+1}: Best fitness = {self.best_fitness:.6f}")
        
        return self.best_position, self.best_fitness
    
    def mean_position(self):
        """Tính vị trí trung bình của quần thể"""
        return np.mean(self.population, axis=0)
    
    def plot_convergence(self):
        """Vẽ đồ thị hội tụ"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, 'b-', linewidth=2)
        plt.xlabel('Số lần lặp')
        plt.ylabel('Giá trị fitness tốt nhất')
        plt.title('Đồ thị hội tụ của thuật toán Harris Hawks Optimization')
        plt.grid(True)
        plt.yscale('log')  # Sử dụng thang log để dễ quan sát
        plt.show()
    
    def plot_population_evolution(self):
        """Vẽ sự tiến hóa của quần thể (chỉ với bài toán 2D)"""
        if self.dim == 2:
            plt.figure(figsize=(10, 8))
            
            # Vẽ contour của hàm mục tiêu
            x = np.linspace(self.lb, self.ub, 100)
            y = np.linspace(self.lb, self.ub, 100)
            X, Y = np.meshgrid(x, y)
            Z = X**2 + Y**2
            
            plt.contour(X, Y, Z, levels=20, alpha=0.5)
            plt.colorbar(label='Giá trị hàm mục tiêu')
            
            # Vẽ quần thể
            plt.scatter(self.population[:, 0], self.population[:, 1], 
                       c='blue', alpha=0.6, s=30, label='Hawks')
            plt.scatter(self.best_position[0], self.best_position[1], 
                       c='red', s=100, marker='*', label='Rabbit (Best)')
            
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('Phân bố quần thể Harris Hawks')
            plt.legend()
            plt.grid(True)
            plt.show()


def main():
    """Hàm chính chạy thuật toán"""
    print("=" * 60)
    print("THUẬT TOÁN HARRIS HAWKS OPTIMIZATION (HHO)")
    print("=" * 60)
    
    # Khởi tạo thuật toán
    hho = HarrisHawksOptimization(
        pop_size=30,
        max_iter=500,
        dim=2,
        lb=-10,
        ub=10
    )
    
    print(f"Tham số thuật toán:")
    print(f"- Kích thước quần thể: {hho.pop_size}")
    print(f"- Số lần lặp tối đa: {hho.max_iter}")
    print(f"- Số chiều: {hho.dim}")
    print(f"- Miền tìm kiếm: [{hho.lb}, {hho.ub}]")
    print(f"- Hàm mục tiêu: Sphere function (f(x) = sum(x_i^2))")
    print("\nBắt đầu tối ưu hóa...\n")
    
    # Chạy thuật toán
    best_position, best_fitness = hho.optimize()
    
    # In kết quả
    print("\n" + "=" * 60)
    print("KẾT QUẢ CUỐI CÙNG:")
    print("=" * 60)
    print(f"Vị trí tốt nhất (Rabbit): {best_position}")
    print(f"Giá trị fitness tốt nhất: {best_fitness:.10f}")
    print(f"Số lần lặp: {hho.max_iter}")
    
    # Vẽ đồ thị hội tụ
    hho.plot_convergence()
    
    # Vẽ phân bố quần thể (nếu là bài toán 2D)
    if hho.dim == 2:
        hho.plot_population_evolution()


if __name__ == "__main__":
    main()