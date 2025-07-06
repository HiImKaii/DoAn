import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class PSO_RF_Optimizer:
    def __init__(self, num_particles=15, max_iterations=20):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = 0.9  # Inertia weight
        self.c1 = 2.0  # Cognitive parameter
        self.c2 = 2.0  # Social parameter
        
    def load_data(self, file_path, target_column=None):
        """Tải và xử lý dữ liệu"""
        df = pd.read_excel(file_path)
        
        # Xử lý missing values - chỉ fill NA cho cột số
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill NA cho cột số bằng mean
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Fill NA cho cột text bằng mode (giá trị xuất hiện nhiều nhất)
        for col in categorical_columns:
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_value, inplace=True)
        
        # Tự động xác định target column
        if target_column is None:
            target_column = df.columns[-1]
        
        # Tách features và target
        X = df.drop(labels=str(target_column), axis=1)
        y = df[target_column]
        
        # Xử lý categorical features
        le_dict = {}
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
        
        # Xác định task type
        if y.dtype == 'object' or len(y.unique()) < 20:
            self.task_type = 'classification'
            if y.dtype == 'object':
                self.target_le = LabelEncoder()
                y = self.target_le.fit_transform(y.astype(str))
        else:
            self.task_type = 'regression'
        
        # Chia dữ liệu và chuẩn hóa
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        print(f"Dữ liệu: {df.shape} | Task: {self.task_type} | Target: {target_column}")
        print(f"Features: {X.shape[1]} | Numeric: {len(numeric_columns)} | Categorical: {len(categorical_columns)}")
        
    def objective_function(self, params):
        """Hàm mục tiêu cho PSO"""
        try:
            n_estimators = max(1, int(params[0]))
            max_depth = max(1, int(params[1]))
            min_samples_split = max(2, int(params[2]))
            min_samples_leaf = max(1, int(params[3]))
            
            if self.task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth,
                    min_samples_split=min_samples_split, 
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=-1
                )
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators, 
                    max_depth=max_depth,
                    min_samples_split=min_samples_split, 
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=-1
                )
                scoring = 'neg_mean_squared_error'
            
            scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=3, scoring=scoring)
            mean_score = scores.mean()
            
            if np.isnan(mean_score) or np.isinf(mean_score):
                return float('inf')
                
            return -mean_score  # Minimize
            
        except Exception as e:
            print(f"Error in objective function with params {params}: {str(e)}")
            return float('inf')
    
    def optimize(self):
        """Chạy PSO"""
        # Định nghĩa bounds: [n_estimators, max_depth, min_samples_split, min_samples_leaf]
        bounds = [(50, 200), (1, 15), (2, 10), (1, 5)]
        
        # Khởi tạo particles
        particles = []
        for _ in range(self.num_particles):
            position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            velocity = np.random.uniform(-0.5, 0.5, len(bounds))
            particles.append({
                'position': position,
                'velocity': velocity,
                'best_pos': position.copy(),
                'best_fitness': float('inf')
            })
        
        global_best_pos = None
        global_best_fitness = float('inf')
        
        print(f"\nBắt đầu PSO: {self.num_particles} particles, {self.max_iterations} iterations")
        print("-" * 60)
        
        for iteration in range(self.max_iterations):
            # Đánh giá fitness
            for particle in particles:
                fitness = self.objective_function(particle['position'])
                
                # Cập nhật best của particle
                if fitness < particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_pos'] = particle['position'].copy()
                
                # Cập nhật global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_pos = particle['position'].copy()
            
            # Cập nhật velocity và position
            for particle in particles:
                w = self.w * (self.max_iterations - iteration) / self.max_iterations
                r1, r2 = np.random.random(), np.random.random()
                
                cognitive = self.c1 * r1 * (particle['best_pos'] - particle['position'])
                social = self.c2 * r2 * (global_best_pos - particle['position'])
                
                particle['velocity'] = w * particle['velocity'] + cognitive + social
                particle['velocity'] = np.clip(particle['velocity'], -1, 1)
                particle['position'] += particle['velocity']
                
                # Giới hạn position
                for i, (min_val, max_val) in enumerate(bounds):
                    particle['position'][i] = np.clip(particle['position'][i], min_val, max_val)
            
            # In kết quả mỗi vòng lặp
            score = -global_best_fitness if self.task_type == 'classification' else np.sqrt(-global_best_fitness)
            metric = "Accuracy" if self.task_type == 'classification' else "RMSE"
            print(f"Iteration {iteration+1:2d}: Best {metric} = {score:.4f}")
        
        # Kết quả tốt nhất
        if global_best_pos is None:
            print("Optimization failed - no valid solution found")
            return None, None
            
        best_params = {
            'n_estimators': int(global_best_pos[0]),
            'max_depth': int(global_best_pos[1]),
            'min_samples_split': int(global_best_pos[2]),
            'min_samples_leaf': int(global_best_pos[3])
        }
        
        print("-" * 60)
        print("KẾT QUẢ TỐI ƯU:")
        print(f"Best parameters: {best_params}")
        
        # Huấn luyện model cuối cùng
        if self.task_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                n_jobs=-1
            )
        
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        
        if self.task_type == 'classification':
            test_score = accuracy_score(self.y_test, y_pred)
            print(f"Test Accuracy: {test_score:.4f}")
        else:
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            print(f"Test RMSE: {np.sqrt(mse):.4f}")
            print(f"Test R²: {r2:.4f}")
        
        return best_params, model

# Sử dụng
if __name__ == "__main__":
    optimizer = PSO_RF_Optimizer(num_particles=15, max_iterations=20)
    
    file_path = r"C:\Users\Admin\Downloads\prj\src\flood_training.xlsx"
    
    try:
        optimizer.load_data(file_path)
        best_params, best_model = optimizer.optimize()
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_path}")
        print("Vui lòng kiểm tra lại đường dẫn file.")
    except Exception as e:
        print(f"Lỗi: {e}")
        print("Vui lòng kiểm tra lại dữ liệu và thử lại.")