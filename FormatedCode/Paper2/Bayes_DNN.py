import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def himmelblau(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = x.shape[-1]
    sum1 = np.sum(x**2, axis=-1)
    sum2 = np.sum(np.cos(c * x), axis=-1)
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)

# Hàm acquisition (Upper Confidence Bound - UCB)
def acquisition_function(X, gp, beta=0.5):
    mean, std = gp.predict(X, return_std=True)
    return mean + std * beta  # Tối ưu hóa dựa trên sự chắc chắn (exploration)

# Chọn 16 điểm tiếp theo dựa trên giá trị lớn nhất của Acquisition Function
def propose_locations(acquisition, gp, bounds, n_samples=1000, n_points_mean=8, n_points_std=8, min_distance=0.5):
    dim = bounds.shape[0]

    # Sinh n_samples điểm ngẫu nhiên trong không gian
    random_points = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, dim))

    # Tính mean, std và giá trị của Acquisition Function
    mean, std = gp.predict(random_points, return_std=True)
    acquisition_values = acquisition(random_points, gp)

    # Lựa chọn các điểm dựa trên acquisition (giảm dần)
    selected_points = []

    # Lựa chọn các điểm dựa trên mean (tăng dần)
    mean_indices = np.argsort(mean)  # Sắp xếp theo mean tăng dần
    for idx in mean_indices:
        point = random_points[idx]
        if len(selected_points) == 0 or np.min(cdist(selected_points, [point])) > min_distance:
            selected_points.append(point)
        if len(selected_points) >= n_points_mean:  # Đủ số điểm dựa trên acquisition và mean
            break

    # Lựa chọn các điểm dựa trên std (tăng dần)
    std_indices = np.argsort(-std)  # Sắp xếp theo std tăng dần
    for idx in std_indices:
        point = random_points[idx]
        if len(selected_points) == 0 or np.min(cdist(selected_points, [point])) > min_distance:
            selected_points.append(point)
        if len(selected_points) >= n_points_mean + n_points_std:  # Đủ số điểm tổng cộng
            break

    return np.array(selected_points)

# Bayesian Optimization
def bayesian_optimization(objective_function, bounds, n_iter, n_init_samples=16, beta=1.96, n_points=16):
    # Tạo dữ liệu ban đầu
    X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_init_samples, bounds.shape[0]))
    Y = np.array([objective_function(x) for x in X])

    # Khởi tạo Gaussian Process
    kernel = Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    # Lưới điểm để hiển thị đồ thị
    x1 = np.linspace(bounds[0, 0], bounds[0, 1], 100)
    x2 = np.linspace(bounds[1, 0], bounds[1, 1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    grid = np.c_[X1.ravel(), X2.ravel()]

    for i in range(n_iter):
        # Huấn luyện Gaussian Process
        print("Fit GP with",len(X),"Value")
        gp.fit(X, Y)

        # Dự đoán trên toàn bộ lưới để vẽ đồ thị
        mean, std = gp.predict(grid, return_std=True)
        mean = mean.reshape(X1.shape)
        std = std.reshape(X1.shape)

        # Tìm 16 điểm tiếp theo
        x_next_points = propose_locations(acquisition_function, gp, bounds)
        y_next_points = np.array([objective_function(x) for x in x_next_points])

        print(f"Iteration {i + 1}: Points added:\n{x_next_points}")

        # Cập nhật dữ liệu
        X = np.vstack((X, x_next_points))
        Y = np.append(Y, y_next_points)

        # Trực quan hóa
        plt.figure(figsize=(16, 6))

        # Đồ thị hàm mục tiêu
        plt.subplot(1, 3, 1)
        plt.contourf(X1, X2, himmelblau(grid).reshape(X1.shape), levels=50, cmap="viridis")
        plt.colorbar()
        plt.scatter(X[:, 0], X[:, 1], color="red", label="Sampled Points")
        plt.scatter(x_next_points[:, 0], x_next_points[:, 1], color="white", edgecolor="black", s=100, label="Next Points")
        plt.title("Himmelblau's Function")
        plt.legend()

        # Đồ thị Acquisition Function
        plt.subplot(1, 3, 2)
        acquisition_vals = acquisition_function(grid, gp).reshape(X1.shape)
        plt.contourf(X1, X2, acquisition_vals, levels=50, cmap="coolwarm")
        plt.colorbar()
        plt.title("Estimate value (mean)")
        plt.scatter(x_next_points[:, 0], x_next_points[:, 1], color="white", edgecolor="black", s=100, label="Next Points")
        plt.legend()

        # Độ không chắc chắn (std)
        plt.subplot(1, 3, 3)
        plt.contourf(X1, X2, std, levels=50, cmap="coolwarm")
        plt.colorbar()
        plt.title("Uncertainty (Std)")
        plt.scatter(x_next_points[:, 0], x_next_points[:, 1], color="white", edgecolor="black", s=100, label="Next Points")
        plt.legend()

        plt.suptitle(f"Iteration {i + 1}")
        plt.show()

    return X, Y

# Định nghĩa miền tối ưu hóa và thực thi
bounds = np.array([[-5, 5], [-5, 5]])  # Giới hạn của hàm mục tiêu
n_iter = 10
X_opt, Y_opt = bayesian_optimization(himmelblau, bounds, n_iter, n_points=16)
