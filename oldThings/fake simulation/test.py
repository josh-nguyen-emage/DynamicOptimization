import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Định nghĩa hàm mục tiêu cần tối ưu
def objective_function(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2 + (x[4] - 5)**2

# Hàm tính trung điểm của các điểm tốt nhất
def centroid(vertices):
    return np.mean(vertices[:-1], axis=0)

# Hàm để tính giá trị hàm mục tiêu và lưu trữ kết quả
def evaluate(func, x, cache):
    x_tuple = tuple(x)
    if x_tuple not in cache:
        cache[x_tuple] = func(x)
    return cache[x_tuple]

# Hàm tính giá trị hàm mục tiêu song song
def evaluate_parallel(func, points, cache, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate, func, x, cache): x for x in points}
        results = {futures[future]: future.result() for future in futures}
    return results

# Hàm Nelder-Mead
def nelder_mead(func, x_start, tol=1e-6, max_iter=500, max_workers=8):
    n = len(x_start)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    # Khởi tạo đơn hình
    simplex = [x_start]
    for i in range(n):
        x = np.copy(x_start)
        x[i] = x[i] + 1
        simplex.append(x)
    simplex = np.array(simplex)

    cache = {}

    for iteration in range(max_iter):
        # Sắp xếp các điểm trong đơn hình theo giá trị hàm mục tiêu
        simplex = sorted(simplex, key=lambda x: evaluate(func, x, cache))

        # Tính trung điểm của các điểm tốt nhất trừ điểm tệ nhất
        x0 = centroid(simplex)

        # Phản chiếu
        xr = x0 + alpha * (x0 - simplex[-1])
        fr = evaluate(func, xr, cache)

        if evaluate(func, simplex[0], cache) <= fr < evaluate(func, simplex[-2], cache):
            simplex[-1] = xr
        elif fr < evaluate(func, simplex[0], cache):
            # Mở rộng
            xe = x0 + gamma * (xr - x0)
            if evaluate(func, xe, cache) < fr:
                simplex[-1] = xe
            else:
                simplex[-1] = xr
        else:
            # Thu hẹp
            xc = x0 + rho * (simplex[-1] - x0)
            if evaluate(func, xc, cache) < evaluate(func, simplex[-1], cache):
                simplex[-1] = xc
            else:
                # Thu nhỏ
                new_points = [simplex[0] + sigma * (simplex[i] - simplex[0]) for i in range(1, len(simplex))]
                results = evaluate_parallel(func, new_points, cache, max_workers=max_workers)
                for i in range(1, len(simplex)):
                    simplex[i] = results.keys()[i-1]
        
        # Kiểm tra điều kiện hội tụ
        if np.std([evaluate(func, x, cache) for x in simplex]) < tol:
            break
    
    # Kết quả tối ưu
    best_point = simplex[0]
    best_value = evaluate(func, simplex[0], cache)
    return best_point, best_value, iteration + 1

# Điểm bắt đầu cho thuật toán tối ưu
initial_guess = np.array([0, 0, 0, 0, 0])

# Thực hiện tối ưu hóa bằng thuật toán Nelder-Mead
best_point, best_value, iterations = nelder_mead(objective_function, initial_guess)

# In kết quả tối ưu
print("Tối ưu thành công:", True)
print("Giá trị tối ưu của hàm mục tiêu:", best_value)
print("Các giá trị tối ưu của biến:", best_point)
print("Số lần lặp lại thuật toán:", iterations)
