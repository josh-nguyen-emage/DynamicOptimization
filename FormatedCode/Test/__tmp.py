import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu mẫu
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Tạo đồ thị miền
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.fill_between(x, y1, y2, where=(y1 > y2), interpolate=True, color='gray', alpha=0.5, label='Area between curves')

# Thêm tiêu đề và chú thích
plt.title('Filled Area Between sin(x) and cos(x)')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()

# Hiển thị đồ thị
plt.show()
