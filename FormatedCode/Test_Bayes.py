
from bayes_opt import BayesianOptimization
import threading
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from Libary.RunSequent import *
from Libary.function import *

class DTW:
    def __init__(self, method, index):
        self.method = method
        self.index = index

    def objective_function(self,x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
        params = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]
        WriteParameter(params, self.index)
        RunSimulation(self.index)
        RunTool4Atena(self.index)
        outputData = ExtractResult(self.index)
        save_to_file(params,outputData,"Log_Run_Bayes_1.txt")
        strain = outputData[0]
        stress = outputData[1]
        MSE, interpolateLine = findF(strain,stress)
        print("curent MSE of",self.method,":",MSE)
        return -1*MSE

if __name__ == "__main__":

    # Giới hạn của các biến (từ -10 đến 10 cho mỗi biến)
    pbounds = {
        'x1': (0, 1),
        'x2': (0, 1),
        'x3': (0, 1),
        'x4': (0, 1),
        'x5': (0, 1),
        'x6': (0, 1),
        'x7': (0, 1),
        'x8': (0, 1),
        'x9': (0, 1),
        'x10': (0, 1),
        'x11': (0, 1)
    }

    dtw = DTW("Bayes",0)

    optimizer = BayesianOptimization(
        f=dtw.objective_function,
        pbounds=pbounds,
        random_state=1,
    )

    # Thực hiện quá trình tối ưu hóa
    optimizer.maximize(
        init_points=10,  # Số lượng điểm khởi tạo ngẫu nhiên
        n_iter=300,      # Số lần lặp tối ưu hóa
    )

    # In kết quả tối ưu
    print(optimizer.max)


