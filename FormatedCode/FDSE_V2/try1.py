import sys, os

import numpy as np
sys.path.append(os.path.abspath(os.path.join('.')))

from Libary.function import calculate_correlation


a = [1,2,3,4,5,6]
b = [2,4,6,8,10,12]
print(calculate_correlation(np.array(a),np.array(b)))