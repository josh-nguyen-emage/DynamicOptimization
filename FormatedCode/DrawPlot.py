# Load the text file
from matplotlib import pyplot as plt
import numpy as np

import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))

from Libary.function import *

realExp, Sim = read_arrays_from_txt("FDSEV2 P3.txt")

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(realExp[0],realExp[1],'o-',label='Real Experiment')
ax.plot(Sim[0],Sim[1],'o-',label='Atena Simulation')

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

ax.tick_params(axis="x",which="major",direction="in",length=8, width=2,
               pad=2,labelsize=14,labelcolor="black",bottom=True) 

ax.tick_params(axis="x",which="minor",direction="in",length=6,width=1,
               pad=0.5,labelsize=18,labelcolor="black",bottom=True)

ax.tick_params(axis="y",which="major",direction="in",length=8, width=2,
               pad=20,labelsize=14,labelcolor="black") 

ax.tick_params(axis="y",which="minor",direction="in",length=6,width=1,
               pad=5,labelsize=18,labelcolor="black")

ax.minorticks_on()

ax.set_xlabel("ε (‰)", fontsize=20)
ax.set_ylabel("σ (MPa)", fontsize=20)
# Show plot
fig.legend()
plt.show()

