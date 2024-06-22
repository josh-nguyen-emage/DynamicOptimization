from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

def load_lists_from_file(filename):
    list1 = []
    list2 = []
    with open(filename, 'r') as file:
        current_list = list1
        for line in file:
            line = line.strip()
            if line == "===":
                current_list = list2
            else:
                current_list.append(float(line))
    return list1, list2

l1, l2 = load_lists_from_file("pltDraw.txt")
plt.figure(figsize=(6, 4))
# plt.text(3,0, TextVal, fontsize=14)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('accuracy', fontsize=20)

ax = plt.gca()

plt.plot(range(len(l1)),l1, label="Train set")
plt.plot(range(len(l2)),l2, label="Val set")
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

ax.tick_params(which='major', length=5, width=2, colors='black', labelsize=14, direction='in')
ax.tick_params(which='minor', length=2, width=1, colors='black', direction='in')

plt.tight_layout()

ax.grid(which='major', linestyle='--')
plt.legend()
plt.show()