import numpy as np
import matplotlib.pyplot as plt

labrador = 500
greyhound = 500

grey_height = 28 + 4 * np.random.randn(greyhound)
lab_height = 24 + 4 * np.random.randn(labrador)

plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()