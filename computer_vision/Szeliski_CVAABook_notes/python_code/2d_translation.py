import numpy as np

pt = np.array([2,3,1])
matrix = np.array([[1,0,4], [0,1,5], [0,0,1]])

result = matrix @ pt

print(f"New point: ({result[0]}, {result[1]})")
