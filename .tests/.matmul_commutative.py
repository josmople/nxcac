import numpy as np


a = np.random.rand(3, 4)
b = np.random.rand(4, 5)

u = a @ b
v = (b.T @ a.T).T

print(u == v)

# CONCLUSION
# A @ B == (B.T @ A.T).T
