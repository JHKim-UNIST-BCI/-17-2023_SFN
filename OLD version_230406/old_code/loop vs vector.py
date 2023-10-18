import numpy as np
import time

# Define the size of the array
n = 10000000

# Create a random array of size n
arr = np.random.rand(n)

# Vectorized calculation using NumPy
start_time = time.time()
result1 = np.sum(arr ** 2)
end_time = time.time()
print("Vectorized calculation time:", end_time - start_time)

# Calculation using for loop
start_time = time.time()
result2 = 0
for i in range(n):
    result2 += arr[i] ** 2
end_time = time.time()
print("For loop calculation time:", end_time - start_time)

# Verify that the results are the same
print("Results are equal:", np.isclose(result1, result2))
