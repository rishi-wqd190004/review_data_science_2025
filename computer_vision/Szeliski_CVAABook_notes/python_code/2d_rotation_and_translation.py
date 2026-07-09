import numpy as np
import matplotlib.pyplot as plt

# angle in radians (pi/3 is 60 degrees)
theta = np.pi / 3
tx, ty = 1, 2
px, py = 5, 5

translation = np.array([
    [tx],
    [ty]    
])

R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

Rt = np.hstack((R, translation))

x_bar = np.array([px, py, 1])

x_prime = Rt @ x_bar

print("The 2x3 [R t] Matrix:")
print(np.round(Rt, 2))
print(f"\nFinal x: ({x_prime[0]:.2f}, {x_prime[1]:.2f})")

print('\nPerofrming rotation and translation for a square')

# 3. Define our Shape (A Square)
# We have 4 points: (0,0), (1,0), (1,1), (0,1)
# To do them all at once, we stack them into a 3x4 matrix (x, y, 1 for each point)
# Notice how each COLUMN is one point [x, y, 1]
square = np.array([
    [0,1,1,0], # x coordinates
    [0,0,1,1], # Y coordinates
    [1,1,1,1] # homogeneous 1s
])

transformed_square = Rt @ square

# --- Quick Visualization ---
plt.figure(figsize=(6,6))
plt.axhline(0, color='black'); plt.axvline(0, color='black'); plt.grid(True, alpha=0.5)

# Plot Original (Blue). We append the first point at the end to close the square box
plt.plot(np.append(square[0], square[0][0]), np.append(square[1], square[1][0]), 'b-', label='Original', lw=2)

# Plot Transformed (Red)
plt.plot(np.append(transformed_square[0], transformed_square[0][0]), 
         np.append(transformed_square[1], transformed_square[1][0]), 'r-', label='Transformed', lw=2)

plt.axis('equal'); plt.legend(); plt.title("Transforming a Whole Shape at Once"); plt.show()