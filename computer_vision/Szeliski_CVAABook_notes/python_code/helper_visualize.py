import numpy as np
import matplotlib.pyplot as plt

def visualize_transform_steps(px, py, tx, ty, theta_deg):
    """
    Visualizes the step-by-step transformation (Rotation, then Translation)
    of a 2D point using Affine Matrices.
    """
    # 1. THE MATH
    theta = np.radians(theta_deg)
    
    # Create the homogeneous input vector [x, y, 1]
    p_original = np.array([px, py, 1])
    
    # Matrix A: Just the Rotation (Translation is 0,0)
    R_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0]
    ])
    
    # Matrix B: Full Affine Matrix (Rotation + Translation)
    Rt_matrix = np.array([
        [np.cos(theta), -np.sin(theta), tx],
        [np.sin(theta),  np.cos(theta), ty]
    ])
    
    # Calculate the points
    p_rotated = R_matrix @ p_original       # Intermediate step
    p_final = Rt_matrix @ p_original        # Final step
    
    # 2. THE VISUALIZATION
    plt.figure(figsize=(10, 8))
    
    # Draw axes and grid
    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 1: Original Point (Blue)
    plt.plot(px, py, 'bo', markersize=9, label=f'1. Original: ({px}, {py})')
    plt.plot([0, px], [0, py], 'b--', alpha=0.4) # Line to origin to show radius
    
    # Plot 2: Rotated Point (Green)
    plt.plot(p_rotated[0], p_rotated[1], 'go', markersize=9, 
             label=f'2. Rotated {theta_deg}°: ({p_rotated[0]:.2f}, {p_rotated[1]:.2f})')
    plt.plot([0, p_rotated[0]], [0, p_rotated[1]], 'g--', alpha=0.4)
    
    # Plot 3: Final Translated Point (Red)
    plt.plot(p_final[0], p_final[1], 'ro', markersize=9, 
             label=f'3. Translated +({tx}, {ty}): ({p_final[0]:.2f}, {p_final[1]:.2f})')
    
    # Draw animated arrows showing the mathematical flow
    plt.annotate('', xy=p_rotated, xytext=(px, py),
                 arrowprops=dict(arrowstyle="->", color='green', lw=2, connectionstyle="arc3,rad=0.3"))
    plt.annotate('', xy=p_final, xytext=p_rotated,
                 arrowprops=dict(arrowstyle="->", color='red', lw=2, ls='--'))
    
    # CRITICAL: Equal aspect ratio prevents rotation from looking skewed/elliptical
    plt.axis('equal') 
    
    # Auto-scale the window with some padding
    all_x = [0, px, p_rotated[0], p_final[0]]
    all_y = [0, py, p_rotated[1], p_final[1]]
    plt.xlim(min(all_x) - 2, max(all_x) + 2)
    plt.ylim(min(all_y) - 2, max(all_y) + 2)
    
    # Formatting
    plt.title(f'Affine Transformation Pipeline', pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.legend(loc='upper left')
    plt.show()

