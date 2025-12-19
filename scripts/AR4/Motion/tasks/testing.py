import numpy as np

def generate_3d_figure8(n_points=300, turns=2, A=0.15, B=0.1, C=0.05, center=np.array([0.4, 0.0, 0.3])):
    """
    Generate a 3D figure-8 (lemniscate) trajectory.

    Parameters:
    - n_points: number of points in the trajectory
    - turns: number of figure-8 repetitions
    - A: amplitude in X-axis
    - B: amplitude in Y-axis
    - C: amplitude in Z-axis
    - center: center of the trajectory (x0, y0, z0)

    Returns:
    - Nx3 numpy array of (X, Y, Z) coordinates
    """
    t = np.linspace(0, 2*np.pi*turns, n_points)
    x = A * np.sin(t)
    y = B * np.sin(t) * np.cos(t)
    z = C * np.sin(2*t)

    points = np.vstack([x, y, z]).T + center
    return points

# Generate trajectory
trajectory_points = generate_3d_figure8()

# Print X, Y, Z coordinates
print("X, Y, Z coordinates of 3D Figure-8 trajectory:")
for i, point in enumerate(trajectory_points):
    print(f"Point {i+1}: X={point[0]:.4f}, Y={point[1]:.4f}, Z={point[2]:.4f}")

# Optional: visualize trajectory
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory_points[:,0], trajectory_points[:,1], trajectory_points[:,2], color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Figure-8 Trajectory')
    plt.show()
except ImportError:
    print("Matplotlib not installed. Skipping visualization.")
