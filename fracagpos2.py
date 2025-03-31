import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Hardcoded parameters
N = 1000  # Number of particles
a = .5  # Radius of the particles
Df = 1.8  # Fractal dimension
kf = 1.1  # Fractal prefactor
Rg = 5.0  # Radius of gyration (not directly used in the algorithm, but related to N, a, Df, and kf)

# Helper functions
def dot(v1, v2):
    return np.dot(v1, v2)

def norm(v):
    return np.linalg.norm(v)

def rotate_vector(axis, theta, v):
    """Rotate vector v around axis by angle theta."""
    axis = axis / norm(axis)
    ct = math.cos(theta)
    st = math.sin(theta)
    rotation_matrix = np.array([
        [ct + axis[0]**2 * (1 - ct), axis[0] * axis[1] * (1 - ct) - axis[2] * st, axis[0] * axis[2] * (1 - ct) + axis[1] * st],
        [axis[1] * axis[0] * (1 - ct) + axis[2] * st, ct + axis[1]**2 * (1 - ct), axis[1] * axis[2] * (1 - ct) - axis[0] * st],
        [axis[2] * axis[0] * (1 - ct) - axis[1] * st, axis[2] * axis[1] * (1 - ct) + axis[0] * st, ct + axis[2]**2 * (1 - ct)]
    ])
    return np.dot(rotation_matrix, v)

def generate_random_point_in_intersection(c1, c2, r1, r2):
    """Generate a random point in the intersection of two spheres."""
    c2_rel = c2 - c1
    dist = norm(c2_rel)
    if dist > r1 + r2 or dist < abs(r1 - r2):
        raise ValueError("No intersection between the spheres.")
    
    # Rotate to a frame where c2 is along the z-axis
    axis = np.array([c2_rel[1], -c2_rel[0], 0])
    axis_norm = norm(axis)
    
    # Check if the axis is a zero vector (no rotation needed)
    if axis_norm < 1e-10:
        # No rotation needed; c2 is already along the z-axis
        ct = 1.0
        st = 0.0
    else:
        # Normalize the axis and compute rotation parameters
        axis = axis / axis_norm
        ct = c2_rel[2] / dist
        st = math.sqrt(1 - ct**2) if ct < 1 else 0.0
    
    # Compute the intersection circle
    z = (r1**2 - r2**2 + dist**2) / (2 * dist)
    circle_radius = math.sqrt(r1**2 - z**2)
    
    # Generate a random point on the circle
    phi = 2 * math.pi * random.random()
    x = circle_radius * math.cos(phi)
    y = circle_radius * math.sin(phi)
    
    # Rotate back to the original frame (if rotation is needed)
    if axis_norm >= 1e-10:
        point = rotate_vector(axis, math.acos(ct), np.array([x, y, z])) + c1
    else:
        point = np.array([x, y, z]) + c1
    
    return point

def compute_radius_valid_pos(Df, kf, radius, Ni):
    """Compute the radius where a new particle can be placed."""
    t1 = Ni * radius**2 / (Ni - 1)
    t2 = Ni * (Ni / kf)**(2 / Df)
    t3 = -(Ni - 1) * ((Ni - 1) / kf)**(2 / Df) - 1
    return math.sqrt(t1 * (t2 + t3))

def generate_fractal_cluster(N, radius, Df, kf):
    """Generate a fractal cluster of particles."""
    pos = [np.array([0.0, 0.0, -radius]), np.array([0.0, 0.0, radius])]
    cm = np.array([0.0, 0.0, 0.0])
    
    for id in range(2, N):
        while True:
            try:
                r_valid = compute_radius_valid_pos(Df, kf, radius, id + 1)
                valid_particles = [i for i in range(len(pos)) if norm(cm - pos[i]) <= r_valid + 2 * radius]
                if not valid_particles:
                    raise ValueError("No valid particles found.")
                
                # Pick a random particle to bond with
                particle_to_bond = random.choice(valid_particles)
                new_pos = generate_random_point_in_intersection(cm, pos[particle_to_bond], r_valid, 2 * radius)
                
                # Check for overlaps
                overlap = any(norm(new_pos - p) < 2 * radius for p in pos)
                if not overlap:
                    pos.append(new_pos)
                    cm = (new_pos + cm * id) / (id + 1)
                    break
            except ValueError:
                # Retry if no valid position is found
                continue
    
    return pos

def plot_fractal_cluster(positions):
    """Plot the fractal cluster in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    z = [p[2] for p in positions]
    
    # Plot the points
    ax.scatter(x, y, z, s=50, c='b', marker='o')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    ax.set_title(f"Fractal Cluster (N={N}, Df={Df}, kf={kf})")
    
    # Show the plot
    plt.show()

def calculate_intensity(positions, q_values):
    """Calculate the intensity I(q) for a range of q values."""
    N = len(positions)
    I_q = []
    
    for q in q_values:
        S_q = 0.0
        for i in range(N):
            for j in range(N):
                r_ij = norm(positions[i] - positions[j])
                if r_ij > 0:  # Avoid division by zero
                    S_q += np.sin(q * r_ij) / (q * r_ij)
        S_q /= N  # Normalize by the number of particles
        I_q.append(S_q)
    
    return I_q

def plot_intensity(q_values, I_q):
    """Plot the intensity I(q) as a function of q."""
    plt.figure()
    plt.plot(q_values, I_q, 'b-', label='I(q)')
    plt.xlabel('Scattering Vector q')
    plt.ylabel('Intensity I(q)')
    plt.title('Scattering Intensity vs Scattering Vector')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """Main function to generate, visualize, and analyze the fractal cluster."""
    # Generate the fractal cluster
    positions = generate_fractal_cluster(N, a, Df, kf)
    
    # Output the positions
    for p in positions:
        print(f"{p[0]} {p[1]} {p[2]}")
    
    # Plot the fractal cluster in 3D
    plot_fractal_cluster(positions)
    
    # Calculate intensity I(q) for a range of q values
    q_values = np.linspace(0.1, 10.0, 100)  # Range of q values
    I_q = calculate_intensity(positions, q_values)
    
    # Plot the intensity I(q) vs q
    plot_intensity(q_values, I_q)

if __name__ == "__main__":
    main()