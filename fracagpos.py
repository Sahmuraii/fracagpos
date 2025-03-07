import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def generate_fractal_cluster(num_particles, radius, fractal_dim, k_f, max_iter=10000):
    """
    Generate a fractal cluster of spheres using a pseudo-random algorithm.
    
    Parameters:
    - num_particles: Number of particles (spheres) in the cluster.
    - radius: Radius of each sphere.
    - fractal_dim: Fractal dimension (D_f).
    - k_f: Proportionality constant (k_f).
    - max_iter: Maximum number of iterations to attempt to form the cluster.
    
    Returns:
    - positions: Array of particle positions in the cluster.
    """
    
    # Initialize particle positions randomly within a bounding box
    positions = np.random.rand(num_particles, 3) * 10 * radius
    
    # Main loop to form the cluster
    for iteration in range(max_iter):
        # Calculate the radius of gyration R_g
        center_of_mass = np.mean(positions, axis=0)
        R_g = np.sqrt(np.mean(np.sum((positions - center_of_mass)**2, axis=1)))
        
        # Calculate the expected number of particles based on fractal scaling
        N_expected = k_f * (R_g / radius)**fractal_dim
        
        # If the expected number of particles is close to the actual number, stop
        if abs(N_expected - num_particles) < 1:
            break
        
        # Move particles randomly
        positions += np.random.normal(0, radius/10, positions.shape)
        
        # Check for overlaps and merge clusters
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        close_pairs = np.argwhere(distances < 2 * radius)
        
        for i, j in close_pairs:
            if i < j:  # Avoid double-counting
                # Merge the two particles by moving them to their midpoint
                midpoint = (positions[i] + positions[j]) / 2
                positions[i] = midpoint
                positions[j] = midpoint
        
        # Remove duplicates (merged particles)
        positions = np.unique(positions, axis=0)
        
        # If we have too few particles, add new ones
        while len(positions) < num_particles:
            new_particle = np.random.rand(1, 3) * 10 * radius
            positions = np.vstack([positions, new_particle])
    
    return positions

def main():
    # Parameters
    num_particles = 5000
    radius = 0.5
    fractal_dim = 1.8
    k_f = 1.1

    # Generate the fractal cluster
    positions = generate_fractal_cluster(num_particles, radius, fractal_dim, k_f)

    # Plot the fractal cluster
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=radius*100)
    ax.set_aspect('auto')
    plt.title(f"Fractal Cluster with D_f = {fractal_dim}")
    plt.show()

if __name__ == "__main__":
    main()