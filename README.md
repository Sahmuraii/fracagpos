# Fractal Cluster Generator

This script generates a 3D fractal cluster of particles, visualizes the cluster, and calculates the scattering intensity as a function of the scattering vector \( q \).

## Features

- Generates a fractal cluster of particles with specified parameters.
- Plots the 3D structure of the fractal cluster.
- Calculates and plots the scattering intensity \( I(q) \) for a range of \( q \) values.

## Parameters

- `N`: Number of particles (default: 1000).
- `a`: Radius of the particles (default: 0.5).
- `Df`: Fractal dimension (default: 1.8).
- `kf`: Fractal prefactor (default: 1.1).
- `Rg`: Radius of gyration (related to other parameters but not directly used in the algorithm).

## Functions

1. **`generate_fractal_cluster(N, radius, Df, kf)`**  
   Generates a fractal cluster of `N` particles with the given radius, fractal dimension, and prefactor.

2. **`plot_fractal_cluster(positions)`**  
   Plots the 3D positions of the particles in the fractal cluster.

3. **`calculate_intensity(positions, q_values)`**  
   Computes the scattering intensity \( I(q) \) for the given positions and range of \( q \) values.

4. **`plot_intensity(q_values, I_q)`**  
   Plots the scattering intensity \( I(q) \) as a function of \( q \).

## Usage

1. Run the script to generate the fractal cluster and visualize it.
2. The script will output the positions of all particles in the cluster.
3. Two plots will be displayed:
   - A 3D scatter plot of the fractal cluster.
   - A plot of the scattering intensity \( I(q) \) versus \( q \).

## Dependencies

- `numpy`
- `random`
- `math`
- `matplotlib`
- `mpl_toolkits.mplot3d`

## Example Output

- The 3D plot shows the fractal structure of the cluster.
- The intensity plot demonstrates the fractal properties through the scattering pattern.
