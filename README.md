# Fractal Cluster Generator

This Python script generates a 3D fractal cluster of particles, visualizes the cluster, and calculates the scattering intensity as a function of the scattering vector \( q \).

## Features

- Generates a fractal cluster of particles with customizable parameters:
  - Number of particles (`N`)
  - Particle radius (`a`)
  - Fractal dimension (`Df`)
  - Fractal prefactor (`kf`)
- Visualizes the cluster in a 3D plot.
- Computes and plots the scattering intensity \( I(q) \) for a range of \( q \) values.

## Parameters

The following hardcoded parameters can be adjusted in the script:

- `N = 1000`: Number of particles.
- `a = 0.5`: Radius of each particle.
- `Df = 1.8`: Fractal dimension of the cluster.
- `kf = 1.1`: Fractal prefactor.
- `Rg = 5.0`: Radius of gyration (related to the cluster's size).

## Functions

- **`generate_fractal_cluster`**: Generates the fractal cluster using a diffusion-limited aggregation (DLA) inspired algorithm.
- **`plot_fractal_cluster`**: Visualizes the cluster in 3D using `matplotlib`.
- **`calculate_intensity`**: Computes the scattering intensity \( I(q) \) for given \( q \) values.
- **`plot_intensity`**: Plots the intensity \( I(q) \) as a function of \( q \).

## Usage

1. Ensure the required libraries are installed:
   ```bash
   pip install numpy matplotlib
   ```
   
