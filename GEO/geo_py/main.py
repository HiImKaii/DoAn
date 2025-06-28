import numpy as np
import matplotlib.pyplot as plt
from geo_visualizer import GEOVisualizer

def main():
    # Define optimization options
    options = {
        'PopulationSize': 25,  # 5x5 grid of eagles
        'MaxIterations': 50,
        'AttackPropensity': [0.1, 0.9],  # Attack rate increases over iterations
        'CruisePropensity': [0.1, 0.1]   # Constant small cruise rate
    }

    # Define a simple objective function (for demonstration)
    def objective_function(x):
        return np.sum(x**2)  # Simple sphere function

    # Create and run the visualizer
    visualizer = GEOVisualizer(
        fun=objective_function,
        nvars=2,              # 2D visualization
        lb=[-5, -5],         # Lower bounds
        ub=[5, 5],          # Upper bounds
        options=options
    )

    # Run the animation
    anim = visualizer.animate(interval=200)  # 200ms between frames
    plt.show()  # Make sure the plot window appears
    
if __name__ == "__main__":
    main() 