import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def numbers_to_viridis_colors(numbers, vmin=None, vmax=None):
    """
    Convert a list of real numbers to colors using the viridis colormap.
    
    Args:
        numbers (list or numpy.ndarray): List of real numbers to convert to colors
        vmin (float, optional): Minimum value for normalization. If None, uses min of numbers
        vmax (float, optional): Maximum value for normalization. If None, uses max of numbers
        
    Returns:
        list: List of RGB colors in the format (r, g, b) where each component is between 0 and 1
    """
    # Convert input to numpy array if it's not already
    numbers = np.array(numbers)
    
    # Create a normalizer to map the numbers to [0, 1] range
    norm = Normalize(vmin=vmin if vmin is not None else numbers.min(), 
                    vmax=vmax if vmax is not None else numbers.max())
    
    # Get the viridis colormap
    cmap = plt.cm.viridis
    
    # Convert numbers to colors
    colors = cmap(norm(numbers))
    
    # Return list of RGB colors (excluding alpha channel)
    return colors[:, :3].tolist()

# Example usage
if __name__ == "__main__":
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create a figure with multiple subplots to show different colormaps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Different Colormaps in Matplotlib', fontsize=16)
    
    # List of colormaps to demonstrate
    colormaps = [
        ('viridis', 'Perceptually uniform, colorblind-friendly'),
        ('plasma', 'Perceptually uniform, colorblind-friendly'),
        ('inferno', 'Perceptually uniform, colorblind-friendly'),
        ('magma', 'Perceptually uniform, colorblind-friendly'),
        ('cividis', 'Colorblind-friendly, optimized for printing'),
        ('RdYlBu', 'Red-Yellow-Blue, diverging')
    ]
    
    # Create scatter plots with different colormaps
    for idx, (cmap_name, description) in enumerate(colormaps):
        row = idx // 3
        col = idx % 3
        
        scatter = axes[row, col].scatter(x, y, c=y, cmap=cmap_name)
        axes[row, col].set_title(f'{cmap_name}\n{description}')
        plt.colorbar(scatter, ax=axes[row, col])
    
    plt.tight_layout()
    plt.show()
    
    # Print all available colormaps
    print("\nAll available colormaps in matplotlib:")
    print(sorted(plt.colormaps())) 