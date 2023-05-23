from matplotlib import colormaps
from matplotlib.colors import ListedColormap, BoundaryNorm
from numpy import linspace

def ObtainCMAPandNORM(OcurrenceArray):
    """
    Obtain colormap and normalization for TIDs' ocurrence array to show
    with discrete colors
    """
    
    # Define the colormap
    CMAP = colormaps["jet"].resampled(256)
    # Extract all colors from the jet map
    NewColors = CMAP(linspace(0, 1, 256))
    # Force the first color entry to be transparent
    NewColors[0] = (0.0, 0.0, 0.0, 0.0)

    # Create the new CMAP
    CMAP = ListedColormap(name="OcurrenceMap", colors=NewColors)
    # Define Bounds array
    BOUNDS = linspace(OcurrenceArray.min(), OcurrenceArray.max(), 11, endpoint=True)
    # Create NORM array
    NORM = BoundaryNorm(BOUNDS, CMAP.N)

    return CMAP, NORM