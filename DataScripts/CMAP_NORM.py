from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from numpy import linspace

def ObtainCMAPandNORM(OcurrenceArray):
    # Define the colormap
    CMAP = get_cmap("jet")
    # Extract all colors from the jet map
    CMAPlist = [CMAP(i) for i in range(CMAP.N)]
    # Force the first color entry to be transparent
    CMAPlist[0] = (0.0, 0.0, 0.0, 0.0)

    # Create the new CMAP
    CMAP = LinearSegmentedColormap.from_list("Ocurrence Map", CMAPlist, CMAP.N)
    # Define Bounds array
    BOUNDS = linspace(OcurrenceArray.min(),
                         OcurrenceArray.max(), 9, endpoint=True)
    # Create NORM array
    NORM = BoundaryNorm(BOUNDS, CMAP.N)

    return CMAP, NORM