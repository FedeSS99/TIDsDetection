from matplotlib.pyplot import figure, plot, show
from atplotlib import rcParams
from mpl_toolkits.basemap import Basemap
from numpy import arange

rcParams["font.size"] = 12
rcParams["savefig.dpi"] = 400
Figure = figure(1, figsize=(12, 6))
Map = Basemap(projection="cyl", resolution="h",
              llcrnrlat=14, urcrnrlat=34,
              llcrnrlon=-118, urcrnrlon=-78)

Map.shadedrelief()
Map.drawcoastlines()
Map.drawcountries()
LinesParallels = arange(14, 34, 4)
LinesMeridians = arange(-78, -118, -8)
Map.drawparallels(LinesParallels, labels=[True, False, False, False])
Map.drawmeridians(LinesMeridians, labels=[False, False, True, False])

Coords = {"MNIG": (-100.285228, 25.609308), "UCOE": (-101.69442, 19.81318),
          "CN24": (-88.05391, 19.57556), "UNPM": (-86.86686, 20.86812),
          "PTEX": (-116.52124, 32.28845), "PALX": (-116.06379, 31.55912)}

N = len(Coords)
Colors = ["red", "orange", "blue", "cyan", "green", "lime"]

CoordsMap = [Map(*Coord) for Coord in Coords.values()]
for CoordMap, Station, Color in zip(CoordsMap, Coords.keys(), Colors):
    plot(*CoordMap, "^", color=Color, markersize=10, label=Station)

Figure.legend(loc=7, fancybox=True, shadow=True)
show()
