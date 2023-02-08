from matplotlib.pyplot import figure, plot, show
from matplotlib import rcParams
from mpl_toolkits.basemap import Basemap
from numpy import arange

rcParams["font.size"] = 12
Figure = figure(1, figsize=(6,6))
Map = Basemap(projection="cyl", resolution="h",
            llcrnrlat=14, urcrnrlat=34,
            llcrnrlon=-118, urcrnrlon=-78)

Map.shadedrelief()
Map.drawcoastlines()
Map.drawcountries()
LinesParallels = arange(14, 34, 4)
LinesMeridians = arange(-78, -118, -8)
Map.drawparallels(LinesParallels, labels = [True, False, False, False])
Map.drawmeridians(LinesMeridians, labels = [False, False, True, False])

Coords = {"MNIG":(-100.285228, 25.609308), "LNIG":(-99.465617, 24.898167),
        "CN24":(-88.054, 19.576), "TGMX":(-86.86686, 20.86812),
        "PTEX":(-116.521214, 32.28845), "UCOE":(-101.694, 19.813)}

N = len(Coords)
Colors = ["red", "blue", "green", "brown", "black", "purple"]

CoordsMap = [Map(*Coord) for Coord in Coords.values()]
for CoordMap, Station, Color in zip(CoordsMap, Coords.keys(), Colors):
    plot(*CoordMap, "^", color=Color, markersize=10, label=Station)

Figure.legend(loc=7)
show()
