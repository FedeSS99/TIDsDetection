def SaveRegionPlot(GenName, RegName, PlotFigure):
    for format in ["png", "pdf"]:
        PlotFigure.savefig(
            f"./../Results/{RegName}/{GenName}{RegName}.{format}")


def SaveStationPlot(GenName, RegName, StationName, PlotFigure):
    for format in ["png", "pdf"]:
        PlotFigure.savefig(
            f"./../Results/{RegName}/{StationName}/{GenName}{StationName}.{format}")

def SaveAllRegionPlot(GenName, PlotFigure):
    for format in ["png", "pdf"]:
        PlotFigure.savefig(f"./../Results/{GenName}.{format}")
