from os import listdir
from numpy import array, zeros, ndarray
from scipy.stats.mstats import mquantiles
from matplotlib.pyplot import figure, show
from matplotlib import rcParams, use
import json

from DataScripts.WSDataRoutines import ReadWSFile

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def AnalysisByCoordinate(Hours:list[int], Years:list[int], Coordinates:dict, FilesList:list[str]) -> dict:
    DispersionCoordinate = dict()

    for Region in Coordinates.keys():
        print(f"Starting dispersion analysis on {Region}", end="")
        LatitudesInRegion = Coordinates[Region]["Lat"]
        LongitudesInRegion = Coordinates[Region]["Long"]

        DispersionCoordinate[Region] = zeros((6,12), dtype=float)

        for Month in range(1,13):
            for n, Hour in enumerate(Hours):
                DataPerHourInMonth = []

                # Search every file in the region [IntLat-1, IntLat+1]x[IntLong-1, IntLong+1]
                for IntLat, IntLong in zip(LatitudesInRegion,LongitudesInRegion):

                    # First, search for neighbours coordinates
                    for dx in range(-1,2):
                        for dy in range(-1,2):
                            NewLat = IntLat + dy
                            NewLong = IntLong + dx

                            for file in FilesList:
                                if file.endswith(".txt"):
                                    SplitFileName = file.split("/")[-1].split("_")
                                    FileLat = SplitFileName[5][:3]
                                    FileLong = SplitFileName[6][:3]

                                    if NewLat == int(FileLat) and NewLong == int(FileLong):
                                        DataPerHourInMonth += ReadWSFile(file, Hour, Month, Years)

                DataQuantiles = mquantiles(array(DataPerHourInMonth))[:]
                if n: 
                    DispersionCoordinate[Region][:3,Month-1] = DataQuantiles[:]
                else:
                    DispersionCoordinate[Region][3:,Month-1] = DataQuantiles[:]

        print("...finished!")

    with open("./WindSpeedData/WindDispersion.json", "w") as OutJSON:
        json.dump(DispersionCoordinate, OutJSON, cls=NumpyArrayEncoder)

    return DispersionCoordinate

if __name__ == "__main__":
    WS_DataDir = "../50M Wind Maps"
    WS_ListFiles = list(map(lambda x: WS_DataDir + "/" + x , listdir(WS_DataDir)))

    Coordinates = dict(
        North = dict(Lat=[31], Long=[116]),
        Center_MNIG = dict(Lat=[25], Long=[100]),
        Center_UCOE = dict(Lat=[19], Long=[101]),
        South = dict(Lat=[20], Long=[87])
    )

    Hours = [12, 20]
    Years = [2018, 2019]

    WSDispersionRegions = AnalysisByCoordinate(Hours, Years, Coordinates, WS_ListFiles)


    #---------------------------------------------------------------------------------------
    use("TkAgg")
    rcParams["font.family"] = "serif"
    rcParams['savefig.dpi'] = 400

    Figure = figure(1, figsize=(7,6))
    Regions = WSDispersionRegions.keys()
    N = len(Regions)
    Subplots = []


    Width = 0.125
    dx = 0.5
    MonthsDay = [x + dx - Width for x in range(1,13)]
    MonthsNight = [x + dx +Width for x in range(1,13)]
    YminList, YmaxList = [], []
    for n, Region in enumerate(Regions):
        Subplots.append(Figure.add_subplot(N,1,n+1))
        Subplots[n].set_title(Region)

        for x in range(1,13):
            Subplots[n].axvline(x, linestyle="--", linewidth=1.0, color="black")

        HalfDispersionDay = 0.5*(WSDispersionRegions[Region][2,:]-WSDispersionRegions[Region][1,:])
        Subplots[n].errorbar(MonthsDay, WSDispersionRegions[Region][1,:], yerr=HalfDispersionDay,
                             ecolor="red", color="red", markerfacecolor="red", fmt="o")

        HalfDispersionNight = 0.5*(WSDispersionRegions[Region][5,:]-WSDispersionRegions[Region][3,:])
        Subplots[n].errorbar(MonthsNight, WSDispersionRegions[Region][4,:], yerr=HalfDispersionNight,
                             ecolor="blue", color="blue", markerfacecolor="blue", fmt="o")
        

        Ymin, Ymax = Subplots[n].get_ylim()
        YminList.append(Ymin)
        YmaxList.append(Ymax)

    minY, maxY = min(YminList), max(YmaxList)
    Months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for n in range(len(Regions)):
        Subplots[n].set_ylim(minY, maxY)
        Subplots[n].set_xlim(1, 13)
        Subplots[n].set_xticks(ticks=list(range(1,13)),
                               labels=Months)

    Figure.tight_layout()
    show()