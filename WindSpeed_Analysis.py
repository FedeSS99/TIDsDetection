from os import listdir
from numpy import ndarray, array, zeros, ndarray
from scipy.stats.mstats import mquantiles
from matplotlib.pyplot import figure, show
from matplotlib import rcParams, use
import json

from DataScripts.WSDataRoutines import ReadWSFile


# Encoder class to save numpy arrays as lists in json files
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Function to get all data in a zone given the latitudes and longitudes
# [IntLat-1, IntLat, IntLat+1], [IntLong-1, IntLong, IntLong+1]
def ExtractData(Latitudes:list[int], Longitudes:list[int], FilesList:list[str])-> dict:
    DataInZone = dict(
        Year = [],
        Month = [],
        Hour = [],
        Velocity = []
    )

    for Lat, Long in zip(Latitudes,Longitudes):
        # First, search for neighbours coordinates
        for dx in range(-1,2):
            for dy in range(-1,2):
                NewLat = Lat + dy
                NewLong = Long + dx

                for file in FilesList:
                    if file.endswith(".txt"):
                        SplitFileName = file.split("/")[-1].split("_")
                        FileLat = SplitFileName[5][:3]
                        FileLong = SplitFileName[6][:3]

                        if NewLat == int(FileLat) and NewLong == int(FileLong):
                            DataInFile = ReadWSFile(file)

                            DataInZone["Year"] += DataInFile[0]
                            DataInZone["Month"] += DataInFile[1]
                            DataInZone["Hour"] += DataInFile[2]
                            DataInZone["Velocity"] += DataInFile[3]
    return DataInZone

def AnalysisByCoordinate(Hours:list[int], Coordinates:dict, FilesList:list[str]) -> dict:
    DataByRegions = dict()
    for Region in Coordinates.keys():
        LatitudesInRegion = Coordinates[Region]["Lat"]
        LongitudesInRegion = Coordinates[Region]["Long"]
        
        DataInRegion = ExtractData(LatitudesInRegion, LongitudesInRegion, FilesList)
        DataByRegions[Region] = DataInRegion
        print(f"Saved data for {Region} coordinates")

    DispersionCoordinateMonths = dict()
    DispersionCoordinateHours = dict()  
    for Region in Coordinates.keys():
        print(f"Starting dispersion analysis on {Region}", end="")

        DispersionCoordinateMonths[Region] = zeros((6,12), dtype=float)
        DispersionCoordinateHours[Region] = zeros((3,24), dtype=float)

        for Month in range(1,13):
            MonthMask = DataByRegions[Region]["Month"] == Month

            for n, Time in enumerate(Hours):
                TimeMask = DataByRegions[Region]["Hour"] == Time

                MaskedVelocity = DataByRegions[Region]["Velocity"][MonthMask*TimeMask]

                DataQuantiles = mquantiles(array(MaskedVelocity))
                if n: 
                    DispersionCoordinateMonths[Region][:3,Month-1] = DataQuantiles[:]
                else:
                    DispersionCoordinateMonths[Region][3:,Month-1] = DataQuantiles[:]

        for Hour in range(24):
            TimeMask = DataByRegions[Region]["Hour"] == Hour
            
            MaskedVelocity = DataByRegions[Region]["Velocity"][TimeMask]
            DataQuantiles = mquantiles(array(MaskedVelocity))
            DispersionCoordinateHours[Region][:,Hour] = DataQuantiles[:]

        print("...finished!")

    with open("./WindSpeedData/WindMonthDispersion.json", "w") as OutJSON:
        json.dump(DispersionCoordinateMonths, OutJSON, cls=NumpyArrayEncoder)

    with open("./WindSpeedData/WindHourDispersion.json", "w") as OutJSON:
        json.dump(DispersionCoordinateHours, OutJSON, cls=NumpyArrayEncoder)

    return DispersionCoordinateMonths, DispersionCoordinateHours

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

    WSDispersion_Month, WSDispersion_Hour = AnalysisByCoordinate(Hours, Coordinates, WS_ListFiles)

    # MATPLOTLIB PARAMETERS
    #---------------------------------------------------------------------------------------
    use("TkAgg")
    rcParams["font.family"] = "serif"
    rcParams['savefig.dpi'] = 400
    Figure1 = figure(1, figsize=(7,6))

    # MONTH DISPERSION FIGURE
    #---------------------------------------------------------------------------------------
    Figure = figure(1, figsize=(8,6))
    Regions = WSDispersion_Month.keys()
    N = len(Regions)
    Subplots = []
    Width = 0.125
    dx = 0.5

    MonthsDay = [x + dx - Width for x in range(1,13)]
    MonthsNight = [x + dx +Width for x in range(1,13)]
    CenterHours = [Hour + 0.5 for Hour in range(24)]
    YminMonthList, YmaxMonthList = [], []
    YminHourList, YmaxHourList = [], []
    for n, Region in enumerate(Regions):
        HourIndex = 2*n+1
        MonthIndex = 2*(n+1)

        Subplots.append(Figure.add_subplot(N,2,HourIndex))
        Subplots.append(Figure.add_subplot(N,2,MonthIndex))
        Subplots[HourIndex].set_title(Region)

        # PLOTTING MONTH DISPERSION
        # ------------------------------------------------------------------------------------------------
        for x in range(1,13):
            Subplots[MonthIndex-1].axvline(x, linestyle="--", linewidth=1.0, color="black")

        HalfDispersionDay = 0.5*(WSDispersion_Month[Region][2,:]-WSDispersion_Month[Region][0,:])
        Subplots[MonthIndex-1].errorbar(MonthsDay, WSDispersion_Month[Region][1,:], yerr=HalfDispersionDay,
                             ecolor="red", color="red", markerfacecolor="red", fmt="o")

        HalfDispersionNight = 0.5*(WSDispersion_Month[Region][5,:]-WSDispersion_Month[Region][3,:])
        Subplots[MonthIndex-1].errorbar(MonthsNight, WSDispersion_Month[Region][4,:], yerr=HalfDispersionNight,
                             ecolor="blue", color="blue", markerfacecolor="blue", fmt="o")
        

        # PLOTTING HOUR DISPERSION
        # ------------------------------------------------------------------------------------------------
        Subplots[HourIndex-1].fill_between(CenterHours, WSDispersion_Hour[Region][0,:], WSDispersion_Hour[Region][2,:],
                                         color="black", alpha=0.25)
        Subplots[HourIndex-1].plot(CenterHours, WSDispersion_Hour[Region][1,:], "--k", linewidth=1)

        YminMonth, YmaxMonth = Subplots[MonthIndex-1].get_ylim()
        YminHour, YmaxHour = Subplots[HourIndex-1].get_ylim()
        YminMonthList.append(YminMonth)
        YmaxMonthList.append(YmaxMonth)
        YminHourList.append(YminHour)
        YmaxHourList.append(YmaxHour)

    minMonthY, maxMonthY = min(YminMonthList), max(YmaxMonthList)
    minHourY, maxHourY = min(YminHourList), max(YmaxHourList)
    Months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    Hours = list(range(0,25,6))

    for n in range(len(Regions)):
        HourIndex = 2*n+1
        MonthIndex = 2*(n+1)

        Subplots[MonthIndex-1].set_ylim(minMonthY, maxMonthY)
        Subplots[MonthIndex-1].set_xlim(1, 13)
        Subplots[MonthIndex-1].set_xticks(ticks=list(range(1,13)),
                               labels=Months)
        
        Subplots[HourIndex-1].set_ylim(minHourY, maxHourY)
        Subplots[HourIndex-1].set_xlim(0, 24)
        Subplots[HourIndex-1].set_xticks(ticks=Hours,
                                       labels=list(map(lambda x: str(x), Hours)))

    Figure.tight_layout()
    show()