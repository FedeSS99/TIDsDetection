from os import listdir
from numpy import ndarray, array, zeros, linspace, arange, loadtxt, ndarray
from scipy.stats.mstats import mquantiles
from matplotlib.pyplot import figure, colorbar, show
from matplotlib import rcParams, use
import json

import warnings

from DataScripts.WSDataRoutines import ReadWSFile


# Encoder class to save numpy arrays as lists in json files
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
# Dictionary to extract filename of Terminator data for each region
TerminatorsDict = dict(
    North="./TerminatorData/TerminatorHours_North.dat",
    Center="./TerminatorData/TerminatorHours_Center.dat",
    South="./TerminatorData/TerminatorHours_South.dat"
    )

# Function to get all data in a zone given the latitudes and longitudes
# [IntLat-1, IntLat, IntLat+1], [IntLong-1, IntLong, IntLong+1]
def ExtractData(Latitudes:list[int], Longitudes:list[int], FilesList:list[str])-> dict:
    DataInZone = dict(
        Year = [],
        Month = [],
        Hour = [],
        Velocity = [],
        VelocityDir = []
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
                            DataInZone["VelocityDir"] += DataInFile[4]

    for Field in DataInZone.keys():
        DataInZone[Field] = array(DataInZone[Field])
    
    return DataInZone

def AnalysisByCoordinate(Hours:list[int], Coordinates:dict, Field:str, FilesList:list[str]) -> dict:
    DataByRegions = dict()
    for Region in Coordinates.keys():
        LatitudesInRegion = Coordinates[Region]["Lat"]
        LongitudesInRegion = Coordinates[Region]["Long"]
        
        DataInRegion = ExtractData(LatitudesInRegion, LongitudesInRegion, FilesList)
        DataByRegions[Region] = DataInRegion

    DispersionCoordinateMonths = dict()
    DispersionCoordinateHours = dict()  
    DispersionCoordinateMonthHour = dict()
    for Region in Coordinates.keys():
        DispersionCoordinateMonths[Region] = zeros((6,12), dtype=float)
        DispersionCoordinateHours[Region] = zeros((3,24), dtype=float)
        DispersionCoordinateMonthHour[Region] = zeros((12, 24), dtype=float)

        # Analysis per month given integer Hours
        for Month in range(1,13):
            MonthMask = DataByRegions[Region]["Month"] == Month

            for n, Time in enumerate(Hours):
                TimeMask = DataByRegions[Region]["Hour"] == Time

                MaskedField = DataByRegions[Region][Field][MonthMask*TimeMask]

                DataQuantiles = mquantiles(MaskedField)
                if n: 
                    DispersionCoordinateMonths[Region][:3,Month-1] = DataQuantiles[:]
                else:
                    DispersionCoordinateMonths[Region][3:,Month-1] = DataQuantiles[:]

        # Analysis per hour (local time)
        for Hour in range(24):
            TimeMask = DataByRegions[Region]["Hour"] == Hour
            
            MaskedField = DataByRegions[Region][Field][TimeMask]
            DataQuantiles = mquantiles(MaskedField)
            DispersionCoordinateHours[Region][:,Hour] = DataQuantiles[:]

        # Analysis of dispersion per month and local time
        for Month in range(1,13):
            MonthMask = DataByRegions[Region]["Month"] == Month
            for Hour in range(24):
                HourMask = DataByRegions[Region]["Hour"] == Hour
            
                MaskedField = DataByRegions[Region][Field][MonthMask*HourMask]
                DataQuantiles = mquantiles(MaskedField)
                DispersionCoordinateMonthHour[Region][Month-1,Hour] = DataQuantiles[2] - DataQuantiles[0]

    with open(f"./WindSpeedData/{Field}MonthDispersion.json", "w") as OutJSON:
        json.dump(DispersionCoordinateMonths, OutJSON, cls=NumpyArrayEncoder)

    with open(f"./WindSpeedData/{Field}HourDispersion.json", "w") as OutJSON:
        json.dump(DispersionCoordinateHours, OutJSON, cls=NumpyArrayEncoder)

    with open(f"./WindSpeedData/{Field}MonthHourDispersion.json", "w") as OutJSON:
        json.dump(DispersionCoordinateMonthHour, OutJSON, cls=NumpyArrayEncoder)

    return DispersionCoordinateMonths, DispersionCoordinateHours, DispersionCoordinateMonthHour

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

    # MATPLOTLIB PARAMETERS
    #---------------------------------------------------------------------------------------
    use("TkAgg")
    rcParams["font.family"] = "serif"
    rcParams['savefig.dpi'] = 400

    Fields = ["Velocity", "VelocityDir"]
    Labels = ["Velocity (m/s)", "Orientation (Deg.)"]
    SaveNames = [("../Results/WindSpeedVariations", "../Results/WindSpeedVariationMaps"),
                 ("../Results/WindDirectionVariations", "../Results/WindDirectionVariationMaps") ]
    for Field, Label, SaveName in zip(Fields, Labels, SaveNames):
        print(f"Starting {Label.split(' ')[0]} analysis...", end="")
        WSDispersion_Month, WSDispersion_Hour, WSDispersion_HourMonth = AnalysisByCoordinate(Hours, Coordinates, Field, WS_ListFiles)

        # MONTH DISPERSION FIGURE
        #---------------------------------------------------------------------------------------
        Figure1 = figure(1, figsize=(8,6))
        Figure2 = figure(2, figsize=(6,6))
        Regions = WSDispersion_Month.keys()
        N = len(Regions)
        Subplots1 = []
        Subplots2 = []
        Width = 0.125
        dx = 0.5

        MonthsDay = [x + dx - Width for x in range(1,13)]
        MonthsNight = [x + dx +Width for x in range(1,13)]
        CenterHours = [Hour + 0.5 for Hour in range(24)]
        YminMonthList, YmaxMonthList = [], []
        YminHourList, YmaxHourList = [], []
        MinHourMonth = min([WSDispersion_HourMonth[Region].min() for Region in Regions]) 
        MaxHourMonth = max([WSDispersion_HourMonth[Region].max() for Region in Regions])

        for n, Region in enumerate(Regions):
            HourIndex = 2*n+1
            MonthIndex = 2*(n+1)

            Subplots1.append(Figure1.add_subplot(N,2,HourIndex))
            Subplots1.append(Figure1.add_subplot(N,2,MonthIndex))
            Subplots2.append(Figure2.add_subplot(2,2,n+1))
            Subplots1[HourIndex-1].set_title(Region)
            Subplots1[HourIndex-1].set_ylabel(Label)
            Subplots2[n].set_title(Region)

            # PLOTTING MONTH DISPERSION
            # ------------------------------------------------------------------------------------------------
            for x in range(1,13):
                Subplots1[MonthIndex-1].axvline(x, linestyle="--", linewidth=1.0, color="black")

            HalfDispersionDay = 0.5*(WSDispersion_Month[Region][2,:]-WSDispersion_Month[Region][0,:])
            Subplots1[MonthIndex-1].errorbar(MonthsDay, WSDispersion_Month[Region][1,:], yerr=HalfDispersionDay,
                                ecolor="red", color="red", markerfacecolor="red", fmt="o")

            HalfDispersionNight = 0.5*(WSDispersion_Month[Region][5,:]-WSDispersion_Month[Region][3,:])
            Subplots1[MonthIndex-1].errorbar(MonthsNight, WSDispersion_Month[Region][4,:], yerr=HalfDispersionNight,
                                ecolor="blue", color="blue", markerfacecolor="blue", fmt="o")

            YminMonth, YmaxMonth = Subplots1[MonthIndex-1].get_ylim()
            YminMonthList.append(YminMonth)
            YmaxMonthList.append(YmaxMonth)
            
            # PLOTTING HOUR DISPERSION
            # ------------------------------------------------------------------------------------------------
            Subplots1[HourIndex-1].fill_between(CenterHours, WSDispersion_Hour[Region][0,:], WSDispersion_Hour[Region][2,:],
                                            color="black", alpha=0.25)
            Subplots1[HourIndex-1].plot(CenterHours, WSDispersion_Hour[Region][1,:], "--k", linewidth=1)

            YminHour, YmaxHour = Subplots1[HourIndex-1].get_ylim()
            YminHourList.append(YminHour)
            YmaxHourList.append(YmaxHour)

            # PLOTTING HOUR-MONTH DISPERSION
            # ------------------------------------------------------------------------------------------------
            HourMonthImage = Subplots2[n].imshow(WSDispersion_HourMonth[Region], cmap="plasma", aspect="auto", 
                                                origin="lower", extent=[0.0, 24.0, 0, 12],
                                                vmin=MinHourMonth, vmax=MaxHourMonth)

            # Extracting rise and set hours for each region
            RiseHours, SetHours = loadtxt(TerminatorsDict[Region.split("_")[0]], dtype=float,
                                        usecols=(1, 2), unpack=True, skiprows=1)

            NumMonthTerminator = linspace(0.0, 12.0, RiseHours.size)
            Subplots2[n].plot(RiseHours, NumMonthTerminator, "--k", linewidth=1.0)
            Subplots2[n].plot(SetHours, NumMonthTerminator, "--k", linewidth=1.0)

        # FORMAT FOR DISPERSION PLOTS
        # ------------------------------------------------------------------------------------------------

        minMonthY, maxMonthY = min(YminMonthList), max(YmaxMonthList)
        minHourY, maxHourY = min(YminHourList), max(YmaxHourList)
        Months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        Hours = list(range(0,25,6))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Figure1.tight_layout()
            Figure2.tight_layout()

        for n in range(len(Regions)):
            HourIndex = 2*n+1
            MonthIndex = 2*(n+1)

            Subplots1[MonthIndex-1].set_ylim(minMonthY, maxMonthY)
            Subplots1[MonthIndex-1].set_xlim(1, 13)
            Subplots1[MonthIndex-1].set_xticks(ticks=list(range(1,13)),
                                labels=Months)
            
            Subplots1[HourIndex-1].set_ylim(minHourY, maxHourY)
            Subplots1[HourIndex-1].set_xlim(0, 24)
            Subplots1[HourIndex-1].set_xticks(ticks=Hours,
                                        labels=list(map(lambda x: str(x), Hours)))
            
            Subplots2[n].set_yticks(linspace(0.5, 11.5, 12, endpoint=True), Months)
            Subplots2[n].set_xticks(arange(0, 25, 3))
            SubplotBox_n0 = Subplots2[n].get_position()
            Subplots2[n].set_position([SubplotBox_n0.x0, SubplotBox_n0.y0,
                        0.75*SubplotBox_n0.width, SubplotBox_n0.height])

            if n % 2 == 0:
                SubplotBox_n0 = Subplots2[n].get_position()
                SubplotBox_n1 = Subplots2[n+1].get_position()

                SeparationBoxes = SubplotBox_n1.x0 - (SubplotBox_n0.x0 + SubplotBox_n0.width)

                Subplots2[n+1].set_position([SubplotBox_n1.x0 - 0.5*SeparationBoxes, SubplotBox_n1.y0,
                                        SubplotBox_n1.width, SubplotBox_n1.height])

        Colorbar_Subplots2 = Figure2.add_axes([0.8, 0.15, 0.05, 0.7])
        colorbar(HourMonthImage, cax=Colorbar_Subplots2, label=f"IQR-{Label}")

        for Format in [".png", ".pdf"]:
            Figure1.savefig(SaveName[0] + Format)
            Figure2.savefig(SaveName[1] + Format)

        print("finished!")
        Figure1.clear()
        Figure2.clear()