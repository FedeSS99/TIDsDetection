from os import listdir, mkdir
from os.path import isdir
from numpy import where, concatenate
from matplotlib import rcParams
from matplotlib.pyplot import close

from DataScripts.GetDataFile import SingleTIDs_Analysis
from DataScripts.HistogramOcurrence import Time_Months_Ocurrence_Analysis
from PlottingScripts.CreatePlots import *

# Start creating plots for either Region and Station to visualize the statistics given the data from DataDict


def CreateAnalysisPlots(RegionName, StationName, Stat_or_Reg, DataDict, CMAP, NORM):
    if Stat_or_Reg == "Stat":
        PlotsResults = CreateFiguresResults(StationName, Stat_or_Reg)
        print(f"Working on {StationName} Station...", end="\t")

    elif Stat_or_Reg == "Reg":
        PlotsResults = CreateFiguresResults(RegionName, Stat_or_Reg)
        print(f"Working on {RegionName} Region...", end="\t")

    Add_TimeMonthsHistogramToPlot(DataDict["OCURRENCE"], CMAP, NORM, PlotsResults,
                                  RegionName, StationName, Stat_or_Reg)

    Add_PeriodHistogramToPlot(DataDict["PERIOD"], DataDict["TIME"], DataDict["MONTH"],
                              PlotsResults, RegionName, StationName, Stat_or_Reg)

    Add_BarsFreq_Month(DataDict["TIME"], DataDict["MONTH"],
                       PlotsResults, RegionName, StationName, Stat_or_Reg)

    for i in range(4, 7):
        PlotsResults[0][i-4].tight_layout()
        close(i)

    print("finished!")

# ---------------------------------------------------------------------------


def StarAnnualAnalysis(DICT_REGION_STATIONS):
    # Ignore events that occoured in dates where a geomagnetic
    # storm had a major effect in the Dst value
    with open("./StormData/tormentas-2018-2021.txt", "r") as StormDaysData:
        StormDays = []
        Lines = StormDaysData.readlines()

        for Line in Lines[1:]:
            Date = Line.split()[0]
            if Date not in StormDays:
                StormDays.append(Line.split()[0])

    # Create dictionary for atributes to use as input information for each Region
    # plot
    RegionsInfo = {
        "North": ["blue", "^", 0],
        "Center": ["green", "*", 1],
        "South": ["red", "s", 2]
    }

    # Create RESULTS list to save data of time, period and power from TIDs
    # saved with VTEC_MainRoutine_IndividualCMN.py
    RESULTS = []
    Nplots = len(RegionsInfo)
    PowerPlot = CreateFigureTimePower()
    AmplitudePowerPlot = CreateFigureAmplitudePower(Nplots)
    AmplitudeVarPlot = CreateFigureAmplitudeVar(Nplots)

    ListBoxPlots = []
    ListLegendsBoxPlots = []
    OcurrenceSampling_AllRegions = []
    for Region in DICT_REGION_STATIONS.keys():
        NameOut = DICT_REGION_STATIONS[Region]["ResultPath"].split("/")[-1]
        print(f"-- Extracting data of {NameOut} Region --")

        # Declare counter for total days and active days for each region and
        # also the lists to save the data for all stations
        ActiveDays = 0
        TotalDays = 0
        Region_TimeTID = []
        Region_PeriodTID = []
        Region_PowerTID = []
        Region_MinAmps = []
        Region_MaxAmps = []
        Region_MonthArray = []

        # Obtain the full path of the files located in each station given the Region
        StationsByRegion = DICT_REGION_STATIONS[Region]["DataPaths"].keys()
        for Station in StationsByRegion:

            # Declare the lists to save the data for each station and compute
            # the statistics for it
            Station_TimeTID = []
            Station_PeriodTID = []
            Station_PowerTID = []
            Station_MinAmps = []
            Station_MaxAmps = []
            Station_MonthArray = []

            # Extract DataPaths for each Station and also date and month data
            TIDs_DataPaths = DICT_REGION_STATIONS[Region]["DataPaths"][Station]
            Dates_TIDs = [fileName.split(".")[0].split(
                "/")[-1][-15:-5] for fileName in TIDs_DataPaths]
            MonthPerFile = [int(fileName.split("/")[-1].split("-")[2])
                            for fileName in TIDs_DataPaths]

            TotalDays += len(TIDs_DataPaths)
            for fileTID, MonthFile, Date_TID in zip(TIDs_DataPaths, MonthPerFile, Dates_TIDs):
                if Date_TID not in StormDays:
                    Results = SingleTIDs_Analysis(fileTID)
                    SizeResults = Results["TIME"].size
                    if SizeResults:
                        ActiveDays += 1

                        Station_MonthArray.append(SizeResults*[MonthFile])

                        # Get the timezone given NameOut
                        if NameOut == "North":
                            TimeZone = -8.0
                        elif NameOut == "Center":
                            TimeZone = -6.0
                        elif NameOut == "South":
                            TimeZone = -5.0

                        # Apply timezone to get correct Local Time Hours
                        Results["TIME"] += TimeZone
                        Results["TIME"] = where(
                            Results["TIME"] < 0, Results["TIME"] + 24.0, Results["TIME"])
                        Station_TimeTID.append(Results["TIME"])
                        Station_PeriodTID.append(Results["PERIOD"])
                        Station_PowerTID.append(Results["POWER"])
                        Station_MinAmps.append(Results["MIN_AMPS"])
                        Station_MaxAmps.append(Results["MAX_AMPS"])

            # Create numpy arrays from the tuple collections of all the TIDs
            # data for each Station
            Station_MonthArray = concatenate(
                tuple(Station_MonthArray), dtype=int)
            Station_TimeTID = concatenate(tuple(Station_TimeTID))
            Station_PeriodTID = concatenate(tuple(Station_PeriodTID))
            Station_PowerTID = concatenate(tuple(Station_PowerTID))
            Station_MinAmps = concatenate(tuple(Station_MinAmps))
            Station_MaxAmps = concatenate(tuple(Station_MaxAmps))

            # Obtain the ocurrence map for the TIDs in this same Station
            StationOcurrenceMap = Time_Months_Ocurrence_Analysis(
                Station_TimeTID, Station_MonthArray)

            # Save all the Station data in one dictionary and...
            StationResultsDict = {
                "TIME": Station_TimeTID,
                "MONTH": Station_MonthArray,
                "OCURRENCE": StationOcurrenceMap,
                "PERIOD": Station_PeriodTID,
                "POWER": Station_PowerTID,
                "MIN_AMPS": Station_MinAmps,
                "MAX_AMPS": Station_MaxAmps,
            }

            # and save these numpy arrays in the correspondent list to manage
            # all the TIDs data for each Region
            Region_MonthArray.append(StationResultsDict["MONTH"])
            Region_TimeTID.append(StationResultsDict["TIME"])
            Region_PeriodTID.append(StationResultsDict["PERIOD"])
            Region_PowerTID.append(StationResultsDict["POWER"])
            Region_MinAmps.append(StationResultsDict["MIN_AMPS"])
            Region_MaxAmps.append(StationResultsDict["MAX_AMPS"])

            # Check if the directory for the Station given its' Region exists,
            # and if not, create it
            StationSavedir = f"./../Results/{Region}/{Station}/"
            if not isdir(StationSavedir):
                mkdir(StationSavedir)

            # Save minimum and maximum values of local ocurrence of TIDs for each Station
            CMAP, NORM = ObtainCMAPandNORM(
                StationResultsDict["OCURRENCE"].flatten())
            CreateAnalysisPlots(Region, Station, "Stat",
                                StationResultsDict, CMAP, NORM)

        # Create numpy arrays for all Stations data from the same Region
        Region_MonthArray = concatenate(tuple(Region_MonthArray), dtype=int)
        Region_TimeTID = concatenate(tuple(Region_TimeTID))
        Region_PeriodTID = concatenate(tuple(Region_PeriodTID))
        Region_PowerTID = concatenate(tuple(Region_PowerTID))
        Region_MinAmps = concatenate(tuple(Region_MinAmps))
        Region_MaxAmps = concatenate(tuple(Region_MaxAmps))
        NumTIDs = Region_TimeTID.size

        # Get ocurrence map for each Region
        HistogramOcurrence = Time_Months_Ocurrence_Analysis(
            Region_TimeTID, Region_MonthArray)
        
        # Get average absolute amplitude for each region and all the correspond data in the directory
        # in a dictionary
        AveAbsAmplitude = (np.abs(Region_MinAmps) + Region_MaxAmps)/2.0

        RegionResultsDict = {
            "TIME": Region_TimeTID,
            "MONTH": Region_MonthArray,
            "OCURRENCE": HistogramOcurrence,
            "PERIOD": Region_PeriodTID,
            "POWER": Region_PowerTID,
            "AVE_AMP": AveAbsAmplitude,
            "NAME": NameOut
        }

        # Save a flatten version of the ocurrence map of the Region
        OcurrenceSampling_AllRegions.append(HistogramOcurrence.flatten())

        RESULTS.append(RegionResultsDict)

        print(
            f"Total Days:{TotalDays}\nNo. of TIDs:{NumTIDs}\nActive Days:{ActiveDays}\nTIDs-Active Day ratio: {NumTIDs/ActiveDays:.3f}\n")

    # Obtain ColorMap and Norm to use for all the regions
    OcurrenceSampling_AllRegions = np.concatenate(
        tuple(OcurrenceSampling_AllRegions))
    CMAP, NORM = ObtainCMAPandNORM(OcurrenceSampling_AllRegions)

    for RegionDataResults in RESULTS:
        # Get a string Coord to use in the analysis' plots results
        NamePlot = RegionDataResults["NAME"]

        RegionInfoData = RegionsInfo[NamePlot]

        # Generate graphics for each region
        CreateAnalysisPlots(NamePlot, "", "Reg", RegionDataResults, CMAP, NORM)

        # Add boxplots for time-power data in a figure for all regions
        BoxPlotObject = Add_TimePowerDataResultsToPlot(RegionDataResults["TIME"], RegionDataResults["POWER"],
                                                       PowerPlot, RegionInfoData[0], RegionInfoData[2])

        ListBoxPlots.append(BoxPlotObject)
        ListLegendsBoxPlots.append(NamePlot)

        # Add boxplots for time-amplitude data in a figure for all regions
        Add_AmplitudesAnalysis(RegionDataResults["AVE_AMP"], RegionDataResults["TIME"], 
                               RegionDataResults["MONTH"], AmplitudeVarPlot, RegionInfoData[2], NamePlot)

        # Add scatter plots of amplitude-power data in a figure for all regions
        Add_AmplitudePowerScatterPlot(RegionDataResults["AVE_AMP"], RegionDataResults["POWER"],
                                      RegionDataResults["TIME"], RegionDataResults["MONTH"], AmplitudePowerPlot,
                                      RegionInfoData[1], RegionInfoData[2], NamePlot)

    # Apply logaritmic scale to Power variance plot and save the figure
    PowerPlot[1].set_yscale("log")
    PowerPlot[0].legend(ListBoxPlots, ListLegendsBoxPlots,
                        loc="upper right", fancybox=True, shadow=True)
    SaveRegionPlot("PowerDistributionStations", "", PowerPlot[0])

    # Create Hours and Months ticks to use change the labels in Amplitude variance
    # plots and save the figure
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May",
                  "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    HOUR_TICKS = [i for i in range(0, 25, 4)]
    MonthAxisData = np.linspace(1.0, 12.0, 12, endpoint=True)
    MinAmps_Plots, MaxAmps_Plots = [], []
    for p in range(Nplots):
        if p < Nplots - 1:
            AmplitudeVarPlot[1][p][0].set_xticks(ticks=[], labels=[])
            AmplitudeVarPlot[1][p][1].set_xticks(ticks=[], labels=[])

        if p == Nplots - 1:
            # Setting x ticks within 24 hours
            AmplitudeVarPlot[1][p][0].set_xticks(
                ticks=HOUR_TICKS, labels=HOUR_TICKS)
            # Setting x ticks with months names
            AmplitudeVarPlot[1][p][1].set_xticks(
                ticks=MonthAxisData, labels=MonthTicks)

            # Align to the right and rotate the x-axis labels of the bottom 2nd column
            for label in AmplitudeVarPlot[1][p][1].get_xticklabels():
                label.set_horizontalalignment('left')
            AmplitudeVarPlot[1][p][1].set_xticklabels(
                MonthTicks, rotation=-45)

        Y_MIN, Y_MAX = AmplitudeVarPlot[1][p][0].get_ylim()
        MinAmps_Plots.append(Y_MIN)
        MaxAmps_Plots.append(Y_MAX)

    # Set the same min and max value for all Amplitude variance plots
    Y_MIN, Y_MAX = min(MinAmps_Plots), max(MaxAmps_Plots)
    for p in range(Nplots):
        for l in range(2):
            AmplitudeVarPlot[1][p][l].set_ylim(Y_MIN, Y_MAX)
        AmplitudeVarPlot[1][p][1].set_yticks(ticks=[], labels=[])

    SaveRegionPlot("AmplitudeVariations", "", AmplitudeVarPlot[0])

    # Apply logaritmic scale to x-axis and y-axis in Amplitude-Power plot
    for p in range(Nplots):
        AmplitudePowerPlot[1][p].set_yscale("log", subs=None)
        # Reduce length size of each subplot by 20%
        SubplotBox = AmplitudePowerPlot[1][p].get_position()
        AmplitudePowerPlot[1][p].set_position([SubplotBox.x0, SubplotBox.y0,
                                               SubplotBox.width*0.8, SubplotBox.height])
    AmplitudePowerPlot[1][Nplots-1].set_xscale("log", subs=None)

    # Fixing position of legends box outside the subplots
    AmplitudePowerPlot[0].legend(fancybox=True, shadow=True, bbox_to_anchor=(1.0, 0.625))
    SaveRegionPlot("AmplitudePowerRegions", "", AmplitudePowerPlot[0])

    for s in range(1, 3):
        close(s)
# ---------------------------------------------------------------------------


def CreateInputDictionary(SubdirsResultsData, DataPath, ResultsPath):
    inputDictionary = dict()

    for Region in SubdirsResultsData:
        DataRegionPath = DataPath + Region
        ResultsRegionPath = ResultsPath + Region

        StationsDict = dict()

        for station in listdir(DataRegionPath):
            StationDataPaths = []

            CompleteStationDir = DataRegionPath + "/" + station
            if not isdir(CompleteStationDir):
                continue

            for yearDir in listdir(CompleteStationDir):
                CompleteYearDir = CompleteStationDir + "/" + yearDir
                if not isdir(CompleteYearDir) or not yearDir.isnumeric():
                    continue

                for monthDir in listdir(CompleteYearDir):
                    NewAnalysisPath = CompleteYearDir + "/" + monthDir
                    if not isdir(NewAnalysisPath) or not monthDir.isnumeric():
                        continue

                    StationDataPaths += list(map(lambda x: NewAnalysisPath +
                                             "/" + x, listdir(NewAnalysisPath)))

            StationsDict[station] = StationDataPaths

        inputDictionary[Region] = dict(
            ResultPath=ResultsRegionPath, DataPaths=StationsDict)

    return inputDictionary


if __name__ == "__main__":
    # Setting plotting format for all figures
    rcParams["font.family"] = "serif"
    rcParams["savefig.dpi"] = 400

    DATA_COMMON_PATH = "../Analysis/"
    RESULTS_COMMON_PATH = "../Results/"

    SUBDIRECTORIES_REGIONS = ["North", "Center", "South"]

    InputRegionsData = CreateInputDictionary(
        SUBDIRECTORIES_REGIONS, DATA_COMMON_PATH, RESULTS_COMMON_PATH)

    StarAnnualAnalysis(InputRegionsData)
