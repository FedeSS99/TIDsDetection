from os import listdir, mkdir
from os.path import isdir
from numpy import where, concatenate
from matplotlib import rcParams
from matplotlib.pyplot import close

from DataScripts.GetDataFile import SingleTIDs_Analysis
from DataScripts.HistogramOcurrence import Time_Months_Ocurrence_Analysis
from PlottingScripts.CreatePlots import *


#Start creating Matplotlib plot to visualize the statistics given the data from DataDict
def CreateAnalysisPlots(RegionName, StationName, Stat_or_Reg, DataDict, MIN_OCC, MAX_OCC):
    if Stat_or_Reg == "Stat":
        PlotsResults = CreateFiguresResults(StationName, Stat_or_Reg)
    elif Stat_or_Reg == "Reg":
        PlotsResults = CreateFiguresResults(RegionName, Stat_or_Reg)

    Add_TimeMonthsHistogramToPlot(DataDict["OCURRENCE"], MIN_OCC, MAX_OCC, PlotsResults, 
                                  RegionName, StationName, Stat_or_Reg)

    Add_PeriodHistogramToPlot(DataDict["PERIOD"], DataDict["TIME"], DataDict["MONTH"], 
                              PlotsResults, RegionName, StationName, Stat_or_Reg)
    
    Add_BarsFreq_Month(DataDict["TIME"], DataDict["MONTH"], PlotsResults, RegionName, StationName, Stat_or_Reg)

    Add_AmplitudesAnalysis(DataDict["MIN_AMPS"], DataDict["MAX_AMPS"],
                            DataDict["TIME"], DataDict["MONTH"], PlotsResults, 
                            RegionName, StationName, Stat_or_Reg)
    

    for i in range(2,6):
        close(i)

#---------------------------------------------------------------------------
def StarAnnualAnalysis(DICT_REGION_STATIONS):
    #Ignore events that occoured in dates where a geomagnetic
    #storm had a major effect in the Dst value
    with open("./StormData/tormentas-2018-2021.txt", "r") as StormDaysData:
        StormDays = []
        Lines = StormDaysData.readlines()

        for Line in Lines[1:]:
            Date = Line.split()[0]
            if Date not in StormDays:
                StormDays.append(Line.split()[0])

    #Create RESULTS list to save data of time, period and power from TIDs
    #saved with VTEC_MainRoutine_IndividualCMN.py
    RESULTS = []
    PowerPlot = CreateResultsFigurePower()
    ListBoxPlots = []
    ListLegendsBoxPlots = []
    for Region in DICT_REGION_STATIONS.keys():
        NameOut = DICT_REGION_STATIONS[Region]["ResultPath"].split("/")[-1]
        print(f"-- Working on {NameOut} region --")
        
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

        #Obtain the full path of the files located in each station given the Region
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
            Dates_TIDs = [fileName.split(".")[0].split("/")[-1][-15:-5] for fileName in TIDs_DataPaths]
            MonthPerFile = [int(fileName.split("/")[-1].split("-")[2]) for fileName in TIDs_DataPaths]

            TotalDays += len(TIDs_DataPaths)
            for fileTID, MonthFile, Date_TID in zip(TIDs_DataPaths, MonthPerFile, Dates_TIDs):
                if Date_TID not in StormDays:
                    Results = SingleTIDs_Analysis(fileTID)
                    SizeResults = Results["TIME"].size
                    if SizeResults:
                        ActiveDays += 1

                        Region_MonthArray.append(SizeResults*[MonthFile])
                        Station_MonthArray.append(SizeResults*[MonthFile])

                        #Get the timezone given NameOut
                        if NameOut == "North":
                            TimeZone = -8.0
                        elif NameOut == "Center":
                            TimeZone = -6.0
                        elif NameOut == "South":
                            TimeZone = -5.0

                        #Apply timezone to get correct Local Time Hours
                        Results["TIME"] += TimeZone
                        Results["TIME"] = where(Results["TIME"] < 0, Results["TIME"] + 24.0, Results["TIME"])
                        Region_TimeTID.append(Results["TIME"])
                        Station_TimeTID.append(Results["TIME"])

                        Region_PeriodTID.append(Results["PERIOD"])
                        Station_PeriodTID.append(Results["PERIOD"])

                        Region_PowerTID.append(Results["POWER"])
                        Station_PowerTID.append(Results["POWER"])
                        
                        Region_MinAmps.append(Results["MIN_AMPS"])
                        Station_MinAmps.append(Results["MIN_AMPS"])
                        
                        Region_MaxAmps.append(Results["MAX_AMPS"])
                        Station_MaxAmps.append(Results["MAX_AMPS"])


            Station_MonthArray = concatenate(tuple(Station_MonthArray), dtype=int)
            Station_TimeTID = concatenate(tuple(Station_TimeTID))
            Station_PeriodTID = concatenate(tuple(Station_PeriodTID))
            Station_PowerTID = concatenate(tuple(Station_PowerTID))
            Station_MinAmps = concatenate(tuple(Station_MinAmps))
            Station_MaxAmps = concatenate(tuple(Station_MaxAmps))

            StationOcurrenceMap = Time_Months_Ocurrence_Analysis(Station_TimeTID, Station_MonthArray)

            StationResultsDict = {
                "TIME":Station_TimeTID,
                "MONTH":Station_MonthArray,
                "OCURRENCE":StationOcurrenceMap,
                "PERIOD":Station_PeriodTID,
                "POWER":Station_PowerTID,
                "MIN_AMPS":Station_MinAmps,
                "MAX_AMPS":Station_MaxAmps,
            }

            StationSavedir = f"./../Results/{Region}/{Station}/"
            if not isdir(StationSavedir):
                mkdir(StationSavedir)

            MIN_OCC, MAX_OCC = StationResultsDict["OCURRENCE"].min(), StationResultsDict["OCURRENCE"].max()
            CreateAnalysisPlots(Region, Station, "Stat", StationResultsDict, MIN_OCC, MAX_OCC)
            

        # Join all stations data from the same region in single arrays
        Region_MonthArray = concatenate(tuple(Region_MonthArray), dtype=int)
        Region_TimeTID = concatenate(tuple(Region_TimeTID))
        Region_PeriodTID = concatenate(tuple(Region_PeriodTID))
        Region_PowerTID = concatenate(tuple(Region_PowerTID))
        Region_MinAmps = concatenate(tuple(Region_MinAmps))
        Region_MaxAmps = concatenate(tuple(Region_MaxAmps))
        NumTIDs = Region_TimeTID.size

        #Get 2D histogram of TIDs' ocurrence and save all the data from the respective directory
        #in RESULTS
        HistogramOcurrence = Time_Months_Ocurrence_Analysis(Region_TimeTID, Region_MonthArray)

        RegionResultsDict = {
            "TIME":Region_TimeTID,
            "MONTH":Region_MonthArray,
            "OCURRENCE":HistogramOcurrence,
            "PERIOD":Region_PeriodTID,
            "POWER":Region_PowerTID,
            "MIN_AMPS":Region_MinAmps,
            "MAX_AMPS":Region_MaxAmps,
            "NAME":NameOut
        }

        RESULTS.append(RegionResultsDict)

        print(f"Total Days:{TotalDays}\nNo. of TIDs:{NumTIDs}\nActive Days:{ActiveDays}\n")

    #Obtain the globam minimum and maximum of the ocurrence arrays of all the directories' data
    MIN_OCC = min([DataResults["OCURRENCE"].min() for DataResults in RESULTS])
    MAX_OCC = max([DataResults["OCURRENCE"].max() for DataResults in RESULTS])
    for RegionDataResults in RESULTS:
        #Get a string Coord to use in the analysis' plots results
        NamePlot = RegionDataResults["NAME"]
        
        if NamePlot == "North":
            NamePower = "PTEX-PALX"
            ColorPower = "green"
            RegIndex = 0
        elif NamePlot == "Center":
            NamePower = "MNIG-UCOE"
            ColorPower = "red"
            RegIndex = 1
        elif NamePlot == "South":
            NamePower = "CN24-UNPM"
            ColorPower = "blue"
            RegIndex = 2

        CreateAnalysisPlots(NamePlot, "", "Reg", RegionDataResults, MIN_OCC, MAX_OCC)

        BoxPlotObject = Add_TimePowerDataResultsToPlot(RegionDataResults["TIME"], RegionDataResults["POWER"], PowerPlot, ColorPower, RegIndex)
        ListBoxPlots.append(BoxPlotObject)
        ListLegendsBoxPlots.append(NamePower)
    
    PowerPlot[1].set_yscale("log", subs=None)
    PowerPlot[0].legend(ListBoxPlots, ListLegendsBoxPlots, loc="upper right")
    SaveRegionPlot("PowerDistributionStations", "", PowerPlot[0])
    PowerPlot[0].savefig(f"./../Results/PowerDistributionStations.png")
    close(1)
#---------------------------------------------------------------------------

def CreateInputDictionary(SubdirsResultsData, DataPath, ResultsPath):
    Dictionary = dict()

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
                
                    StationDataPaths += list(map(lambda x: NewAnalysisPath + "/" + x, listdir(NewAnalysisPath)))
            
            StationsDict[station] = StationDataPaths
        
        Dictionary[Region] = dict(ResultPath = ResultsRegionPath, DataPaths = StationsDict)

    return Dictionary



if __name__=="__main__":
    # Setting plotting format for all figures
    rcParams["font.family"] = "serif"
    rcParams["savefig.dpi"] = 400

    DATA_COMMON_PATH = "../Analysis/"
    RESULTS_COMMON_PATH = "../Results/"

    SUBDIRECTORIES_REGIONS = ["North", "Center", "South"]

    InputRegionsData = CreateInputDictionary(SUBDIRECTORIES_REGIONS, DATA_COMMON_PATH, RESULTS_COMMON_PATH)
    
    StarAnnualAnalysis(InputRegionsData)
