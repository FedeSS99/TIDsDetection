from tqdm import tqdm
from os import listdir
from os.path import isdir
from numpy import where, concatenate
from matplotlib import rcParams
from matplotlib.pyplot import close

from DataScripts.GetDataFile import SingleTIDs_Analysis
from DataScripts.HistogramOcurrence import Time_Months_Ocurrence_Analysis
from PlottingScripts.CreatePlots import *

#---------------------------------------------------------------------------
def StarAnnualAnalysis(DICT_REGION_STATIONS):
    #Ignore events that occoured in dates where a geomagnetic
    #storm had a major effect in the Dst value
    with open("tormentas-2018-2021.txt", "r") as StormDaysData:
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
        
        ActiveDays = 0
        TotalDays = 0
        ResultsTimeTID = []
        ResultsPeriodTID = []
        ResultsPowerTID = []
        ResultsMinAmps = []
        ResultsMaxAmps = []
        MonthArray = []

        #Obtain the full path of the files located in the selected directories and
        #the dates and months of these same files
        TIDs_DataPaths = DICT_REGION_STATIONS[Region]["DataPaths"]
        
        Dates_TIDs = [fileName.split(".")[0].split("/")[-1][-15:-5] for fileName in TIDs_DataPaths]
        MonthPerFile = [int(fileName.split("/")[-1].split("-")[2]) for fileName in TIDs_DataPaths]

        TotalDays += len(TIDs_DataPaths)
        for fileTID, MonthFile, Date_TID in tqdm(zip(TIDs_DataPaths, MonthPerFile, Dates_TIDs)):
            if Date_TID not in StormDays:
                Results = SingleTIDs_Analysis(fileTID)
                SizeResults = Results[0].size
                if SizeResults:
                    ActiveDays += 1

                    MonthArray.append(SizeResults*[MonthFile])
                    ResultsTimeTID.append(Results[0])
                    ResultsPeriodTID.append(Results[1])
                    ResultsPowerTID.append(Results[2])
                    ResultsMinAmps.append(Results[3])
                    ResultsMaxAmps.append(Results[4])

        MonthArray = concatenate(tuple(MonthArray), dtype=int)
        ResultsTimeTID = concatenate(tuple(ResultsTimeTID))
        #Get the timezone given NameOut
        if NameOut == "Center":
            TimeZone = -6.0
        elif NameOut == "North":
            TimeZone = -8.0
        elif NameOut == "South":
            TimeZone = -5.0
        #Apply timezone to get correct Local Time Hours
        ResultsTimeTID += TimeZone
        ResultsTimeTID = where(ResultsTimeTID < 0, ResultsTimeTID + 24.0, ResultsTimeTID)

        ResultsPeriodTID = concatenate(tuple(ResultsPeriodTID))
        ResultsPowerTID = concatenate(tuple(ResultsPowerTID))
        ResultsMinAmps = concatenate(tuple(ResultsMinAmps))
        ResultsMaxAmps = concatenate(tuple(ResultsMaxAmps))
        NumTIDs = ResultsTimeTID.size

        #Get 2D histogram of TIDs' ocurrence and save all the data from the respective directory
        #in RESULTS
        HistogramOcurrence = Time_Months_Ocurrence_Analysis(ResultsTimeTID, MonthArray)
        RESULTS.append([ResultsTimeTID, MonthArray, HistogramOcurrence, ResultsPeriodTID, ResultsPowerTID, ResultsMinAmps, ResultsMaxAmps, NameOut])

        print(f"No. of TIDs:{NumTIDs}\nActive Days:{ActiveDays} Total Days:{TotalDays}\n")

    #Obtain the globam minimum and maximum of the ocurrence arrays of all the directories' data
    MIN, MAX = min([DataResults[2].min() for DataResults in RESULTS]), max([DataResults[2].max() for DataResults in RESULTS])
    for DataResults in RESULTS:
        #Get a string Coord to use in the analysis' plots results
        NamePlot = DataResults[-1]
        
        if NamePlot == "Center":
            NamePower = "LNIG-MNIG-UCOE"
            ColorPower = "red"
            RegIndex = 0
        elif NamePlot == "North":
            NamePower = "PTEX"
            ColorPower = "green"
            RegIndex = 1
        elif NamePlot == "South":
            NamePower = "CN24"
            ColorPower = "blue"
            RegIndex = 2

        #Start creating Matplotlib plot to visualize the statistics given the data from RESULTS
        PlotsResults = CreateFiguresResults(NamePlot)
        Add_TimeMonthsHistogramToPlot(DataResults[2], MIN, MAX, PlotsResults, NamePlot)
        Add_PeriodHistogramToPlot(DataResults[3], DataResults[0], DataResults[1], PlotsResults, NamePlot)
        BoxPlotObject = Add_TimePowerDataResultsToPlot(DataResults[0], DataResults[4], PowerPlot, ColorPower, RegIndex)
        ListBoxPlots.append(BoxPlotObject)
        ListLegendsBoxPlots.append(NamePower)

        Add_BarsFreq_Month(DataResults[0], DataResults[1], PlotsResults, NamePlot)
        Add_AmplitudesAnalysis(DataResults[5], DataResults[6], DataResults[0], DataResults[1], PlotsResults, NamePlot)
        for i in range(2,6):
            close(i)
    
    PowerPlot[1].set_yscale("log", subs=None)
    PowerPlot[0].legend(ListBoxPlots, ListLegendsBoxPlots, loc="upper right")
    SavePlot("PowerDistributionStations", "", PowerPlot[0])
    PowerPlot[0].savefig(f"./../Results/PowerDistributionStations.png")
    close(1)
#---------------------------------------------------------------------------

def CreateInputDictionary(SubdirsResultsData, DataPath, ResultsPath):
    Dictionary = dict()

    for Region in SubdirsResultsData:
        DataRegionPath = DataPath + Region
        ResultsRegionPath = ResultsPath + Region
        
        DataPaths = []
        
        for station in listdir(DataRegionPath):
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
                
                    DataPaths += list(map(lambda x: NewAnalysisPath + "/" + x, listdir(NewAnalysisPath)))
        
        Dictionary[Region] = dict(ResultPath = ResultsRegionPath, DataPaths = DataPaths)

    return Dictionary



if __name__=="__main__":
    # Setting plotting format for all figures
    rcParams["font.family"] = "serif"
    rcParams["savefig.dpi"] = 400

    DATA_COMMON_PATH = "/home/federicosalinas/Documentos/FCFM/Proyecto TIDs/Analysis/"
    RESULTS_COMMON_PATH = "/home/federicosalinas/Documentos/FCFM/Proyecto TIDs/Results/"

    SUBDIRECTORIES_ResultsData_REGIONS = ["Center", "North", "South"]

    InputRegionsData = CreateInputDictionary(SUBDIRECTORIES_ResultsData_REGIONS, DATA_COMMON_PATH, RESULTS_COMMON_PATH)
    
    StarAnnualAnalysis(InputRegionsData)
