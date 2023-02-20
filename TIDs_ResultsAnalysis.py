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
    for Region in DICT_REGION_STATIONS.keys():
        ActiveDays = 0
        TotalDays = 0
        ResultsTimeTID = []
        ResultsPeriodTID = []
        ResultsPowerTID = []
        MonthArray = []
        ListBoxPlots = []
        ListLegendsBoxPlots = []

        NameOut = DICT_REGION_STATIONS[Region]["Path"].split("/")[-1]
        #Obtain the full path of the files located in the selected directories and
        #the dates and months of these same files
        for path in DICT_REGION_STATIONS[Region]["Stations"]:
            if isdir(path):
                files_full_path = [path+"/"+file for file in listdir(path) if file.endswith("_TIDs.dat")]
                Dates_TIDs = [fileName.split(".")[0].split("/")[-1][-15:-5] for fileName in files_full_path]
                MonthPerFile = [int(fileName.split("/")[-1].split("-")[2]) for fileName in files_full_path]

                #Get the name for the analysis' plots results
                print(f"--Obtaining results of {path}--")
                TotalDays += len(files_full_path)

                for fileTID, MonthFile, Date_TID in tqdm(zip(files_full_path, MonthPerFile, Dates_TIDs)):
                    if Date_TID not in StormDays:
                        Results = SingleTIDs_Analysis(fileTID)
                        SizeResults = Results[0].size
                        if SizeResults:
                            ActiveDays += 1

                            MonthArray.append(SizeResults*[MonthFile])
                            ResultsTimeTID.append(Results[0])
                            ResultsPeriodTID.append(Results[1])
                            ResultsPowerTID.append(Results[2])

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
        NumTIDs = ResultsTimeTID.size

        #Get 2D histogram of TIDs' ocurrence and save all the data from the respective directory
        #in RESULTS
        HistogramOcurrence = Time_Months_Ocurrence_Analysis(ResultsTimeTID, MonthArray)
        RESULTS.append([ResultsTimeTID, MonthArray, HistogramOcurrence, ResultsPeriodTID, ResultsPowerTID, NameOut])

        print(f"# of TIDs:{NumTIDs}\nActive Days:{ActiveDays} Total Days:{TotalDays}\n")

    #Obtain the globam minimum and maximum of the ocurrence arrays of all the directories' data
    MIN, MAX = min([DataResults[2].min() for DataResults in RESULTS]), max([DataResults[2].max() for DataResults in RESULTS])
    for DataResults in RESULTS:
        #Get a string Coord to use in the analysis' plots results
        NamePlot = DataResults[-1]
        Coord = NameOut
        
        if NameOut == "Center":
            NamePower = "LNIG-MNIG-UCOE"
            ColorPower = "red"
            RegIndex = 0
        elif NameOut == "North":
            NamePower = "PTEX"
            ColorPower = "green"
            RegIndex = 1
        elif NameOut == "South":
            NamePower = "CN24"
            ColorPower = "blue"
            RegIndex = 2

        #Start creating Matplotlib plot to visualize the statistics given the data from RESULTS
        PlotsResults = CreateFiguresResults(Coord)
        AddTimeMonthsHistogramToPlot(DataResults[2], MIN, MAX, PlotsResults, NamePlot)
        AddPeriodHistogramToPlot(DataResults[3], DataResults[0], DataResults[1], PlotsResults, NamePlot)
        BoxPlotObject = addTimePowerDataResultsToPlot(DataResults[0], DataResults[4], PowerPlot, ColorPower, RegIndex)
        ListBoxPlots.append(BoxPlotObject)
        ListLegendsBoxPlots.append(NamePower)

        BarsFreq_Month(DataResults[0], DataResults[1], PlotsResults, NamePlot)
        for i in range(2,5):
            close(i)
    
    PowerPlot[1].set_yscale("log", subs=None)
    PowerPlot[0].legend(ListBoxPlots, ListLegendsBoxPlots, loc="upper right")
    PowerPlot[0].savefig(f"./../Resultados/PowerDistributionStations.png")
    close(1)
#---------------------------------------------------------------------------

def CreateInputDictionary(SubdirsData, SubDirsResults, DataPath, ResultsPath):
    Dictionary = dict()

    for Region, StationsRegion in zip(SubDirsResults, SubdirsData):
        Dictionary[Region] = dict(Path = ResultsPath + Region, Stations = [DataPath + Station for Station in StationsRegion])

    return Dictionary

if __name__=="__main__":
    # Setting plotting format for all figures
    rcParams["font.family"] = "serif"
    rcParams["savefig.dpi"] = 400

    DATA_COMMON_PATH = "/home/fssamaniego/Documentos/FCFM/TIDs/Análisis/"
    RESULTS_COMMON_PATH = "/home/fssamaniego/Documentos/FCFM/TIDs/Resultados/"

    SUBDIRECTORIES_DATA = [["mnig", "lnig"]]
    SUBDIRECTORIES_RESULTS = ["Center"]

    InputDict = CreateInputDictionary(SUBDIRECTORIES_DATA, SUBDIRECTORIES_RESULTS,
                                        DATA_COMMON_PATH, RESULTS_COMMON_PATH)
    StarAnnualAnalysis(InputDict)
