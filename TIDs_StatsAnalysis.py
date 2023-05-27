from os import mkdir
from os.path import isdir
from datetime import date
import numpy as np
from matplotlib import rcParams
from matplotlib.pyplot import close

# Numerical and data routines
from DataScripts.Input_TID_Data import CreateInputDictionary
from DataScripts.GetDataFile import GetSingleTID_Data
from DataScripts.HistogramOcurrence import GetOcurrenceArray
from DataScripts.ListUnpacking import UnpackListOfLists

# CMAP and NORM routines
from DataScripts.CMAP_NORM import ObtainCMAPandNORM

# All Regions plotting routines
from PlottingScripts.AllRegions.Reg_Plots import CreateFiguresForAllRegions

# Format function for All Regions figures
from PlottingScripts.FormatPlots import FormatAndSave_AllRegionPlots

# All regions plotting routines
from PlottingScripts.GenPlots import StartAnalysisForOnlyRegion

# Stations and Regions plotting routines
from PlottingScripts.GenPlots import StartAnalysisForStationsAndRegions

# --------------------------------------------------------------------------------------------------------

def StarAnnualAnalysis(DICT_REGION_STATIONS, REGIONS_ATRIBS):
    # Ignore events that occoured in dates where a geomagnetic
    # storm had a major effect in the Dst value
    with open("./StormData/tormentas-2018-2021.txt", "r") as StormDaysData:
        StormDays = []
        Lines = StormDaysData.readlines()

        for Line in Lines[1:]:
            Date = Line.split()[0]
            if Date not in StormDays:
                StormDays.append(Line.split()[0])

    # Create RESULTS list to save data of time, period and power from TIDs
    # saved with VTEC_MainRoutine_IndividualCMN.py
    RESULTS = []
    Nplots = len(REGIONS_ATRIBS)
    OcurrenceSampling_AllRegions = []
    for Region in DICT_REGION_STATIONS.keys():
        NameOut = DICT_REGION_STATIONS[Region]["ResultPath"].split("/")[-1]
        print(f"-- Extracting data of {NameOut} Region --")

        # Declare counter for total days and active days for each region and
        # also the lists to save the data for all stations
        TotalDays = 0
        Num_RejectedDates = 0
        ActiveDays = 0

        Region_TimeTID = []
        Region_DateTID = []
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
            Station_DateTID = []
            Station_PeriodTID = []
            Station_PowerTID = []
            Station_MinAmps = []
            Station_MaxAmps = []
            Station_MonthArray = []

            # Extract DataPaths for each Station and also date and month data
            TIDs_DataPaths = DICT_REGION_STATIONS[Region]["DataPaths"][Station]
            Dates_TIDs = [fileName.split("/")[-1][8:18] for fileName in TIDs_DataPaths]
            MonthPerFile = [int(fileName.split("/")[-1].split("-")[2]) for fileName in TIDs_DataPaths]


            TotalDays += len(TIDs_DataPaths)
            for fileTID, MonthFile, Date_TID in zip(TIDs_DataPaths, MonthPerFile, Dates_TIDs):
                if Date_TID not in StormDays:
                    Results = GetSingleTID_Data(fileTID)
                    if SizeResults := Results["TIME"].size:
                        ActiveDays += 1

                        Station_MonthArray.append(SizeResults*[MonthFile])
                        Station_DateTID.append( SizeResults*[date.fromisoformat(Date_TID)])

                        # Get the timezone given NameOut
                        if NameOut == "North":
                            TimeZone = -8.0
                        elif NameOut == "Center":
                            TimeZone = -6.0
                        elif NameOut == "South":
                            TimeZone = -5.0

                        # Apply timezone to get correct Local Time Hours
                        Results["TIME"] += TimeZone
                        Results["TIME"] = np.where(Results["TIME"] < 0, Results["TIME"] + 24.0, Results["TIME"])
                        Station_TimeTID.append(Results["TIME"])
                        Station_PeriodTID.append(Results["PERIOD"])
                        Station_PowerTID.append(Results["POWER"])
                        Station_MinAmps.append(Results["MIN_AMPS"])
                        Station_MaxAmps.append(Results["MAX_AMPS"])
                else:
                    Num_RejectedDates += 1

            # Create numpy arrays from the tuple collections of all the TIDs
            # data for each Station
            Station_MonthArray = np.concatenate(tuple(Station_MonthArray), dtype=int)
            Station_DateTID = UnpackListOfLists(Station_DateTID)
            Station_DateTID = np.array(Station_DateTID, dtype=np.datetime64)
            Station_TimeTID = np.concatenate(tuple(Station_TimeTID))
            Station_PeriodTID = np.concatenate(tuple(Station_PeriodTID))
            Station_PowerTID = np.concatenate(tuple(Station_PowerTID))
            Station_MinAmps = np.concatenate(tuple(Station_MinAmps))
            Station_MaxAmps = np.concatenate(tuple(Station_MaxAmps))

            # Obtain the ocurrence map for the TIDs in this same Station
            StationOcurrenceMap = GetOcurrenceArray(Station_TimeTID, Station_MonthArray)

            # Save all the Station data in one dictionary and...
            StationResultsDict = {
                "TIME": Station_TimeTID,
                "DATETIME": Station_DateTID,
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
            Region_DateTID.append(StationResultsDict["DATETIME"])
            Region_TimeTID.append(StationResultsDict["TIME"])
            Region_PeriodTID.append(StationResultsDict["PERIOD"])
            Region_PowerTID.append(StationResultsDict["POWER"])
            Region_MinAmps.append(StationResultsDict["MIN_AMPS"])
            Region_MaxAmps.append(StationResultsDict["MAX_AMPS"])

            # Check if the directory for the Station given its' Region exists,
            # and if not, create itOcurrenceSampling_AllRegions
            StationSavedir = f"./../Results/{Region}/{Station}/"
            if not isdir(StationSavedir):
                mkdir(StationSavedir)

            # Save minimum and maximum values of local ocurrence of TIDs for each Station
            CMAP, NORM = ObtainCMAPandNORM(StationResultsDict["OCURRENCE"].flatten())
            StartAnalysisForStationsAndRegions(Region, Station, "Stat", StationResultsDict, CMAP, NORM)

        # Create numpy arrays for all Stations data from the same Region
        Region_MonthArray = np.concatenate(tuple(Region_MonthArray), dtype=int)
        Region_DateTID = np.concatenate(tuple(Region_DateTID), dtype=np.datetime64)
        Region_TimeTID = np.concatenate(tuple(Region_TimeTID))
        Region_PeriodTID = np.concatenate(tuple(Region_PeriodTID))
        Region_PowerTID = np.concatenate(tuple(Region_PowerTID))
        Region_MinAmps = np.concatenate(tuple(Region_MinAmps))
        Region_MaxAmps = np.concatenate(tuple(Region_MaxAmps))
        NumTIDs = Region_TimeTID.size

        # Get ocurrence map for each Region
        HistogramOcurrence = GetOcurrenceArray(Region_TimeTID, Region_MonthArray)

        # Get average absolute amplitude for each region and all the correspond data in the directory
        # in a dictionary
        AveAbsAmplitude = (np.abs(Region_MinAmps) + Region_MaxAmps)/2.0

        RegionResultsDict = {
            "TIME": Region_TimeTID,
            "DATETIME": Region_DateTID,
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

        CMAP, NORM = ObtainCMAPandNORM(OcurrenceSampling_AllRegions[-1])
        StartAnalysisForStationsAndRegions(Region, "", "Reg", RegionResultsDict, CMAP, NORM)

        print(f"Total days:{TotalDays:d}\nRejected days:{Num_RejectedDates:d}\nActive days:{ActiveDays:d}\nActive day ratio: {NumTIDs/ActiveDays:.3f}\n")

    # Obtain ColorMap and Norm to use for all the regions
    OcurrenceSampling_AllRegions = np.concatenate(
        tuple(OcurrenceSampling_AllRegions))

    ListBoxPlots, ListLegendsBoxPlots = [], []
    CMAP, NORM = ObtainCMAPandNORM(OcurrenceSampling_AllRegions)
    PlotsResults = CreateFiguresForAllRegions(Nplots)

    for RegionDataResults in RESULTS:
        # Get a string Coord to use in the analysis' plots results
        NamePlot = RegionDataResults["NAME"]

        # Generate graphics to add results of each region
        StartAnalysisForOnlyRegion(RegionDataResults, NamePlot, REGIONS_ATRIBS, CMAP, NORM, 
                                   PlotsResults, ListBoxPlots, ListLegendsBoxPlots)

    FormatAndSave_AllRegionPlots(Nplots, PlotsResults, ListBoxPlots, ListLegendsBoxPlots)

    for i in range(1,7):
        close(i)


# --------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Setting plotting format for all figures
    rcParams["font.family"] = "serif"
    rcParams["savefig.dpi"] = 400

    DATA_COMMON_PATH = "../Analysis/"
    RESULTS_COMMON_PATH = "../Results/"

    SUBDIRECTORIES_REGIONS = ["North", "Center", "South"]

    # Create dictionary for atributes to use as input information for each Region
    # plot; the information has to be given in the following order
    # [ Color, Symbol marker, Index]
    RegionsInfo = {
        "North": ["blue", "^", 0],
        "Center": ["green", "*", 1],
        "South": ["red", "s", 2]
    }

    InputRegionsData = CreateInputDictionary(
        SUBDIRECTORIES_REGIONS, DATA_COMMON_PATH, RESULTS_COMMON_PATH)

    StarAnnualAnalysis(InputRegionsData, RegionsInfo)
