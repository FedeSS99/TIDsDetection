from os import mkdir
from os.path import isdir
import numpy as np
from matplotlib import rcParams
from matplotlib.pyplot import close

# Numerical and data routines
from DataScripts.Input_TID_Data import CreateInputDictionary
from DataScripts.GetDataFile import GetSingleTID_Data
from DataScripts.HistogramOcurrence import GetOcurrenceArray

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
        NameRegion = DICT_REGION_STATIONS[Region]["ResultPath"].split("/")[-1]
        print(f"-- Extracting data of {NameRegion} Region --")

        # Declare counter for total days and active days for each region and
        # also the lists to save the data for all stations
        TotalDays = 0
        Num_RejectedDates = 0
        ActiveDays = 0

        Region_TimeTID = []
        Region_PeriodTID = []
        Region_PowerTID = []
        Region_MinAmps = []
        Region_MaxAmps = []
        Region_MonthArray = []
        Region_YearArray = []

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
            Station_YearArray = []

            # Extract DataPaths for each Station and also date and month data
            TIDs_DataPaths = DICT_REGION_STATIONS[Region]["DataPaths"][Station]
            Dates_TIDs = [fileName.split("/")[-1][8:18] for fileName in TIDs_DataPaths]

            TotalDays += len(TIDs_DataPaths)
            for fileTID, Date in zip(TIDs_DataPaths, Dates_TIDs):
                if Date not in StormDays:
                    SplitDate = Date.split("-")
                    Year = int(SplitDate[0])
                    Month = int(SplitDate[1])

                    Results = GetSingleTID_Data(fileTID)
                    if SizeResults := Results["TIME"].size:
                        ActiveDays += 1
                        Station_MonthArray.append(SizeResults*[Month])
                        Station_YearArray.append(SizeResults*[Year])

                        # Apply timezone to get correct Local Time Hours
                        Results["TIME"] += REGIONS_ATRIBS[NameRegion][3]
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
            Station_YearArray = np.concatenate(tuple(Station_YearArray), dtype=int)
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
                "MONTH": Station_MonthArray,
                "YEAR": Station_YearArray,
                "OCURRENCE": StationOcurrenceMap,
                "PERIOD": Station_PeriodTID,
                "POWER": Station_PowerTID,
                "MIN_AMPS": Station_MinAmps,
                "MAX_AMPS": Station_MaxAmps,
            }

            # and save these numpy arrays in the correspondent list to manage
            # all the TIDs data for each Region
            Region_MonthArray.append(StationResultsDict["MONTH"])
            Region_YearArray.append(StationResultsDict["YEAR"])
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
        Region_YearArray = np.concatenate(tuple(Region_YearArray), dtype=int)
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
            "MONTH": Region_MonthArray,
            "YEAR": Region_YearArray,
            "OCURRENCE": HistogramOcurrence,
            "PERIOD": Region_PeriodTID,
            "POWER": Region_PowerTID,
            "AVE_AMP": AveAbsAmplitude,
            "NAME": NameRegion
        }

        # Save a flatten version of the ocurrence map of the Region
        OcurrenceSampling_AllRegions.append(HistogramOcurrence.flatten())
        RESULTS.append(RegionResultsDict)

        CMAP, NORM = ObtainCMAPandNORM(OcurrenceSampling_AllRegions[-1])
        StartAnalysisForStationsAndRegions(Region, "", "Reg", RegionResultsDict, CMAP, NORM)

        print(f"Total days:{TotalDays:d}\nRejected days:{Num_RejectedDates:d}\nActive days:{ActiveDays:d}\nNo. of TIDs: {NumTIDs}\n#TIDs/Active Days: {NumTIDs/ActiveDays:.3f}\n")

    # Obtain ColorMap and Norm to use for all the regions
    OcurrenceSampling_AllRegions = np.concatenate(tuple(OcurrenceSampling_AllRegions))

    CMAP, NORM = ObtainCMAPandNORM(OcurrenceSampling_AllRegions)
    PlotsResults = CreateFiguresForAllRegions(Nplots)
    for RegionDataResults in RESULTS:
        # Get a string Coord to use in the analysis' plots results
        NamePlot = RegionDataResults["NAME"]

        # Generate graphics to add results of each region
        StartAnalysisForOnlyRegion(RegionDataResults, NamePlot, REGIONS_ATRIBS, CMAP, NORM, PlotsResults)

    FormatAndSave_AllRegionPlots(Nplots, PlotsResults)

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
    # [ Color, Symbol marker, Index, Timezone]
    RegionsInfo = {
        "North": ["blue", "^", 0, -8.0],
        "Center": ["green", "*", 1, -6.0],
        "South": ["red", "s", 2, -5.0]
    }

    InputRegionsData = CreateInputDictionary(
        SUBDIRECTORIES_REGIONS, DATA_COMMON_PATH, RESULTS_COMMON_PATH)

    StarAnnualAnalysis(InputRegionsData, RegionsInfo)