from DataScripts.Input_TID_Data import CreateInputDictionary
from DataScripts.FullAnalysisRoutine import StarFullAnalysis

if __name__ == "__main__":
    DATA_COMMON_PATH = "../Analysis/"
    RESULTS_COMMON_PATH = "../Results/"

    SUBDIRECTORIES_REGIONS = ["North", "Center-MNIG", "Center-UCOE", "South"]

    # Create dictionary for atributes to use as input information for each Region
    # plot; the information has to be given in the following order
    # [ Color, Symbol marker, Index, Timezone]
    RegionsInfo = {
        "North": ["blue", "^", 0, -8.0],
        "Center-MNIG": ["green", "*", 1, -6.0],
        "Center-UCOE": ["lime", "o", 2, -6.0],
        "South": ["red", "s", 3, -5.0]
    }

    InputRegionsData = CreateInputDictionary(
        SUBDIRECTORIES_REGIONS, DATA_COMMON_PATH, RESULTS_COMMON_PATH)

    StarFullAnalysis(InputRegionsData, RegionsInfo)