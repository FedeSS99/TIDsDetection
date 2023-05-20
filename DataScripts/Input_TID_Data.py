from os import listdir
from os.path import isdir

def CreateInputDictionary(SubdirsResultsData, DataPath, ResultsPath):
    # Create dictionary to save the paths of TIDs analysis
    InputDictionary = {}

    # Run over every region given the keys in SubdirsResultsData
    for Region in SubdirsResultsData:

        # Define the paths for the Data and the Results for Region
        DataRegionPath = DataPath + Region
        ResultsRegionPath = ResultsPath + Region

        # Generate a dictionary for the station in proccess
        StationsDict = {}

        # Run over every station directory in DataRegionPath
        for station in listdir(DataRegionPath):
            StationDataPaths = []

            # Check if the CompletStationDir is a directory and not any other
            # thing like a file
            CompleteStationDir = f"{DataRegionPath}/{station}"
            if not isdir(CompleteStationDir):
                # Jump if it is a file
                continue

            # If it is a directory which has year and month directories in it, run the
            # double cycle over every year and month
            for yearDir in listdir(CompleteStationDir):
                CompleteYearDir = f"{CompleteStationDir}/{yearDir}"
                if not isdir(CompleteYearDir) or not yearDir.isnumeric():
                    continue

                for monthDir in listdir(CompleteYearDir):
                    NewAnalysisPath = f"{CompleteYearDir}/{monthDir}"
                    if not isdir(NewAnalysisPath) or not monthDir.isnumeric():
                        continue

                    # Create a list with the complete path for each analysis file in
                    # each year and month directory
                    StationDataPaths += list(
                        map(
                            lambda x: f"{NewAnalysisPath}/{x}",
                            listdir(NewAnalysisPath),
                        )
                    )

            # Save the station list in StationsDict
            StationsDict[station] = StationDataPaths

        # Save the StationsDict in InputDictionary
        InputDictionary[Region] = dict(
            ResultPath=ResultsRegionPath, DataPaths=StationsDict)

    return InputDictionary