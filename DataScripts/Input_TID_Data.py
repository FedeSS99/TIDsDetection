from os import listdir
from os.path import isdir

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