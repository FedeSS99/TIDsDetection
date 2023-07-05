from DataScripts.Input_TID_Data import CreateInputDictionary

def SeparateRegionDataByYear(DataDictionary:dict):
    RegionNames = DataDictionary.keys()
    RegionYearsData = dict()

    for Region in RegionNames:
        DataPathsOfRegion = DataDictionary[Region]["DataPaths"]
        Stations = DataPathsOfRegion.keys()

        DataByYears = dict()

        # First, create empty list for every year that appears
        # in DataPathsOfRegion
        Years = []
        for Station in Stations:
            ListDataPaths = DataPathsOfRegion[Station]

            YearsPerStation = tuple( (map(lambda e: e.split("/")[4], ListDataPaths)) )
            for Year in YearsPerStation:
                if Year not in Years:
                    Years.append(Year)
            
        # Filter every file that has the same year and
        # save it with a key of the same year number
        for Year in Years:
            strYear = str(Year)
            DataByYears[strYear] = []
            for Station in Stations:
                ListDataPaths = DataPathsOfRegion[Station]

                YearsPerStation = tuple( (map(lambda e: e.split("/")[4], ListDataPaths)) )
                DataByYears[strYear] += [DataPath for DataPath, YearPath in zip(ListDataPaths,YearsPerStation) if YearPath == strYear]

        # Save DataByYears in RegionYearsData with Region key
        RegionYearsData[Region] = DataByYears

    return RegionYearsData
    

def WriteRegionDataByYears(RegionYearsData:dict):
    RegionNames = RegionYearsData.keys()

    # Run over every Region
    for Region in RegionNames:

        # Run over every existent Year of data
        Years = RegionYearsData[Region].keys()
        for Year in Years:
            # Start counter
            Counter = 0

            #Create output file in which all the data from one
            #Region on a specific Year will be saved
            with open(f"{Region}_{Year}.dat", "w") as Output:
                Output.write(f"{Year}\n")

                # Run over every path saved in a Region in Year
                DataPaths = RegionYearsData[Region][Year]
                for DataPath in DataPaths:

                    # Read all the events' data of DataPath
                    with open(DataPath, "r") as Input:
                        Lines = Input.readlines()

                        DataLines = Lines[2:]                        
                        # Check if there is, at least, one event in file:
                        if DataLines:

                            # Write headers for columns if Counter is zero
                            if not Counter:
                                Output.write("#Date " + Lines[1][1:])

                            DateString = DataPath.split("/")[-1][8:18]
                            for DataLine in DataLines:
                                Output.write(DateString + " " + DataLine)

                    Counter += 1
                        

if __name__ == "__main__":
    """
    Define paths for input data   
    """
    DATA_COMMON_PATH = "../Analysis/"
    RESULTS_COMMON_PATH = "../Analysis"

    SUBDIRECTORIES_REGIONS = ["North", "Center", "South"]

    RegionsData = CreateInputDictionary(SUBDIRECTORIES_REGIONS,
                                        DATA_COMMON_PATH, RESULTS_COMMON_PATH) 
    
    RegionYearsData = SeparateRegionDataByYear(RegionsData)

    WriteRegionDataByYears(RegionYearsData)