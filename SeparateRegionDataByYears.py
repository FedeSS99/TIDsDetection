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
    # Run over every Region
    RegionNames = RegionYearsData.keys()
    for Region in RegionNames:
        # Run over every existent Year of data
        Years = RegionYearsData[Region].keys()
        for Year in Years:
            # Start counter
            Counter = 0

            #Create output file in which all the data from one
            #Region on a specific Year will be saved
            NameFile = f"{Region}_{Year}.dat"
            SavePath = f"../Analysis/{Region}/"
            print(f"Writing {NameFile} in {SavePath}", end=" ")
            with open(f"{SavePath}{NameFile}", "w") as Output:
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
                                Output.write("#Date Station " + Lines[1][1:])

                            FileName = DataPath.split("/")[-1]
                            DateString = FileName[8:18]
                            StationString = FileName[:4]
                            for DataLine in DataLines:
                                Output.write(DateString + " " + StationString + " " + DataLine)

                    Counter += 1
            print(f"-> Saved {NameFile} in {SavePath}")
                        

if __name__ == "__main__":
    """
    Define paths for input data   
    """
    DATA_COMMON_PATH = "../Analysis/"

    SUBDIRECTORIES_REGIONS = ["North", "Center", "South"]

    RegionsData = CreateInputDictionary(SUBDIRECTORIES_REGIONS,
                                        DATA_COMMON_PATH, "") 
    
    RegionYearsData = SeparateRegionDataByYear(RegionsData)

    WriteRegionDataByYears(RegionYearsData)