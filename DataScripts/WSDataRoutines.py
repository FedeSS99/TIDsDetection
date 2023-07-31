def ReadWSFile(FileName:str):
    YearList = []
    MonthList = []
    HourList = []
    VelList = []

    with open(FileName, "r") as WSFile:
        
        for numLine, Line in enumerate(WSFile):
            if numLine >= 14:
                SplitLine = [Val for Val in Line.split(" ") if Val != ""]
                    
                YearList.append(float(SplitLine[6]))
                MonthList.append(float(SplitLine[6]))
                HourList.append(float(SplitLine[6]))
                VelList.append(float(SplitLine[6]))

    return (YearList, MonthList, HourList, VelList)
