def ReadWSFile(FileName:str):
    YearList = []
    MonthList = []
    HourList = []
    VelList = []

    with open(FileName, "r") as WSFile:
        
        for numLine, Line in enumerate(WSFile):
            if numLine >= 14:
                SplitLine = [Val for Val in Line.split(" ") if Val != ""]
                    
                YearList.append(int(SplitLine[0]))
                MonthList.append(int(SplitLine[1]))
                HourList.append(int(SplitLine[3]))
                VelList.append(float(SplitLine[6]))

    return (YearList, MonthList, HourList, VelList)
