def ReadWSFile(FileName:str, Hour:int, Month:int, Years:list[int]):
    MagVelArray = []

    with open(FileName, "r") as WSFile:
        
        for numLine, Line in enumerate(WSFile):
            if numLine >= 14:
                SplitLine = [Val for Val in Line.split(" ") if Val != ""]
                YearLine = int(SplitLine[0])
                MonthLine = int(SplitLine[1])
                HourLine = int(SplitLine[3])
                if YearLine in Years and MonthLine == Month and HourLine == Hour:
                    MagVelArray.append(float(SplitLine[6]))

    return MagVelArray
