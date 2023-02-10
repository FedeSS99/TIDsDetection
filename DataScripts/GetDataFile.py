from numpy import float64, array

def SingleTIDs_Analysis(TID_path):
    #------------------------------------------------------------------------------------
    #Reading TIDs events from TID_path
    with open(TID_path, "+r") as tid_file:
        tid_file_lines = tid_file.readlines()

        #Reading and saving data from tid_file
        MinPeriods, MaxPeriods = [], []
        MidTimeTIDs, TimeTIDs = [], []
        MidPeriodTIDs, PeriodTIDs = [], []
        MaxPowerTIDs = []
        #Starting at third line
        for line in tid_file_lines[2:]:
            splitData = line.split()

            MinPeriods.append(splitData[2])
            MaxPeriods.append(splitData[3])
            MidTimeTIDs.append(splitData[4])
            TimeTIDs.append(splitData[5])
            MidPeriodTIDs.append(splitData[6])
            PeriodTIDs.append(splitData[7])
            MaxPowerTIDs.append(splitData[8])

        #Declaring lists of data as numpy arrays
        MinPeriods, MaxPeriods = array(MinPeriods, dtype=float64), array(MaxPeriods, dtype=float64)
        MidTimeTIDs = array(MidTimeTIDs, dtype=float64)
        TimeTIDs = array(TimeTIDs, dtype=float64)
        MidPeriodTIDs = array(MidPeriodTIDs, dtype=float64)
        PeriodTIDs = array(PeriodTIDs, dtype=float64)
        MaxPowerTIDs = array(MaxPowerTIDs, dtype=float64)

    return MinPeriods, MaxPeriods, MidTimeTIDs, TimeTIDs, MidPeriodTIDs, PeriodTIDs, MaxPowerTIDs