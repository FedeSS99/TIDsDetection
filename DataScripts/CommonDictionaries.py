# Dictionary to extract filename of Terminator data for each region
TerminatorsDict = {
    "North":"./TerminatorData/TerminatorHours_North.dat",
    "Center-MNIG":"./TerminatorData/TerminatorHours_Center.dat",
    "Center-UCOE":"./TerminatorData/TerminatorHours_Center.dat",
    "South":"./TerminatorData/TerminatorHours_South.dat"
    }

# Dictionary to extract colors to use in day-night filter for amplitude-power data
DayNightColors = dict(Day="red", Night="blue")

# Dictionary to assign title to each region subplot
IndexName = {
    0: "A) North",
    1: "B) Center-MNIG",
    2: "C) Center-UCOE",
    3: "D) South"
    }