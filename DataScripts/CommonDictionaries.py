# Dictionary to extract filename of Terminator data for each region
TerminatorsDict = {
    "West":"./TerminatorData/TerminatorHours_West.dat",
    "Center-MNIG":"./TerminatorData/TerminatorHours_Center-MNIG.dat",
    "Center-UCOE":"./TerminatorData/TerminatorHours_Center-UCOE.dat",
    "East":"./TerminatorData/TerminatorHours_East.dat"
    }

# Dictionary to extract colors to use in day-night filter for amplitude-power data
DayNightColors = dict(Day="red", Night="blue")

# Dictionary to assign title to each region subplot
IndexName = {
    0: "(A) West",
    1: "(B) Center-MNIG",
    2: "(C) Center-UCOE",
    3: "(D) East"
    }