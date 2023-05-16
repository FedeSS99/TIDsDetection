from pandas import ExcelFile

"""
Python script to read and save solar terminators' time
for all the 365 days of the year. The data is separated in
columns in the following format:
Time    Sunrise Time    Sunset Time
"""

# Reading NOAA calculations data
file = "NOAA_Solar_Calculations_year.xlsx"
data = ExcelFile(file)

# Only parsin data related to solar terminators
df = data.parse("Calculations")
strSunrise, strSunset = "Sunrise Time (LST)", "Sunset Time (LST)"

# Getting lists for terminators data
SunriseTime = df[strSunrise].tolist()
SunsetTime = df[strSunset].tolist()

# Lambda function to apply
ToNum = lambda dateTime: dateTime.hour + dateTime.minute/60.0 + dateTime.second/3600.0
SunriseTime = list(map(ToNum, SunriseTime))
SunsetTime = list(map(ToNum, SunsetTime))

with open("TerminatorHours.dat", "w") as OutFile:
    if len(SunriseTime) == len(SunsetTime):
        OutFile.write("Sunrise Time (LST) Sunset Time (LST)\n")
        for dateRise, dateSet in zip(SunriseTime, SunsetTime):
            OutFile.write(f"{dateRise} {dateSet}\n")
    else:
        print("Non existent or equal amount of data for each time sets")
