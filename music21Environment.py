from  music21 import *
us = environment.UserSettings()
for key in sorted(us.keys()):
    print(key)

us['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore 3/bin/MuseScore3.exe'
us['musicxmlPath'] = 'C:/Program Files/MuseScore 3/bin/MuseScore3.exe'
