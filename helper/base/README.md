# mathusla_prototype
This project descibes the uvic MATHUSLA prototype geometry through a JSON file
## JSON format
The JSON file is structed as lists of dictionaries, Bars nested in fibres nested in layers, eg.

{Layers:[ {Fibres:[{Bars:[]}]}]}

Layers contain {ID, Direction, z, Fibres[]} <br>
Fibres contain {ID, path, Interfaces[], Bars[], Loops[]}<br>
Interfaces contain {ID, length, channel, bar}<br>
Bars contain {ID, length, width, thickness, x, y}<br>
Loops contain {ID, length, bar0, bar1}<br>

The key,type(value) definitions are:<br>
ID:string - a unique string identifier within a list<br>
Direction:string - x or y for the direction the bars extend in<br>
z:int - the height of the layer in cm<br>
path:[strings] - An ordered list of ID's of objects the fiber passes through<br>
length:int - the length of fibre in the specified object<br>
channel:int - the SiPM channel the fibre connects to<br>
bar:string - ID of the bar the fibre enters<br>
width:int - the width of the bar in cm<br>
thickness:int - the height/thickness of the bar in cm<br>
x:int - the distance from the origin to the face of the bar in cm, defined on the side which connects to the SiPM<br> 
y:int - the distance from the origin to the face of the bar in cm<br>

other defintions:<br>
origin: intersection of the DAQ and back wall at the height of the bottom layer