#=================================================
#Geometry loader for MATHUSLA UVic Prototype
#by Caleb Miller and Charlie Chen
#editted by Bec, 2026
#
#Contains methods to convert JSON file to easily 
#referenceable channel maps as well as a xyz 
#coordiante finder for a given hit
#=================================================

import json
import numpy as np
import pandas as pd

#=================================================
#DETECTOR OBJECT
#=================================================
class detector:
    '''
    base detector object for constructing the prototype
    '''
    def __init__(self, geo_data):
        '''
        builds the detector: this object contains a dict of 'layer' objects indexed by ID
        '''
        self.layers = {}
        #layer object has ID, direction (x, y), z position, and a dict of fibre objects indexed by ID
        for lyr in geo_data["Layers"]:
            layer_in = layer(lyr["Direction"], lyr["z"])
            process_fibres(lyr, layer_in)
            self.layers[lyr["ID"]] = layer_in

    def __str__(self):
        msg="Detector size: %d Layer(s) \n" %(len(self.layers))
        for key,val in self.layers.items():
            msg+="\t"+str(val)
        return msg

#=================================================
#LAYER OBJECT
#=================================================
class layer:
    '''
    layer class, contains the bar orientation, z position information, and a list of fibres indexed by ID
    '''
    def __init__(self, f_direction, f_z):
        self.direction = f_direction
        self.z = f_z
        self.fibres ={}
    
    def __str__(self):
        msg = "Layer height: {z}, Direction: {d} \n".format(z =self.z, d=self.direction)
        for key,val in self.fibres.items():
            msg+="\t"+str(val)
        return msg
    
    def get_path(self, f_channel):
        '''
        Takes a channel number and returns an ordered list of objects the fibre passes through starting at the channel given
        '''
        objs=[]

        for fibre_obj in self.fibres.values():  #find fibre with the channel
            if f_channel in fibre_obj.interfaces:
                for item in fibre_obj.path:  #loop over all items in the list, looking for the next one in the path, once found add to list and find next
                    for _,inter in fibre_obj.interfaces.items():
                        if(inter.ID == item):
                            objs.append(inter)
                    for _,bar in fibre_obj.bars.items():
                        if(bar.ID == item):
                            objs.append(bar)       
                    for _,loop in fibre_obj.loops.items():
                        if(loop.ID == item):
                            objs.append(loop)

                if(fibre_obj.interfaces[f_channel].ID=="I1"): #If the channel corresponds to 2nd interface, reverse list
                    objs.reverse()
        return objs
    
#=================================================
#FIBRE OBJECT
#=================================================
class fibre:
    '''
    fibre object: contains dictionaries of interfaces, bars, and loops, unique by ID, and contain a path which orders those objects by ID
    '''
    def __init__(self):
        self.interfaces = {}
        self.bars = {}
        self.loops = {}
        self.ID = -1
        self.path = []
    def __str__(self):
        msg = "Fibre ID: {ID}, fibre path: {path}\n".format(ID=self.ID, path=self.path)
        for key,val in self.interfaces.items():
            msg+="\t\t"+"Interface to channel: {ch}".format(ch=key)
            msg+=str(val)
        for key,val in self.bars.items():
            msg+="\t\t"+"Bar: {bar}".format(bar=key)
            msg+=str(val)
        for key,val in self.loops.items():
            msg+="\t\t"+"Loop between: {ch}".format(ch=key)
            msg+=str(val)
        return msg

#=================================================
#INTERFACE OBJECT
#=================================================  
class interface:
    '''
    interface object: contains the length of fibre between a channel and a bar, as well as the channel and bar ID
    '''
    def __init__(self, f_length, f_bar, f_ID):
        self.length = f_length
        self.bar = f_bar
        self.ID = f_ID
    def __str__(self) -> str:
        return ", interface length: {leng}\n".format(leng=self.length)
    
#=================================================
#BAR OBJECT
#=================================================
class bar:
    '''
    bar object: contains the bar dimensions, a unique ID, and the x,y coordinate of the face along +x or -y (closest to SiPM)
    '''
    def __init__(self, f_length, f_width, f_thickness, f_x, f_y, f_ID):
        self.length = f_length
        self.width = f_width
        self.thickness = f_thickness
        self.x = f_x
        self.y = f_y
        self.ID = f_ID
    def __str__(self) -> str:
        msg=""
        msg+=", bar length: {var}".format(var=self.length)
        msg+=", bar width: {var}".format(var=self.width)
        msg+=", bar thickness: {var}".format(var=self.thickness)
        msg+=", bar x: {var}".format(var=self.x)
        msg+=", bar y: {var}".format(var=self.y)
        msg+="\n"
        return msg
    def get_bounds(self, layer):
        '''
        returns the x and y bounds of the bar as a tuple: (x_min, x_max, y_min, y_max)
        '''
        if layer == 1 or layer == 3:  # x-oriented layers
            x_min = self.x - self.length
            x_max = self.x
            y_min = self.y - self.width / 2
            y_max = self.y + self.width / 2
        elif layer == 0 or layer == 2:  # y-oriented layers
            x_min = self.x - self.width / 2
            x_max = self.x + self.width / 2
            y_min = self.y
            y_max = self.y + self.length
        else:
            raise ValueError("Invalid layer number. Must be 0, 1, 2, or 3.")
        return (x_min, x_max, y_min, y_max)

#=================================================
#LOOP OBJECT
#=================================================
class loop:
    '''
    loop object: contains the length of fibre between bars, a unique ID, and the linked bar IDs
    '''
    def __init__(self, f_length, f_ID, f_b0, f_b1):
        self.length = f_length
        self.ID = f_ID
        self.bar0 = f_b0
        self.bar1= f_b1
    def __str__(self) -> str:
        return ", loop length: {var}\n".format(var=self.length)

#=================================================
#FUNCTIONS TO ADD INTERFACES, BARS & LOOPS TO FIBRES
#=================================================
def process_interfaces(f_fibre, f_fibre_obj):
    '''
    Takes in a fibre dictionary from JSON and a 'fibre' object
    creates an interface object and adds it to the fibre, indexed by channel number
    '''
    for i in f_fibre[ "Interfaces" ]:
        input_interface = interface(i[ "length" ], i[ "bar" ], i["ID"] )
        f_fibre_obj.interfaces[ i[ "channel" ] ] = input_interface


def process_bars(f_fibre, f_fibre_obj):
    '''
    Takes in a fibre dictionary from JSON and a 'fibre' object
    creates an bar object and adds it to the fibre, indexed by bar ID
    '''
    for b in f_fibre["Bars"]:
        input_bar = bar(b["length"], b["width"], b["thickness"], b["x"], b["y"], b["ID"])
        f_fibre_obj.bars[b["ID"]] = input_bar


def process_loops(f_fibre, f_fibre_obj):
    '''
    Takes in a fibre dictionary from JSON and a 'fibre' object
    creates a loop object and adds it to the fibre, indexed by ID
    '''
    for l in f_fibre[ "Loops" ]:
        input_loop = loop(l[ "length" ], l["ID"], l["bar0"], l["bar1"])
        f_fibre_obj.loops[ l[ "ID"] ] = input_loop


def process_fibres(f_layer, f_layer_obj):
    '''
    Takes a dict f_layer from json and a 'layer' object and builds fibre objects from the details and adds to the layer object
    fibres indexed by ID
    '''
    for f in f_layer[ "Fibres" ]:
        input_fibre = fibre()
        input_fibre.ID = f['ID']
        input_fibre.path = f['path']
        process_interfaces(f, input_fibre)
        process_bars(f, input_fibre)
        process_loops(f, input_fibre)
        f_layer_obj.fibres[f["ID"]] = input_fibre


#=================================================
#FUNCTION TO BUILD CHANNEL MAP
#=================================================
def build_ch_map(the_dect):
    '''
    takes the detector object and returns a list of objects (path) the fibre passes through for each channel
    '''
    map = []
    for ch in range(64):
        for _, layer_obj in the_dect.layers.items():
            temp = layer_obj.get_path(ch)
            if len(temp) == 0:
                continue
            else:
                map.append([temp, layer_obj.z, layer_obj.direction])
                break
    return map
