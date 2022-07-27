import numpy as np
from .classes import RadiantElement

def element_array (xpos, ypos, zpos, amplitude, phase):
    length = len(xpos)
    # Raise exception if not all lists have the same length
    if any(len(lst) != length for lst in [ypos, zpos, amplitude, phase]):
        raise IndexError 
    # Create list of radiant elements
    array = list()
    for i, element in enumerate(xpos):
        array.append(RadiantElement(xpos[i], ypos[i], zpos[i], amplitude[i], phase[i]))
    return array