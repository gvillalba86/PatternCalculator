import numpy as np
import pandas as pd
from .classes import RadiantElement


def loadElePattern(filename:str):
    """
    Loads a file containing a diagram patterns for all space and several frequencies

    Args:
        filename: path to file containing the data

    Returns:
        3D Numpy array containing patterns for all space (theta, phi) and frequencies
    """    
    df = pd.read_csv(filename)
    nFreq = df.iloc[:,0].nunique()
    nPhi = df.iloc[:,1].nunique()
    nTheta = df.iloc[:,2].nunique()
    data = df.iloc[:,3].to_numpy().reshape((nPhi, nFreq, nTheta), order='C')
    return data.transpose((2, 0, 1)), df.iloc[:,0].unique()*1000


def element_array (xpos, ypos, zpos, amplitude, phase):
    """
    Creates an array of RadiantElement objects from lists of properties

    Args:
        xpos: List with elements positions in x-axis
        ypos: List with elements positions in y-axis
        zpos: List with elements positions in z-axis
        amplitude: List with elements amplitudes
        phase: List with elements applied phases

    Raises:
        IndexError: If lists does not have the same length

    Returns:
        Array of RadiantElement objects
    """      

    ArrayLength = len(xpos)

    # Raise exception if not all lists have the same length
    if any(len(lst) != ArrayLength for lst in [ypos, zpos, amplitude, phase]):
        raise IndexError 

    # Create list of radiant elements
    array = list()
    for i in range(ArrayLength):
        array.append(RadiantElement(xpos[i], ypos[i], zpos[i], amplitude[i], phase[i]))
    return array