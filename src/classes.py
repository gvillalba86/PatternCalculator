import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import resize


# COSINE power 2D
def cos_power_2d(x:np.ndarray, y:np.ndarray, power:float=2):
    """
    Generates a cosine power radiant element in 2D

    Args:
        x: vector with x values
        y: vector with y values
        power: power of the cosine. The higher, the narrower the beamwith you get. 
        Defaults to 2.

    Returns:
        Radiant element
    """    
    # We add pi/2to the y-term because we want maximun of radiation to be on 
    # pi=0, theta=90 (as in antenna arrays)
    res = np.abs(np.float_power(np.cos(x), power, dtype=complex) * 
        np.float_power(np.cos(y+np.pi/2), power, dtype=complex))
    size = x.shape[1]
    res[:,:size//4] = np.finfo(np.float64).eps
    res[:,size//4*3:] = np.finfo(np.float64).eps
    return 20*np.log10(res)


class RadiantElement:
    """
    Radiant Element class
    """
    
    def __init__(self, xpos, ypos, zpos, amplitude=1, phase=0):
        self.xpos = xpos    # instance variable unique to each instance
        self.ypos = ypos
        self.zpos = zpos
        self.amplitude = amplitude
        self.phase = phase
    
    def get_position(self):
        return np.array((self.xpos, self.ypos, self.zpos))

    def __repr__(self):
        return f'{self.amplitude}∠{round(self.phase, 2)}º @ [{self.xpos} {self.ypos} {self.zpos}]'
    
    
class Array:
    """
    Array class
    """
    
    def __init__(self, elements:list, pattern=None):
        # Elements in the array
        self.elements = elements 
        self.nElements = len(self.elements)
        
        # Phi/Theta points
        self.nPhi = 1000
        self.nTheta = 500
        phi = np.linspace(-np.pi, np.pi, self.nPhi)
        theta = np.linspace(0, np.pi, self.nTheta)
        self.PHI, self.THETA = np.meshgrid(phi, theta)
        
        # Element pattern
        if pattern is not None:
            self.setElementPattern(pattern)
        else:
            # If pattern is not provided, we use cosine power
            self.elePattern = cos_power_2d(self.PHI, self.THETA, power=2)


    def setElementPattern(self, pattern:np.ndarray):
        """
        Sets the radiant element pattern to use for the array. If the given pattern does not match
        the size for the pattern os the array given by self.nPhi and self.nTheta, it will be interpolated.

        Args:
            pattern: Element pattern to use, for all space theta/phi. It assumes theta goes
            from 0 to pi and phi from -pi/2 to pi/2.
        """
        if pattern.shape == (self.nPhi, self.nTheta):
            # If pattern is provided annd matches the Pattern size,
            # we use it directly
            self.elePattern = pattern
        else:
            # If pattern is provided but do not match the size of the
            # pattern size, we interpolate
            self.elePattern = resize(pattern, (self.nTheta,self.nPhi))


            
    def arrayFactor(self, freq:float):
        """
        Computes the array factor for the array and returns it for all space (phi, theta)

        Args:
            freq: Frequency to calculate the array factor
            
        Returns:
           Array factor
        """
        
        def CalculateRelativePhase(element_pos, Lambda, PHI, THETA):
            """
            _summary_

            Args:
                element_pos: Tuple with element position in (x, y, z) coordinates
                Lambda: wavelength in meters
                PHI: phi 2D matrix
                THETA: theta 2D matrix

            Returns:
                Relative phase for the given element in (x, y, z) in all space (for all phi/theta).
            """
            phaseConstant = (2 * np.pi / Lambda)

            xVector = element_pos[0] * np.sin(THETA) * np.cos(PHI)
            yVector = element_pos[1] * np.sin(THETA) * np.sin(PHI)
            zVector = element_pos[2] * np.cos(THETA)
            psi = phaseConstant * (xVector + yVector + zVector)

            return psi
        
    
        # Create empty array
        AF = np.zeros((self.nTheta, self.nPhi), dtype=complex)

        # Calculate wavelength
        Lambda = 3e8 / freq

        # Calculate field amplitudes
        wAmp = [element.amplitude for element in self.elements]
        wAmp = np.sqrt(wAmp/np.sum(wAmp))

        # For all theta/phi positions
        for eleIdx, element in enumerate(self.elements):
            kd = CalculateRelativePhase(element.get_position(), Lambda, self.PHI, self.THETA)
            AF += wAmp[eleIdx] * np.exp((kd + element.phase) * 1j)
    
        # Returns power in dB
        return 20*np.log10(np.abs(AF))

        
    def arrayPattern(self, freq:float):
        """
        Returns array pattern at the specified frequency. It will be the sum of element 
        pattern and Array Factor in dB, since elemennt pattern will be the same for all elements.

        Args:
            freq: Frequency in MHz

        Returns:
            Array pattern
        """
        return self.arrayFactor(freq) + self.elePattern
    

    def plotArray(self):
        """
        Plots the array configuration

        Raises:
            ValueError: If x-position it is not null for all elements in array, 
            it is not possible to plot the array in 2D.
        """
        if all(ele.xpos == self.elements[0].xpos for ele in self.elements):
            fig, ax = plt.subplots(1, 1, figsize=(10,3.5))
            ax.set_xlabel('Z-position (mm)')
            ax.set_ylabel('Y-position (mm)')
            sns.set_theme()
            for ele in self.elements:
                ax.scatter(ele.zpos*1000, ele.ypos*1000, marker="x", s=200, 
                    linewidths=5, c="firebrick")
                ax.annotate(ele.amplitude, 
                            xy = (ele.zpos*1000, ele.ypos*1000),
                            xytext=(-5, 15),
                            textcoords='offset points')
                ax.annotate(f'{np.rint(np.rad2deg(ele.phase))}º', 
                            xy = (ele.zpos*1000, ele.ypos*1000),
                            xytext=(-15, -25),
                            textcoords='offset points')
            plt.show()
        else:
            raise ValueError('Cannot plot 3D arrays')


    def __len__(self):
        return self.nElements
    
    
    def __getitem__(self, pos):
        return self.elements[pos]
    
        
    def __repr__(self):
        repr_str = str()
        for i, ele in enumerate(self.elements):
            repr_str += f'#{i}: {ele}\n'
        return repr_str