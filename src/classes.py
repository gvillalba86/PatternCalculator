import numpy as np
import matplotlib.pyplot as plt


# COSINE power 2D
def cos_power_2d(x, y, power=2):
    # We add pi/2to the y-term because we want maximun of radiation to be on pi=0, theta=90 (as in antenna arrays)
    res = np.abs(np.float_power(np.cos(x), power, dtype=complex) * np.float_power(np.cos(y+np.pi/2), power, dtype=complex))
    size = x.shape[1]
    res[:,:size//4] = np.finfo(np.float64).eps
    res[:,size//4*3:] = np.finfo(np.float64).eps
    return 20*np.log10(res)


class RadiantElement:
    """
    Summation of field contributions from each element in array, at frequency freq at theta 0°-95°, phi 0°-360°.
    Element = xPos, yPos, zPos, ElementAmplitude, ElementPhaseWeight
    Returns RadiantElement[theta, phi, elementSum]
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
    Summation of field contributions from each element in array, at frequency freq at theta 0°-95°, phi 0°-360°.
    Element = xPos, yPos, zPos, ElementAmplitude, ElementPhaseWeight
    Returns arrayFactor[theta, phi, elementSum]
    """
    
    
    def __init__(self, elements, pattern=None):
        self.elements = elements    # instance variable unique to each instance
        self.nElements = len(self.elements)
        # Phi/Theta points
        self.nPhi = 1000
        self.nTheta = 500
        phi = np.linspace(-np.pi, np.pi, self.nPhi)
        theta = np.linspace(0, np.pi, self.nTheta)
        self.PHI, self.THETA = np.meshgrid(phi, theta)
        # Element pattern
        if pattern is not None:
            if pattern.shape == (self.nPhi, self.nTheta):
                self.elePattern = pattern
            else:
                self.elePattern = pattern
        else:
            self.elePattern = cos_power_2d(self.PHI, self.THETA, power=2)
        
        
    def plot_array(self):
        if all(ele.xpos == self.elements[0].xpos for ele in self.elements):
            plt.rcParams["figure.figsize"] = [7.00, 3.50]
            plt.rcParams["figure.autolayout"] = True
            plt.xlabel('Z-position (mm)')
            plt.ylabel('Y-position (mm)')
            plt.grid()
            for ele in self.elements:
                plt.plot(ele.zpos*1000, ele.ypos*1000, marker="o", markersize=20, markeredgecolor="tomato", markerfacecolor="mediumaquamarine")
                #plt.text(ele.xpos*1000, ele.ypos*1000, f'{ele.amplitude} ∠ {ele.phase}º')
                plt.gca().annotate(f'{ele.amplitude} ∠ {ele.phase}º', (ele.zpos*1000, ele.ypos*1000))
            plt.show()
        else:
            raise ValueError('Cannot plot 3D arrays')

            
    def arrayFactor(self, freq):
        
        def CalculateRelativePhase(element_pos, Lambda, PHI, THETA):
            """
            Incident wave treated as plane wave. Phase at element is referred to phase of plane wave at origin.
            Element = element_pos
            THETA & PHI in radians
            Lambda in Hz
            See Eqn 3.1 @ https://theses.lib.vt.edu/theses/available/etd-04262000-15330030/unrestricted/ch3.pdf
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

        # Power in dB
        AFdB = 20*np.log10(np.abs(AF))
    
        return AFdB

        
    def arrayPattern(self, freq):
        return self.arrayFactor(freq) + self.elePattern
    
    def __repr__(self):
        return f'Array of {self.nElements} elements.'