from cmath import polar
from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

    
def plotHM(z, normalize=True):
    """
    Plots heatmap with complete sphere data, so that phi goes from -180º o 180º 
    and theta from 0º to 180º.

    Args:
        z: 2-dimmensinal numpy array with data
        normalize: indicates if protting must be normalized

    Raises:
        ValueError: If matrix is not 2-dimmensional
    """ 

    if len(z.shape) != 2: 
        raise ValueError("Matrix must be 2-dimensional to plot cuts")

    # Normalization
    if normalize:
        ylim_sup = 0;
        Z = z - np.max(z)
    else:
        ylim_sup = (np.max(z)//5+1)*5;
        Z = z
    
    # Plots data
    sns.set_style("white")
    fig, ax = plt.subplots(1,1, figsize=(16,8))
    cp = ax.imshow(Z, cmap='jet', vmin=ylim_sup-40, vmax=ylim_sup)
    fig.colorbar(cp) # Add a colorbar to a plot

    # Axis ticks
    ax.set_xticks(np.linspace(0, z.shape[1], num=13))
    ax.set_xticklabels([str(i)+'º' for i in np.arange(0,361,30)], rotation=0)
    ax.set_yticks(np.linspace(0, z.shape[0], num=7))
    ax.set_yticklabels([str(i)+'º' for i in np.arange(0,181,30)], rotation=0)

    # Labels
    ax.set_title('Diagram contour plot', fontsize=18)
    ax.set_xlabel('Azimuth ($\phi$)', fontsize=14)
    ax.set_ylabel('Elevation ($\\theta$)', fontsize=14)
    
    plt.show()

    
def plotCuts(z, normalize=True):
    """
    Plots azimuth and elevation cuts. Cuts are made where input z-matrix is maximum

    Args:
        z: 2-dimensional numpy array with data
        normalize: indicates if plotting must be normalized. Defaults to True.

    Raises:
        ValueError: If matrix is not 2-dimensional
    """

    if len(z.shape) != 2: 
        raise ValueError("Matrix must be 2-dimensional to plot cuts")

    # Normalization
    if normalize:
        zlim_sup = 0;
        Z = z - np.max(z)
    else:
        zlim_sup = (np.max(z)//5+1)*5;
        Z = z

    sns.set_theme()
    _, ax = plt.subplots(1,2, figsize=(16,6), sharey=True)
    max_theta, max_phi = np.unravel_index(np.argmax(z, axis=None), z.shape)

    # Plots
    ax[0].plot(Z[max_theta, :], linewidth=2)
    ax[1].plot(Z[:, max_phi], linewidth=2)
    ax[0].plot(np.repeat(np.max(Z)-3, z.shape[1]), linewidth=2, color='indianred', linestyle='dashed')
    ax[1].plot(np.repeat(np.max(Z)-3, z.shape[0]), linewidth=2, color='indianred', linestyle='dashed')

    # Set limits & axis
    ax[0].set_ylim(zlim_sup-30, zlim_sup)
    ax[1].set_ylim(zlim_sup-30, zlim_sup)
    ax[0].set_xlim(0, Z.shape[1])
    ax[1].set_xlim(0, Z.shape[0])
    ax[0].set_xticks(np.linspace(0, z.shape[1], num=13))
    ax[0].set_xticklabels([str(i)+'º' for i in np.arange(0,361,30)], rotation=0)
    ax[1].set_xticks(np.linspace(0, z.shape[0], num=7))
    ax[1].set_xticklabels([str(i)+'º' for i in np.arange(0,181,30)], rotation=0)

    # Labels
    ax[0].set_title('Horizontal cut', fontsize=18)
    ax[1].set_title('Vertical cut', fontsize=18)
    ax[0].set_xlabel('Azimuth ($\phi$)', fontsize=14)
    ax[1].set_xlabel('Elevation ($\\theta$)', fontsize=14)
    ax[0].set_ylabel('Gain (dB)', fontsize=14)

    # Place info
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.5)
    param_str = '\n'.join((f'Gain: {np.round(np.max(z),2)} dB',
                         f'HBW: {np.sum(z[max_theta, :]>np.max(z)-3)*360/z.shape[1]}º',
                         f'VBW: {np.sum(z[:, max_phi]>np.max(z)-3)*180/z.shape[0]}º'))
    ax[1].text(1.05, 1, param_str, transform=ax[1].transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.show()
    

def plotCutsPolar (z, normalize=True):
    """
    Plots azimuth and elevation cuts in polar coordinates. Cuts are made where input z-matrix is maximun

    Args:
        z: 2-dimensional numpy array with data
        normalize: indicates if plotting must be normalized. Defaults to True.
        
    Raises:
        ValueError: If matrix is not 2-dimensional
    """    
    
    if len(z.shape) != 2: 
        raise ValueError("Matrix must be 2-dimensional to plot cuts")
    
    # Normalization
    if normalize:
        zlim_sup = 0
        Z = z - np.max(z)
    else:
        zlim_sup = (np.max(z)//5+1)*5
        Z = z

    sns.set_theme()
    fig = plt.figure(figsize=(16, 8))
    
    max_theta, max_phi = np.unravel_index(np.argmax(Z, axis=None), z.shape)
    
    # Plot horizontal pattern
    ax1 = plt.subplot(1, 2, 1, polar=True)
    x1 = np.linspace(-np.pi, np.pi, z.shape[1])
    ax1.plot(x1, Z[max_theta, :], linewidth=2)
    ax1.plot(x1, np.repeat(np.max(Z)-3, z.shape[1]), linewidth=2, color='indianred', linestyle='dashed')
    ax1.set_ylim(zlim_sup-30, zlim_sup)
    ax1.set_title('Horizontal cut', fontsize=18)
    ax1.set_xlabel('Azimuth ($\phi$)', fontsize=14)

    # Plot vertical pattern
    ax2 = plt.subplot(1, 2, 2, polar=True)
    x2 = np.linspace(-np.pi/2, np.pi/2, z.shape[0])
    ax2.plot(x2, Z[:, max_phi], linewidth=2)
    ax2.plot(x1, np.repeat(np.max(Z)-3, z.shape[1]), linewidth=2, color='indianred', linestyle='dashed')
    ax2.set_ylim(zlim_sup-30, zlim_sup)
    ax2.set_title('Vertical cut', fontsize=18)
    ax2.set_xlabel('Elevation ($\\theta$)', fontsize=14)

    # Place info
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.5)
    param_str = '\n'.join((f'Gain: {np.round(np.max(z),2)} dB',
                         f'HBW: {np.sum(z[max_theta, :]>np.max(z)-3)*360/z.shape[1]}º',
                         f'VBW: {np.sum(z[:, max_phi]>np.max(z)-3)*180/z.shape[0]}º'))
    ax2.text(1.05, 1, param_str, transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        
    plt.show()