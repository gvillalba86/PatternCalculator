import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Plot cut
def plot_polar (x, y, normalize=False):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='polar')
    if normalize:
        ylim_sup = 0;
        xlim_inf = -40;
        Y = y - np.max(y)
    else:
        ylim_sup = (np.max(y)//5+1)*5;
        xlim_inf = ylim_sup-40;
        Y = y
    ax.plot(x, Y)
    ax.plot(x, np.repeat(np.max(Y)-3, len(x)), color='red', linestyle='dashed', linewidth=2)
    ax.set_ylim((xlim_inf, ylim_sup))
    ax.set_rlabel_position(0)
    plt.show()

    
def plotHM(z, normalize=True):
    sns.set_style("white")
    fig, ax = plt.subplots(1,1, figsize=(16,8))
    if normalize:
        ylim_sup = 0;
        Z = z - np.max(z)
    else:
        ylim_sup = (np.max(z)//5+1)*5;
        Z = z
    cp = ax.imshow(Z, cmap='jet', vmin=ylim_sup-40, vmax=ylim_sup)
    fig.colorbar(cp) # Add a colorbar to a plot
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
    Plots azimuth and elevation cuts. Cuts are made where z-matrix is maximun

    Args:
        z: _description_
        normalize: _description_. Defaults to True.
    """    
    if len(z.shape) != 2: 
        raise ValueError("Matrix must be 2-dimmensional to plot cuts")
    if normalize:
        ylim_sup = 0;
        Z = z - np.max(z)
    else:
        ylim_sup = (np.max(z)//5+1)*5;
        Z = z
    sns.set_theme()
    #sns.set_context("talk")
    fig, ax = plt.subplots(1,2, figsize=(16,6), sharey=True)
    max_theta, max_phi = np.unravel_index(np.argmax(z, axis=None), z.shape)
    # Plots
    ax[0].plot(Z[max_theta, :], linewidth=2)
    ax[1].plot(Z[:, max_phi], linewidth=2)
    ax[0].plot(np.repeat(ylim_sup, z.shape[1]), linewidth=2, color='indianred', linestyle='dashed')
    ax[1].plot(np.repeat(ylim_sup, z.shape[0]), linewidth=2, color='indianred', linestyle='dashed')
    # Set limits & axis
    ax[0].set_ylim(ylim_sup-30, ylim_sup)
    ax[1].set_ylim(ylim_sup-30, ylim_sup)
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