import glob
import warnings
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import scoreatpercentile

def load_field(fileglob='CK1_A01_1_*.tif'):
    filenames = sorted(glob.glob(fileglob))
    assert len(filenames) > 0, 'tiff files not found'
    imlist = [Image.open(f) for f in filenames]
    # field will be Width * Height * Channels
    field = np.array(
        [np.array(im.getdata(), dtype=np.uint16).
         reshape(tuple(reversed(im.size)))
         for im in imlist]
        )
    field = field.transpose(1,2,0)
    return field

def show_panels(field):
    (w, h, d) = field.shape
    panels = np.ravel(field.transpose(2,1,0), 'C').reshape(w*d,h).transpose(1,0)
    plt.imshow(panels, vmin=0, vmax=2**16-1)

def show_pseudocolor(field):
    if field.shape[2] != 3:
        warnings.warn('field does not contain exactly 3 channels')
    plt.imshow(field/float(2**16), vmin=0.0, vmax=1.0)
    plt.gca().xaxis.set_label_position('top')

def check_decimation(field, bits):
    k = 2**bits;
    for c in range(0, field.shape[2]):
        print c, ':', np.histogram(field[:,:,c] & (k-1), k, (0,k))[0]

def build_histograms(field, num_bins):
    channels = np.dsplit(field, field.shape[2])
    bins = np.linspace(0, 2**16, num_bins+1)
    hist = np.array([np.histogram(c, bins)[0] for c in channels]).T
    return (bins, hist)

def show_histograms(field):
    bins, hist = build_histograms(field, 2**8)
    # +1 prevents taking log of 0, and only imperceptibly alters the result
    log_hist = np.log(hist+1)
    log_hist[log_hist == -np.inf] = 0  
    color_idx = list(enumerate(('red', 'green', 'blue', 'orange')))
    for i, color in color_idx[0:field.shape[2]]:
        plt.plot(bins[0:-1], log_hist[:,i], color=color)
    plt.xlim(0, bins[-2])

def auto_contrast(field):
    (w, h, d) = field.shape

    # initial pre-scaling to help background calculation resolution
    peak = scoreatpercentile(field.reshape(w*h, d), 99.9)
    field_ac = field * (float(2**16-1) / peak)  # do scaling to full-scale

    # calculate background (bottom of dynamic range) and subtract it, shifting it to zero
    bins, hist = build_histograms(field_ac, 2**8)
    # take histogram peak as background (works for images seen so far...)
    background = hist.argmax(0) * bins[1]
    #### alternative background calculation using median values 
    ###background = median(field.reshape(w*h, d), 0)
    # do background subtraction
    field_ac = field_ac - background            

    # calculate Nth percentile (top of dynamic range) and rescale it to the
    # full-scale intensity value.
    # must calculate percentile from new bg-corrected values
    peak = scoreatpercentile(field_ac.reshape(w*h, d), 99.9)
    # do scaling to full-scale
    field_ac = field_ac * (float(2**16-1) / peak)
    field_ac.clip(0, 2**16-1, field_ac)

    return field_ac.astype('uint16')

def show_pc_hist(field):
    plt.axes((.15, .2, .7, .7))
    show_pseudocolor(field)
    
    plt.axes((.1, .1, .8, .09))
    show_histograms(field)
