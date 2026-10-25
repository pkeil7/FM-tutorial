import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.feature import ShapelyFeature
from matplotlib.colors import ListedColormap
from scipy import ndimage
import cartopy.io.shapereader as shpreader
import warnings
import torch


def plot_netcdf(images, masks, index, ax, name, coords):
    warnings.filterwarnings("ignore")
    # create colors for cmap
    nws_precip_colors = [
        "#04e9e7",  # 0.01 - 0.10 inches
        "#019ff4",  # 0.10 - 0.25 inches
        "#0300f4",  # 0.25 - 0.50 inches
        "#02fd02",  # 0.50 - 0.75 inches
        "#01c501",  # 0.75 - 1.00 inches
        "#008e00",  # 1.00 - 1.50 inches
        "#fdf802",  # 1.50 - 2.00 inches
        "#e5bc00",  # 2.00 - 2.50 inches
        "#fd9500",  # 2.50 - 3.00 inches
        "#fd0000",  # 3.00 - 4.00 inches
        "#d40000",  # 4.00 - 5.00 inches
        "#bc0000",  # 5.00 - 6.00 inches
        "#f800fd",  # 6.00 - 8.00 inches
        "#9854c6",  # 8.00 - 10.00 inches
        "#1f0a2e",  # 10.00+
    ]
    
    images = images.detach().numpy().squeeze()
    masks = masks.detach().numpy().squeeze()
    
    precip_colormap = matplotlib.colors.ListedColormap(nws_precip_colors)
    levels = [0.01, 0.1, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
              6.0, 8.0, 10.]
    norm = matplotlib.colors.BoundaryNorm(levels, 15)
    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.family': "serif"})

    image = images[index]
    image = np.where(image < 0, 0, image)

    mask = masks[0]
    new_mask = ndimage.binary_erosion(mask, iterations=1)
    mask = np.logical_xor(new_mask, mask)

    if name == "Input":
        image *= masks[0]

    image = np.where(mask == False, image, np.nan)

    cmap = precip_colormap
    new_cmap = cmap(np.arange(cmap.N))
    new_cmap[:, -1] = np.ones(cmap.N)
    new_cmap[:, -1][0] = 0
    new_cmap = ListedColormap(new_cmap)
    new_cmap.set_bad('black', 1.)
    cmap.set_bad('black', 1.)

    ax.pcolormesh(coords['lon'], coords['lat'], image, cmap=new_cmap, norm=norm)

    # Read Natural Earth data
    shpfilename = shpreader.natural_earth(resolution='10m',
                                          category='cultural',
                                          name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    surround_countries = ["Belgium", "Netherlands", "Denmark", "Italy", "Switzerland", "Poland", "Czech Republic",
                          "Luxembourg", "France", "Sweden", "Austria", "Slovenia"]
    surround_country_records = []
    germany = None
    for record in reader.records():
        if "Germany" in record.attributes["NAME_LONG"]:
            germany = record
        for country in surround_countries:
            if country in record.attributes["NAME_LONG"]:
                surround_country_records.append(record)
    shape_feature = ShapelyFeature([germany.geometry], ccrs.PlateCarree(), facecolor="white", edgecolor='darkgray',
                                   lw=1, alpha=0.3)
    ax.add_feature(shape_feature)

    for record in surround_country_records:
        shape_feature = ShapelyFeature([record.geometry], ccrs.PlateCarree(), facecolor="gray", edgecolor='darkgray',
                                       lw=1, alpha=0.3)
        ax.add_feature(shape_feature)
    ax.add_feature(cartopy.feature.OCEAN)

    ax.add_feature(cartopy.feature.COASTLINE, edgecolor="darkgray", linewidth=1)
    ax.add_feature(cartopy.feature.BORDERS, color="darkgray", linewidth=1)

    ax.set_title(name)

def plot_netcdf_tensor(images, ax, name):
        # create colors for cmap
    nws_precip_colors = [
        "#04e9e7",  # 0.01 - 0.10 inches
        "#019ff4",  # 0.10 - 0.25 inches
        "#0300f4",  # 0.25 - 0.50 inches
        "#02fd02",  # 0.50 - 0.75 inches
        "#01c501",  # 0.75 - 1.00 inches
        "#008e00",  # 1.00 - 1.50 inches
        "#fdf802",  # 1.50 - 2.00 inches
        "#e5bc00",  # 2.00 - 2.50 inches
        "#fd9500",  # 2.50 - 3.00 inches
        "#fd0000",  # 3.00 - 4.00 inches
        "#d40000",  # 4.00 - 5.00 inches
        "#bc0000",  # 5.00 - 6.00 inches
        "#f800fd",  # 6.00 - 8.00 inches
        "#9854c6",  # 8.00 - 10.00 inches
        "#1f0a2e",  # 10.00+
    ]
    precip_colormap = matplotlib.colors.ListedColormap(nws_precip_colors)
    levels = [0.01, 0.1, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
              6.0, 8.0, 10.]
    norm = matplotlib.colors.BoundaryNorm(levels, 15)

    cmap = precip_colormap
    new_cmap = cmap(np.arange(cmap.N))
    new_cmap[:, -1] = np.ones(cmap.N)
    new_cmap[:, -1][0] = 0
    new_cmap = ListedColormap(new_cmap)
    new_cmap.set_bad('black', 1.)
    cmap.set_bad('black', 1.)
    ax.imshow(images.squeeze().flip(dims=(0,)), cmap=new_cmap, norm=norm)
    ax.set_title(name)


def calculate_CRPS(ens_prediction, gts, mask) :
    # CRPS over the masked-out (infilled) region only.
    assert ens_prediction.ndim == 5, "Tensor must have shape (num_realisations, n_test, channels, H, W)"
    assert mask.ndim == 3, "Mask must have shape (channels, H, W)"
    mask_bool = (1 - mask[0]).bool()          # (H, W)
    ens_masked = ens_prediction[:, :, 0, mask_bool]  # (num_realisations, n_test, channels=1, H, W) --> ( num_realisations, n_test, P) where P is the number of masked-out pixels
    truth_masked = gts[:, 0,  mask_bool]                     # (n_test, channels=1, H, W)  (n_test, P)
    
    # CRPS ensemble estimator:  E|X - y| - 0.5 * E|X - X'|
    term1 = (ens_masked - truth_masked.unsqueeze(0)).abs().mean(dim=0)   # (n_test, P)
    term2 = torch.zeros_like(term1)
    num_realisations = ens_prediction.shape[0]
    for i in range(num_realisations):
        term2 += (ens_masked - ens_masked[i].unsqueeze(0)).abs().sum(dim=0)
    term2 = term2 / (num_realisations ** 2)                              # (1/M^2) sum_i sum_j |x_i - x_j|
    
    crps_per_pixel = term1 - 0.5 * term2                 # per-pixel CRPS
    crps = crps_per_pixel.mean().item()
    return crps

def MAE_masked(prediction, gts, mask):
    mask_bool = (1 - mask[0]).bool()          # (H, W)
    truth_masked = gts[:, 0,  mask_bool]                     # (n_test, P)
    if prediction.ndim == 5:
        prediction_masked = prediction[:, :, 0, mask_bool]  # (num_realisations, n_test, P)
        mae = (prediction_masked - truth_masked.unsqueeze(0)).abs().mean().item()
        return mae
    elif prediction.ndim == 4:
        prediction_masked = prediction[:, 0, mask_bool]  # (n_test, P)
        mae = (prediction_masked - truth_masked).abs().mean().item()
        return mae
    else:
        raise ValueError("Prediction tensor must have shape (num_realisations, n_test, channels, H, W) or (n_test, channels, H, W)")

def radial_psd(img):
    """Radially-averaged 2D power spectrum. img: (..., H, W) -> (..., n_bins)."""
    power = np.abs(np.fft.fftshift(np.fft.fft2(img, axes=(-2, -1)), axes=(-2, -1))) ** 2
    h, w = img.shape[-2:]
    y, x = np.indices((h, w))
    r = np.hypot(y - h // 2, x - w // 2).astype(int)
    n_bins = r.max() + 1
    flat = power.reshape(-1, h * w)
    r_flat = r.ravel()
    counts = np.bincount(r_flat, minlength=n_bins)
    sums = np.stack([np.bincount(r_flat, weights=row, minlength=n_bins) for row in flat])
    return (sums / counts).reshape(img.shape[:-2] + (n_bins,))

def prepare_cartopy() :
    from cartopy.io import shapereader
    shapereader.natural_earth(resolution='110m', category='physical', name='coastline')
    shapereader.natural_earth(resolution='110m', category='cultural', name='admin_0_boundary_lines_land')
    