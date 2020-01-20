# -*- coding: utf-8 -*-
__author__ = "Sven Kiefer, Alexander Bohn"

import numpy as np
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture
import photutils as pu
from scipy.special import erf
from astropy.modeling import fitting, models
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clip
from astropy.table import Table


def get_false_alarm_probability(image_in,
                                rough_pos,
                                aperture_radius,
                                method='exact',
                                skip_closest_apertures=False,
                                plot=False):

    # Get image shape
    im_shape = image_in.shape

    # Perform Gaussian fit, if required
    if method == "fit":

        # Cut image around pos to resolution of 5 fwhm for Gaussian fit
        image_cut = Cutout2D(data=image_in,
                             position=rough_pos,
                             size=(4 * aperture_radius, 4 * aperture_radius),
                             mode="partial",
                             fill_value=0.).data

        # Fit the source in the science image using a 2D Gaussian
        gauss_science_init = models.Gaussian2D(amplitude=image_cut[image_cut.shape[0] / 2, image_cut.shape[1] / 2],
                                               x_mean=image_cut.shape[0] / 2.,
                                               y_mean=image_cut.shape[1] / 2.,
                                               x_stddev=aperture_radius,
                                               y_stddev=aperture_radius,
                                               theta=0.)

        fit_gauss = fitting.LevMarLSQFitter()

        y, x = np.mgrid[:image_cut.shape[1], :image_cut.shape[0]]
        gauss_science = fit_gauss(gauss_science_init, x, y, image_cut)

        tmp_cc_pos = (gauss_science.x_mean.value, gauss_science.y_mean.value)

        # Transform tmp_cc_pos to cc_pos
        pos = (rough_pos[0] + (tmp_cc_pos[0] - image_cut.shape[0] / 2.) + .5,
               rough_pos[1] + (tmp_cc_pos[1] - image_cut.shape[1] / 2.) + .5)

    else:
        pos = rough_pos

    # Vector from image center to point source
    offset_vector = np.asarray(pos)-np.asarray(im_shape)/2.

    # Radial separation of point source
    separation = np.linalg.norm(offset_vector)

    # Position angle of point source
    pos_angle = np.arctan2(offset_vector[0], offset_vector[1])

#    print(separation, aperture_radius)

    # Calculate number of apertures
    number_of_apertures = int(np.floor(np.pi/np.arcsin(aperture_radius/separation)))

    # Azimuthal angles of the apertures
    azimuthal_angles = np.linspace(0, 2.*np.pi, number_of_apertures+1)[:-1]

    # Rotation Matrix
    def rot_matrx(phi):
        return np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

    # Get cartesian positions of the apertures
    aperture_pos = np.array([np.asarray(im_shape)/2. + np.dot(rot_matrx(i), offset_vector) for i in azimuthal_angles]).T

    # Define science and background apertures
    science_aperture = CircularAperture(positions=aperture_pos[:, 0],
                                        r=aperture_radius)

    if skip_closest_apertures:
        bg_apertures = CircularAperture(positions=aperture_pos[:, 2:-1],
                                        r=aperture_radius)
    else:
        bg_apertures = CircularAperture(positions=aperture_pos[:, 1:],
                                        r=aperture_radius)

    # Determine science flux
    science_flux = pu.aperture_photometry(data=image_in,
                                          apertures=science_aperture,
                                          method="exact")["aperture_sum"].data[0]

    # Determine background fluxes
    bg_fluxes = pu.aperture_photometry(data=image_in,
                                       apertures=bg_apertures,
                                       method="exact")["aperture_sum"].data

    # Perform sigma clipping of background fluxes
    bg_fluxes_clipped = sigma_clip(bg_fluxes, sigma=3, maxiters=5)

    # Perform statistics
    bg_rms = np.sqrt(np.mean(bg_fluxes_clipped**2))
    bg_mean = np.average(bg_fluxes_clipped)
    bg_std = np.std(bg_fluxes_clipped)

    signal_to_noise = science_flux/bg_rms
    # Gaussain error propagation
    snr_sigma = signal_to_noise * np.sqrt((bg_std/science_flux)**2 + (bg_std/bg_rms)**2)

    sigma = np.abs((bg_mean-science_flux)/bg_std)

    signal = science_flux-bg_mean

    fap = 1. - erf(sigma/np.sqrt(2))

    if plot:
        f, ax = plt.subplots()
        plt.imshow(image_in, origin="lower left")
        science_aperture.plot(color="r", ax=ax)
        bg_apertures.plot(color="w", ax=ax)
        plt.show()

    return Table(data=[(pos[0], ), (pos[1], ), (signal_to_noise, ), (sigma, ), (fap, ), (signal, ), (snr_sigma, )],
                 names=("pos_x", "pos_y", "signal_to_noise", "sigma", "fap", "signal", "snr_sigma"))
