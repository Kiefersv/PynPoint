#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 08:27:30 2019

@author: kiefer
"""


import os
import math

import numpy as np
import time

from typing import Tuple
from typeguard import typechecked
from sklearn.decomposition import PCA
from scipy.ndimage import rotate

from photutils import aperture_photometry, CircularAperture
from astropy.modeling import models
from astropy.io import ascii
from uncertainties import ufloat

from pynpoint.util.analysis import fake_planet
from pynpoint.util.residuals import combine_residuals
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.ifs import sdi_scaling
from pynpoint.util.analysis import student_t, false_alarm
from pynpoint.util.image import polar_to_cartesian, center_subpixel






@typechecked
def spec_contrast_limit(path_images: str,
                        path_psf: str,
                        noise: np.ndarray,
                        mask: np.ndarray,
                        parang: np.ndarray,
                        lambdas: np.ndarray,
                        timee: np.ndarray,
                        scales: np.ndarray,
                        pixelscale: float,
                        extra_rot: float,
                        pca_number: int,
                        threshold: Tuple[str, float],
                        aperture: float,
                        residuals: str,
                        snr_inject: float,
                        position: Tuple[float, float],
                        processing_type: str) -> Tuple[float, float, float, float]:

    """
    Function for calculating the contrast limit at a specified position for a given sigma level or
    false positive fraction, both corrected for small sample statistics.

    Parameters
    ----------
    path_images : str
        System location of the stack of images (3D).
    path_psf : str
        System location of the PSF template for the fake planet (3D). Either a single image or a
        stack of images equal in size to science data.
    noise : numpy.ndarray
        Residuals of the PSF subtraction (3D) without injection of fake planets. Used to measure
        the noise level with a correction for small sample statistics.
    mask : numpy.ndarray
        Mask (2D).
    parang : numpy.ndarray
        Derotation angles (deg).
    extra_rot : float
        Additional rotation angle of the images in clockwise direction (deg).
    pca_number : int
        Number of principal components used for the PSF subtraction.
    threshold : tuple(str, float)
        Detection threshold for the contrast curve, either in terms of 'sigma' or the false
        positive fraction (FPF). The value is a tuple, for example provided as ('sigma', 5.) or
        ('fpf', 1e-6). Note that when sigma is fixed, the false positive fraction will change with
        separation. Also, sigma only corresponds to the standard deviation of a normal distribution
        at large separations (i.e., large number of samples).
    aperture : float
        Aperture radius (pix) for the calculation of the false positive fraction.
    residuals : str
        Method used for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
    snr_inject : float
        Signal-to-noise ratio of the injected planet signal that is used to measure the amount
        of self-subtraction.
    position : tuple(float, float)
        The separation (pix) and position angle (deg) of the fake planet.
    processing_type : str
        Type of post processing. Currently supported:
            Tnan: Applaying no PCA reduction and returning one wavelength avaraged image (Equivalent to Classical ADI)
            Wnan: Applaying no PCA reduction and returing one image per Wavelengths
            Tadi: Applaying ADI and creturning one wavelength avaraged image (Equivalent to IRDIS SDI if all wavelengths are the same)
            Wadi: Applaying ADI and returing one image per Wavelengths
            Tsdi: Applaying SDI and returning one wavelength avaraged image
            Wsdi: Applaying SDI and returing one image per Wavelengths
            Tsaa: Applaying SDI and ADI simultaniously and returning one wavelength avaraged image
            Wsaa: Applaying SDI and ADI simultaniously and returing one image per Wavelengths
            Tsap: Applaying SDI then ADI and returning one wavelength avaraged image
            Wsap: Applaying SDI then ADI and returing one image per Wavelengths
            Tasp: Applaying ADI then SDI and returning one wavelength avaraged image
            Wasp: Applaying ADI then SDI and returing one image per Wavelengths
        Each reduction step uses pca_number PCA components to reduce the images.

    Returns
    -------
    float
        Separation (pix).
    float
        Position angle (deg).
    float
        Contrast (mag).
    float
        False positive fraction.

    """

    images = np.load(path_images)
    psf = np.load(path_psf)

    if threshold[0] == 'sigma':
        sigma = threshold[1]

        # Calculate the FPF for a given sigma level
        fpf = student_t(t_input=threshold,
                        radius=position[0],
                        size=aperture,
                        ignore=False)

    elif threshold[0] == 'fpf':
        fpf = threshold[1]

        # Calculate the sigma level for a given FPF
        sigma = student_t(t_input=threshold,
                          radius=position[0],
                          size=aperture,
                          ignore=False)

    else:
        raise ValueError('Threshold type not recognized.')

    # Cartesian coordinates of the fake planet
    yx_fake = polar_to_cartesian(images, position[0], position[1]-extra_rot)

    # Determine the noise level
    t_noise = np.zeros_like(noise[:, 0, 0])

    for n, no in enumerate(noise):
        _, t_noise[n], _, _ = false_alarm(image=no,
                                          x_pos=yx_fake[1],
                                          y_pos=yx_fake[0],
                                          size=aperture,
                                          ignore=False)

    # Aperture properties
    im_center = center_subpixel(images)

    # Measure the flux of the star
    ll = noise.shape[0]
    lam_splites = np.sort(list(set(lambdas)))
    mag = np.zeros((len(psf)))
    flux_in = np.zeros((ll))
    star = np.zeros((ll))
    
    ap_phot = CircularAperture((im_center[1], im_center[0]), aperture)

    for f in range(ll):
        mask_f = (lam_splites[f] == lambdas)
        psf_median = np.median(psf[mask_f], axis=0)
        phot_table = aperture_photometry(psf_median, ap_phot, method='exact')
        star[f] = phot_table['aperture_sum'][0]

        # Magnitude of the injected planet
        flux_in[f] = snr_inject*t_noise[f]
        mag[mask_f] = -2.5*math.log10(flux_in[f]/star[f])

    # Inject the fake planet
    fake = fake_planet(images=images,
                       psf=psf,
                       parang=parang,
                       position=(position[0], position[1]),
                       magnitude=mag,
                       psf_scaling=1.)
    
    # apply post processing
    _, res_rot = postprocessor(images=fake,
                               parang=-1.*parang+extra_rot,
                               scales=scales,
                               pca_number=pca_number,
                               mask=mask,
                               processing_type=processing_type)

    im_res = combine_residuals(method=residuals,
                               res_rot=res_rot,
                               lam=scales,
                               processing_type=processing_type)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(im_res[0])
    plt.plot(yx_fake[1], yx_fake[0], 'rx')
    plt.savefig(str(time.time()) + 'three_little_piggies.png')

    # Measure the flux of the fake planet
    flux_out = np.zeros((im_res.shape[0]))
    for i, im in enumerate(im_res):
        flux_out[i], _, _, _ = false_alarm(image=im,
                                           x_pos=yx_fake[1],
                                           y_pos=yx_fake[0],
                                           size=aperture,
                                           ignore=False)
        
    # Calculate the amount of self-subtraction
    attenuation = flux_out/flux_in


    # Calculate the detection limit
    contrast = sigma*t_noise/(attenuation*star)

    # The flux_out can be negative, for example if the aperture includes self-subtraction regions

    for c, con in enumerate(contrast):
        if con > 0.:
            contrast[c] = -2.5*math.log10(con)
        else:
            contrast[c] = np.nan

    # assign positions with same shape as contrast
    pos_0 = np.ones_like(contrast) * position[0]
    pos_1 = np.ones_like(contrast) * position[1]
    fpf_3 = np.ones_like(contrast) * fpf

    # Separation [pix], position antle [deg], contrast [mag], FPF
    return pos_0, pos_1, contrast, fpf_3






@typechecked
def filter_scaling_calc(science_time: float,
                        flux_time: float,
                        wavelength: float,
                        delta_wavelength: float,
                        flux_filter: str):
    """
        Calculates and returns the scaling factor between science and flux

        Parameters
        ----------
        science_time : float
            Exposure time of the science images.
        flux_time : float
            Exposure time of the flux images.
        wavelength : float
            Wavelenght of the science images. Needs to be the same for all science images.
        flux_filter: str
            Type of filter used for the flux images. Currently possible filters:
            \'ND_0.0\', \'ND_1.0\', \'ND_2.0\', \'ND_3.5\''

        Returns
        -------
        float
            psf_scaling factor

    """

    path_nd_table = os.path.join(os.path.dirname(os.path.abspath(__file__)) + '/SPHERE_CPI_ND.dat')
    nd_table = ascii.read(path_nd_table)
    fl_filt = 'col2'

    if flux_filter == 'ND_0.0': fl_filt = 'col2'
    elif flux_filter == 'ND_1.0': fl_filt = 'col3'
    elif flux_filter == 'ND_2.0': fl_filt = 'col4'
    elif flux_filter == 'ND_3.5': fl_filt = 'col5'
    else:
        raise ValueError('Choosen filter does not exist')

    #Gaussian weighting
    gauss_weight = models.Gaussian1D(1, wavelength * 1000, delta_wavelength * 1000)

    #calculate nd filter effect and std deviation
    tmp_nd_wght_avrg_flux = np.average(nd_table[fl_filt].data,
                                       weights=gauss_weight(nd_table["col1"].data))
    tmp_nd_wght_std_flux=np.sqrt(np.average((nd_table[fl_filt].data-tmp_nd_wght_avrg_flux)**2,
                                            weights=gauss_weight(nd_table["col1"].data)))

    #Assigne filter scaling as ufloat
    filter_scaling = ufloat(tmp_nd_wght_avrg_flux,
                            tmp_nd_wght_std_flux)

    # Calculate psf scaling
    psf_scaling = science_time / (flux_time * filter_scaling)

    return psf_scaling






@typechecked
def postprocessor(images: np.ndarray,
                  parang: np.ndarray,
                  scales: np.ndarray,
                  pca_number: int,
                  pca_sklearn: PCA = None,
                  im_shape: Tuple[int, int, int] = None,
                  indices: np.ndarray = None,
                  mask: np.ndarray = None,
                  processing_type: str = 'Tadi'):


    """
    Function to apply different kind of post processings. If processing_type = \'Cadi\'
    and mask = None, it is equivalent to pynpoint.util.psf.pca_psf_subtraction.

    Parameters
    ----------
    images : numpy.array
        Input images which should be reduced.
    parang : numpy.ndarray
        Derotation angles (deg).
    scales : numpy.array
        Additional scaling factor of the planet flux (e.g., to correct for a neutral density
        filter). Should have a positive value.
    pca_number : int
        Number of principal components used for the PSF subtraction.
    pca_sklearn : sklearn.decomposition.pca.PCA, None
        PCA object with the basis if not set to None.
    im_shape : tuple(int, int, int), None
        Original shape of the stack with images. Required if `pca_sklearn` is not set to None.
    indices : numpy.ndarray, None
        Non-masked image indices. All pixels are used if set to None.
    mask : numpy.ndarray
        Mask (2D).
    processing_type : str
        Type of post processing. Currently supported:
            Tnan: Applaying no PCA reduction and returning one wavelength avaraged image (Equivalent to Classical ADI)
            Wnan: Applaying no PCA reduction and returing one image per Wavelengths
            Tadi: Applaying ADI and creturning one wavelength avaraged image (Equivalent to IRDIS SDI if all wavelengths are the same)
            Wadi: Applaying ADI and returing one image per Wavelengths
            Tsdi: Applaying SDI and returning one wavelength avaraged image
            Wsdi: Applaying SDI and returing one image per Wavelengths
            Tsaa: Applaying SDI and ADI simultaniously and returning one wavelength avaraged image
            Wsaa: Applaying SDI and ADI simultaniously and returing one image per Wavelengths
            Tsap: Applaying SDI then ADI and returning one wavelength avaraged image
            Wsap: Applaying SDI then ADI and returing one image per Wavelengths
            Tasp: Applaying ADI then SDI and returning one wavelength avaraged image
            Wasp: Applaying ADI then SDI and returing one image per Wavelengths
        Each reduction step uses pca_number PCA components to reduce the images.

    Returns
    -------
    numpy.ndarray
        Residuals of the PSF subtraction.
    numpy.ndarray
        Derotated residuals of the PSF subtraction.

    """

    if mask is None:
        mask = 1.
    
    lam_splits = np.sort(list(set(scales)))
    tim_splits = np.sort(list(set(parang)))
    
    if processing_type not in ['Wnan', 'Tnan', 'Wadi', 'Tadi']:
        if im_shape is not None:
            swup = np.zeros((im_shape[0], im_shape[1]*im_shape[2]))
            if indices is not None:
                swup[:, indices] = images
            else:
                swup = images
            ims = swup.reshape(im_shape)
    
        else:
            ims = images
            
    
        pca_sklearn = None
        im_shape = None
        indices = None
        
        res_raw = np.zeros_like(ims)
        res_rot = np.zeros_like(ims)
    
    # --- For backward compatablitiy
    else:
        ims = images
        if im_shape is not None:
            swup = np.zeros((im_shape[0], im_shape[1]*im_shape[2]))
            if indices is not None:
                swup[:, indices] = images
            else:
                swup = images
            ims_size = swup.reshape(im_shape)
        else:
            ims_size = ims

        res_raw = np.zeros_like(ims_size)
        res_rot = np.zeros_like(ims_size)

    #----------------------------------------- List of different processing
    # No reduction
    if processing_type in ['Wnan', 'Tnan']:

        res_raw = ims
        for j, item in enumerate(parang):
            res_rot[j, ] = rotate(ims[j, ], item, reshape=False)



    # Wavelength specific adi
    elif processing_type in ['Wadi', 'Tadi']:

        for ii, lam_i in enumerate(lam_splits):
            mask_i = (scales == lam_i)
            res_raw_i, res_rot_i = pca_psf_subtraction(images=ims[mask_i]*mask,
                                                       angles=parang[mask_i],
                                                       scales=np.array([None]),
                                                       pca_number=pca_number,
                                                       pca_sklearn=pca_sklearn,
                                                       im_shape=im_shape,
                                                       indices=indices)

            res_raw[mask_i] = res_raw_i
            res_rot[mask_i] = res_rot_i



    # SDI for each time frame
    elif processing_type in ['Wsdi', 'Tsdi']:

        im_scaled, _, _ = sdi_scaling(ims, scales)
        for ii, tim_i in enumerate(tim_splits):
            mask_i = (parang == tim_i)
            res_raw_i, res_rot_i = pca_psf_subtraction(images=im_scaled[mask_i]*mask,
                                                       angles=parang[mask_i],
                                                       scales=scales[mask_i],
                                                       pca_number=pca_number,
                                                       pca_sklearn=pca_sklearn,
                                                       im_shape=im_shape,
                                                       indices=indices)

            res_raw[mask_i] = res_raw_i
            res_rot[mask_i] = res_rot_i



    # SDI and ADI simultaniously
    elif processing_type in ['Wsaap', 'Tsaa']:
        im_scaled, _, _ = sdi_scaling(ims, scales)

        res_raw, res_rot = pca_psf_subtraction(images=im_scaled*mask,
                                               angles=parang,
                                               scales=scales,
                                               pca_number=pca_number,
                                               pca_sklearn=pca_sklearn,
                                               im_shape=im_shape,
                                               indices=indices)



    # SDI then ADI
    elif processing_type in ['Wsap', 'Tsap']:
        res_raw_int = np.zeros_like(res_raw)

        im_scaled, _, _ = sdi_scaling(ims, scales)
        for ii, tim_i in enumerate(tim_splits):
            mask_i = (parang == tim_i)
            res_raw_i, _ = pca_psf_subtraction(images=im_scaled[mask_i]*mask,
                                               angles=np.array([None]),
                                               scales=scales[mask_i],
                                               pca_number=pca_number,
                                               pca_sklearn=pca_sklearn,
                                               im_shape=im_shape,
                                               indices=indices)

            res_raw_int[mask_i] = res_raw_i

        for jj, lam_j in enumerate(lam_splits):
            mask_j = (scales == lam_j)
            res_raw_i, res_rot_i = pca_psf_subtraction(images=res_raw_int[mask_j]*mask,
                                                       angles=parang[mask_j],
                                                       scales=np.array([None]),
                                                       pca_number=pca_number,
                                                       pca_sklearn=pca_sklearn,
                                                       im_shape=im_shape,
                                                       indices=indices)

            res_raw[mask_j] = res_raw_i
            res_rot[mask_j] = res_rot_i



    # ADI then SDI
    elif processing_type in ['Wasp', 'Tasp']:
        res_raw_int = np.zeros_like(res_raw)

        for jj, lam_j in enumerate(lam_splits):
            mask_j = (scales == lam_j)
            res_raw_i, _ = pca_psf_subtraction(images=ims[mask_j]*mask,
                                               angles=np.array([None]),
                                               scales=np.array([None]),
                                               pca_number=pca_number,
                                               pca_sklearn=pca_sklearn,
                                               im_shape=im_shape,
                                               indices=indices)

            res_raw_int[mask_j] = res_raw_i

        im_scaled, _, _ = sdi_scaling(res_raw_int, scales)
        for ii, tim_i in enumerate(tim_splits):
            mask_i = (parang == tim_i)
            res_raw_i, res_rot_i = pca_psf_subtraction(images=im_scaled[mask_i]*mask,
                                                       angles=parang[mask_i],
                                                       scales=scales[mask_i],
                                                       pca_number=pca_number,
                                                       pca_sklearn=pca_sklearn,
                                                       im_shape=im_shape,
                                                       indices=indices)

            res_raw[mask_i] = res_raw_i
            res_rot[mask_i] = res_rot_i



    else:
        # Error message if unknown processing type
        st = 'Processing type ' + processing_type + ' is not supported'
        raise ValueError(st)



    return res_raw, res_rot
