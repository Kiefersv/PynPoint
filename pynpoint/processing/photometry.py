# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:01:49 2019

@author: Sven Kiefer
"""

import sys
import math
import warnings

import numpy as np

from typing import List

from photutils import aperture_photometry, CircularAperture
from uncertainties import ufloat, umath
from astropy.modeling import models, fitting

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.sdi import filter_scaling_calc
from pynpoint.util.FalseAlarmTests import get_false_alarm_probability
from pynpoint.util.image import crop_image





class SdiAperturePhotometryModule(ProcessingModule):
    """
    Module to measure the flux and position of a planet by performing aperture photometry
    """

    __author__ = 'Alex Bohn, Sven Kiefer'

    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 psf_in_tag: str,
                 flux_position_tag: str,
                 rough_position: List[int],
                 flux_filter: str = 'ND_0.0',
                 psf_scaling: float = None,
                 pixscale_corr: ufloat = ufloat(0.01227, 0.00002),
                 TN: ufloat = ufloat(-1.75, 0.1),
                 cutout_size: int = 21,
                 fit_each_image: bool = False,
                 aperture_size: str = "fwhm") -> None:
        """
        Constructor of AperturePhotometryModule_background.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the science images that are read as input.
        psf_in_tag : str
            Tag of the reference pdf
        flux_position_tag : str
            Output tag of all calculated vlaues
        rough_position : List[int, int]
            Rough position of the object which should be analysed
        psf_scaling: float
            Optional parameter to linearly scale the psf
        pixscale_corr : ufloat
            Pixel scale of the images with uncertainties
        TN : ufloat

        cutout_size : int
            Size of the image after centring around the rough position
        fit_each_image : bool
            If True, each image gets calculated individually. If False, the flux contrast
            of the median will be calculated.
        apperture_size : str
            Either 'fwhm' or anything else. Used to dermine the false alarm probability

        Returns
        -------
        NoneType
            None

        """

        super(SdiAperturePhotometryModule, self).__init__(name_in)

        #add relevent ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)
        self.m_flux_position_port = self.add_output_port(flux_position_tag)

        #assigne parameters
        self.m_rough_position = rough_position
        self.m_flux_filter = flux_filter
        self.m_psf_scaling = psf_scaling
        self.m_pixscale_corr = pixscale_corr
        self.m_TN = TN
        self.m_cutout_size = cutout_size
        self.m_fit_each_image = fit_each_image
        self.m_aperture_size = aperture_size

        self.m_image_in_tag = image_in_tag
        self.m_psf_in_tag = psf_in_tag


    def _perform_aperture_photometry(self, image_arr, flux_arr):
        """
        Perform aperture photometry an science frames as well as flux frames
        """

        #initalize variables
        pos_x_arr = np.array([])
        pos_y_arr = np.array([])
        fwhm_science_arr = np.array([])
        sep_arr = np.array([])
        PA_arr = np.array([])
        signal_science_arr = np.array([])
        converged_arr = np.array([], dtype=bool)

        #loop over all images
        for tmp_im in image_arr[:,]:

            # get correct position of object
            tmp_im_cut = crop_image(image=tmp_im,
                                    center=(int(self.m_rough_position[1]),
                                            int(self.m_rough_position[0])),
                                    size=self.m_cutout_size)

            # model of the Gaussian fit
            gauss_init = models.Gaussian2D(amplitude=np.max(tmp_im_cut),
                                           x_mean=np.unravel_index(np.argmax(tmp_im_cut),
                                                                   shape=tmp_im_cut.shape)[0],
                                           y_mean=np.unravel_index(np.argmax(tmp_im_cut),
                                                                   shape=tmp_im_cut.shape)[1],
                                           x_stddev=5.,
                                           y_stddev=5.,
                                           theta=0.)

            fit_gauss = fitting.LevMarLSQFitter()

            #Fit the Gaussian to the current frame and save the result
            y, x = np.mgrid[:tmp_im_cut.shape[1], :tmp_im_cut.shape[0]]
            gauss_science = fit_gauss(gauss_init, x, y, tmp_im_cut)

            tmp_cc_pos = (gauss_science.x_mean.value,
                          gauss_science.y_mean.value)

            fwhm_science_arr = np.append(fwhm_science_arr,
                                         np.average([gauss_science.x_stddev.value * math.sqrt(8.*math.log(2.)),
                                                     gauss_science.y_stddev.value * math.sqrt(8.*math.log(2.))]))


            #Check for convergenc and assigne position with uncertainties
            if fit_gauss.fit_info['ierr'] not in [1, 2, 3, 4]:
                position = self.m_rough_position
                err_params = np.ones(5) * 5
                converged = False
            else:
                position = [int(self.m_rough_position[0]) + (tmp_cc_pos[0] - tmp_im_cut.shape[0] / 2.) + .5,
                            int(self.m_rough_position[1]) + (tmp_cc_pos[1] - tmp_im_cut.shape[1] / 2.) + .5]
                try:
                    err_params = np.sqrt(np.diag(fit_gauss.fit_info['param_cov']))
                except:
                    err_params = np.ones(5) * 5
                converged = True


            #Calcualte false alarm probability, checking th fwhm criteria
            if self.m_aperture_size == "fwhm":
                if np.isnan(fwhm_science_arr[-1]) or fwhm_science_arr[-1] > 10 or fwhm_science_arr[-1] < 0.5:
                    snr = get_false_alarm_probability(image_in=tmp_im,
                                                      rough_pos=position,
                                                      aperture_radius=4,
                                                      method="exact",
                                                      skip_closest_apertures=True,
                                                      plot=False)
                else:
                    snr = get_false_alarm_probability(image_in=tmp_im,
                                                      rough_pos=position,
                                                      aperture_radius=fwhm_science_arr[-1],
                                                      method="exact",
                                                      skip_closest_apertures=True,
                                                      plot=False)
            else:
                snr = get_false_alarm_probability(image_in=tmp_im,
                                                  rough_pos=position,
                                                  aperture_radius=self.m_aperture_size,
                                                  method="exact",
                                                  skip_closest_apertures=True,
                                                  plot=False)


            # If snr negative set to high std_dev // if snr positive, claculate signal with uncertainties
            if snr["signal"].data[0] <= 0:
                warnings.warn("Negative signal encountered. Contrast extraction not precise.")
                converged = False
                signal_science_arr = np.append(signal_science_arr,ufloat(1, 1e5))
            else:
                signal_science_arr = np.append(signal_science_arr,
                                               ufloat(snr["signal"].data[0], np.abs(snr["signal"].data[0] / snr["sigma"].data[0])))

            #add positions with uncertainties
            pos_x = ufloat(snr["pos_x"].data[0], err_params[1])
            pos_y = ufloat(snr["pos_y"].data[0], err_params[2])

            #add center with uncertainties                
            center_x = ufloat(tmp_im.shape[0] / 2. - .5,
                              2.5e-3 / self.m_pixscale_corr.nominal_value)  # Center uncertainty of 2.5mas (SPHERE manual)
            center_y = ufloat(tmp_im.shape[1] / 2. - .5,
                              2.5e-3 / self.m_pixscale_corr.nominal_value)  # Center uncertainty of 2.5mas (SPHERE manual)

            offset_x = pos_x - center_x
            offset_y = pos_y - center_y


            # Get sep and PA with correct plate scale and TN
            sep = umath.sqrt(offset_x ** 2 + offset_y ** 2) * self.m_pixscale_corr
            PA = umath.atan2(offset_x, -offset_y) + np.pi + (self.m_TN / 180. * np.pi)


            # # aperture photometry of companion
            # aperture_science = CircularAperture(positions=(pos_x.nominal_value, pos_y.nominal_value),
            #                                     r=self.m_aperture)


            # add to other arrays
            pos_x_arr = np.append(pos_x_arr, pos_x.nominal_value)
            pos_y_arr = np.append(pos_y_arr, pos_y.nominal_value)
            sep_arr = np.append(sep_arr, sep)
            PA_arr = np.append(PA_arr, PA)
            converged_arr =  np.append(converged_arr, converged)

            #End of For loop
            #----------------------------------------------------------------------------------------------------------------



        #add uncertainties to fwhm
        fwhm_science_std_scaled = ufloat(1., np.std(fwhm_science_arr[converged_arr])/np.average(fwhm_science_arr[converged_arr]))

        signal_flux_arr = np.zeros(len(flux_arr[:, ]))
        fwhm_flux_arr = np.zeros(len(flux_arr[:, ]))


        #loop over flux array
        for i, tmp_flux_im in enumerate(flux_arr[:, ]):

            # crop flux image for Gaussian fit (FWHM determination)
            tmp_flux_crop = crop_image(image=tmp_flux_im,
                                       center=None,
                                       size=41,
                                       copy=True)

            # model of the Gaussian fit for flux image
            gauss_init_flux = models.Gaussian2D(amplitude=np.max(tmp_flux_crop),
                                                x_mean = np.unravel_index(np.argmax(tmp_flux_crop), shape=tmp_flux_crop.shape)[0],
                                                y_mean = np.unravel_index(np.argmax(tmp_flux_crop), shape=tmp_flux_crop.shape)[1],
                                                x_stddev=5.,
                                                y_stddev=5.,
                                                theta=0.)

            fit_gauss_flux = fitting.LevMarLSQFitter()
            
            #fit gaussian to flux images
            y_flux, x_flux = np.mgrid[:tmp_flux_crop.shape[1], :tmp_flux_crop.shape[0]]
            gauss_flux = fit_gauss_flux(gauss_init_flux, x_flux, y_flux, tmp_flux_crop)

            #assigne errors to the flux measurements
            if self.m_aperture_size == "fwhm":
                fwhm_flux_arr[i] = np.average([gauss_flux.x_stddev.value * math.sqrt(8.*math.log(2.)),
                                               gauss_flux.y_stddev.value * math.sqrt(8.*math.log(2.))])
                aperture_flux = CircularAperture(positions=(flux_arr.shape[-2] / 2. - .5, flux_arr.shape[-1] / 2. - .5),
                                                 r=fwhm_flux_arr[-1])
            else:
                fwhm_flux_arr[i] = 1.
                aperture_flux = CircularAperture(positions=(flux_arr.shape[-2] / 2. - .5, flux_arr.shape[-1] / 2. - .5),
                                                 r=self.m_aperture_size)


            signal_flux_arr[i] = aperture_photometry(tmp_flux_im, aperture_flux, method='exact')['aperture_sum'].data[0]
            
            #End of For loop
            #------------------------------------------------------------------------------------------------------------------------------



        # add uncertainties to the flux calculations
        fwhm_flux_std_scaled = ufloat(1., np.std(fwhm_flux_arr)/np.average(fwhm_flux_arr))
        signal_flux = ufloat(np.median(signal_flux_arr), np.std(signal_flux_arr)) * fwhm_science_std_scaled


        #calculate contrast in magnitudes
        magnitude_arr = np.array([-2.5 * umath.log10(i / (signal_flux * self.m_psf_scaling * fwhm_flux_std_scaled))
                                  for i in signal_science_arr])


        #Make everything neat for printing
        res = np.asarray((np.average(pos_x_arr[converged_arr]),
                          np.average(pos_y_arr[converged_arr]),
                          np.average([i.nominal_value for i in sep_arr[converged_arr]]),
                          np.std([i.nominal_value for i in sep_arr[converged_arr]]) + np.average([i.std_dev for i in sep_arr[converged_arr]]),
                          np.rad2deg(np.average([i.nominal_value for i in PA_arr[converged_arr]])),
                          np.rad2deg(np.std([i.nominal_value for i in PA_arr[converged_arr]]) + np.average([i.std_dev for i in PA_arr[converged_arr]])),
                          np.average([i.nominal_value for i in magnitude_arr[converged_arr]]),
                          np.std([i.nominal_value for i in magnitude_arr[converged_arr]]) + np.average([i.std_dev for i in magnitude_arr[converged_arr]]),
                          np.any(converged_arr)))

        return res #[PosX (0), PosY (1), separation (2), separationERROR (3), pa (4), paERROR (5), magnitude(6), magnitudeERROR(7), additional (8)]









    def run(self):
        """
        Run method of the module.

        :return: None
        """
        
        # Set up ports and parameters
        self.m_flux_position_port.del_all_data()
        self.m_flux_position_port.del_all_attributes()

#        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
#        self.m_aperture /= pixscale

        sys.stdout.write("Running AperturePhotometryModule_background ....")
        sys.stdout.flush()

        flux_arr = self.m_psf_in_port.get_all()
        science_arr = self.m_image_in_port.get_all()
        science_im = np.median(science_arr, axis=0)


        science_time = self.m_image_in_port.get_attribute('EXPTIME')[0]
        flux_time = self.m_psf_in_port.get_attribute('EXPTIME')[0]
        lam = self.m_psf_in_port.get_attribute('LAMBDA')[0]
        lamd = self.m_psf_in_port.get_attribute('LAMBDAD')[0]



        #set scaling factor
        if self.m_psf_scaling is None:
            self.m_psf_scaling = filter_scaling_calc(science_time=science_time,
                                                     flux_time=flux_time,
                                                     wavelength=lam,
                                                     delta_wavelength=lamd,
                                                     flux_filter=self.m_flux_filter)

        #calculate flux via aperture
        if self.m_fit_each_image:
            res = self._perform_aperture_photometry(image_arr=science_arr,
                                                    flux_arr=flux_arr)
        else:
            res = self._perform_aperture_photometry(image_arr=science_im.reshape(1,*science_im.shape),
                                                    flux_arr=flux_arr)

        self.m_flux_position_port.append(res, data_dim=2)


        # self.m_flux_position_port.add_history("AperturePhotometryModule_background",
        #                                                   "Aperture size = &.2f"%self.m_aperture)
        self.m_flux_position_port.copy_attributes(self.m_image_in_port)
        lam = self.m_image_in_port.get_attribute('LAMBDA')
        self.m_flux_position_port.del_attribute('LAMBDA')
        self.m_flux_position_port.add_attribute('LAMBDA', np.array([lam[0]]), static=False)


        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()
