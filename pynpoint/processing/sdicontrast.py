#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 08:22:50 2019

@author: kiefer
"""


import sys
import os
import time

import numpy as np
import multiprocessing as mp

from sklearn.decomposition import PCA
from typing import Union, Tuple, List
from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import center_subpixel, rotate_coordinates
from pynpoint.util.module import progress
from pynpoint.util.ifs import i_want_to_seperate_wavelengths, scaling_calculation
from pynpoint.util.sdi import filter_scaling_calc
from pynpoint.util.sdisimplex import simplex_minimizer
                                

        
        
        
        
        
class SdiSimplexMinimizationModule(ProcessingModule):
    """
    Pipeline module to measure the flux and position of a planet by injecting negative fake planets
    and minimizing a figure of merit.
    """

    __author__ = 'Tomas Stolker'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 psf_in_tag: str,
                 res_out_tag: str,
                 flux_position_tag: str,
                 position: Tuple[int, int],
                 magnitude: float,
                 psf_scaling: float = None,
                 flux_filter: str = 'ND_0.0',
                 merit: str = 'hessian',
                 aperture: float = 0.1,
                 sigma: float = 0.0,
                 tolerance: float = 0.1,
                 pca_number: Union[int, range, List[int]] = 10,
                 cent_size: float = None,
                 edge_size: float = None,
                 extra_rot: float = 0.,
                 residuals: str = 'median',
                 reference_in_tag: str = None,
                 processing_type: str = 'Tadi') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the science images that are read as input.
        psf_in_tag : str
            Tag of the database entry with the reference PSF that is used as fake planet. Can be
            either a single image or a stack of images equal in size to ``image_in_tag``.
        res_out_tag : str
            Tag of the database entry with the image residuals that are written as output. The
            residuals are stored for each step of the minimization. The last image contains the
            best-fit residuals.
        flux_position_tag : str
            Tag of the database entry with the flux and position results that are written as output.
            Each step of the minimization stores the x position (pix), y position (pix), separation
            (arcsec), angle (deg), contrast (mag), and the chi-square value. The last row contains
            the best-fit results.
        position : tuple(int, int)
            Approximate position (x, y) of the planet (pix). This is also the location where the
            figure of merit is calculated within an aperture of radius ``aperture``.
        magnitude : float
            Approximate magnitude of the planet relative to the star.
        psf_scaling : float
            Additional scaling factor of the planet flux (e.g., to correct for a neutral density
            filter). Should be negative in order to inject negative fake planets.
        merit : str
            Figure of merit for the minimization. Can be 'hessian', to minimize the sum of the
            absolute values of the determinant of the Hessian matrix, or 'poisson', to minimize the
            sum of the absolute pixel values, assuming a Poisson distribution for the noise
            (Wertz et al. 2017), or 'gaussian', to minimize the ratio of the squared pixel values
            and the variance of the pixels within an annulus but excluding the aperture area.
        aperture : float
            Aperture radius (arcsec) at the position specified at *position*.
        sigma : float
            Standard deviation (arcsec) of the Gaussian kernel which is used to smooth the images
            before the figure of merit is calculated (in order to reduce small pixel-to-pixel
            variations).
        tolerance : float
            Absolute error on the input parameters, position (pix) and contrast (mag), that is used
            as acceptance level for convergence. Note that only a single value can be specified
            which is used for both the position and flux so tolerance=0.1 will give a precision of
            0.1 mag and 0.1 pix. The tolerance on the output (i.e., the chi-square value) is set to
            np.inf so the condition is always met.
        pca_number : int, range, list(int, )
            Number of principal components (PCs) used for the PSF subtraction. Can be either a
            single value or a range/list of values. In the latter case, the `res_out_tag` and
            `flux_position_tag` contain a 3 digit number with the number of PCs.
        cent_size : float
            Radius of the central mask (arcsec). No mask is used when set to None. Masking is done
            after the artificial planet is injected.
        edge_size : float
            Outer radius (arcsec) beyond which pixels are masked. No outer mask is used when set to
            None. The radius will be set to half the image size if the argument is larger than half
            the image size. Masking is done after the artificial planet is injected.
        extra_rot : float
            Additional rotation angle of the images in clockwise direction (deg).
        residuals : str
            Method for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
        reference_in_tag : str, None
            Tag of the database entry with the reference images that are read as input. The data of
            the ``image_in_tag`` itself is used as reference data for the PSF subtraction if set to
            None. Note that the mean is not subtracted from the data of ``image_in_tag`` and
            ``reference_in_tag`` in case the ``reference_in_tag`` is used, to allow for flux and
            position measurements in the context of RDI.

        Returns
        -------
        NoneType
            None
        """

        super(SdiSimplexMinimizationModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        if reference_in_tag is None:
            self.m_reference_in_port = None
            self.m_there_is_ref = False
        else:
            self.m_reference_in_port = self.add_input_port(reference_in_tag)
            self.m_there_is_ref = True

        self.m_res_out_port = self.add_output_port(res_out_tag)
        self.m_flux_pos_port = self.add_output_port(flux_position_tag)


        self.m_position = position
        self.m_magnitude = magnitude
        self.m_psf_scaling = psf_scaling
        self.m_flux_filter = flux_filter
        self.m_merit = merit
        self.m_aperture = aperture
        self.m_sigma = sigma
        self.m_tolerance = tolerance
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size
        self.m_extra_rot = extra_rot
        self.m_residuals = residuals
        self.m_processing_type = processing_type

        if isinstance(pca_number, int):
            self.m_pca_number = [pca_number]
        else:
            self.m_pca_number = pca_number

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. The position and contrast of a planet is measured by injecting
        negative copies of the PSF template and applying a simplex method (Nelder-Mead) for
        minimization of a figure of merit at the planet location.

        Returns
        -------
        NoneType
            None
        """
        
        # preparation of data
        self.m_res_out_port.del_all_data()
        self.m_res_out_port.del_all_attributes()

        self.m_flux_pos_port.del_all_data()
        self.m_flux_pos_port.del_all_attributes()
            
        # read in pipeline variables
        cpu = self._m_config_port.get_attribute('CPU')
        working_place = self._m_config_port.get_attribute('WORKING_PLACE')
        
        # read in image variables
        parang = self.m_image_in_port.get_attribute('PARANG')
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        lam = self.m_image_in_port.get_attribute('LAMBDA')
        science_time = self.m_image_in_port.get_attribute('EXPTIME')
        lamd = self.m_image_in_port.get_attribute('LAMBDAD')
        
        # read in psf varaibales
        flux_time = self.m_psf_in_port.get_attribute('EXPTIME')
        lam_fl = self.m_psf_in_port.get_attribute('LAMBDA')
        
        # pre calculation and variable setting
        aperture = (self.m_position[1], self.m_position[0], self.m_aperture/pixscale)
        scaling = scaling_calculation(pixscale, lam)
        
        self.m_sigma /= pixscale

        if self.m_cent_size is not None:
            self.m_cent_size /= pixscale

        if self.m_edge_size is not None:
            self.m_edge_size /= pixscale

        psf_pre = self.m_psf_in_port.get_all()
        images = self.m_image_in_port.get_all()
        
        
        # zero padding of psf
        psf_nbr_size = psf_pre.shape[0]
        img_pic_size = images.shape[1]
        psf_pic_size = psf_pre.shape[1]
        
        psf_padded = np.zeros((psf_nbr_size, img_pic_size, img_pic_size))
        
        for i, psf_i in enumerate(psf_pre):
            
            psf_padded[i] = np.median(psf_i)
            
            f1 = (img_pic_size - psf_pic_size)//2
            f2 = (img_pic_size + psf_pic_size)//2
            
            psf_padded[i, f1:f2, f1:f2] = psf_i 
        
        psf_pre = psf_padded
        
        
        # Setting psf according to images and wavelength information
        if self.m_psf_scaling is None:
            self.m_psf_scaling = np.ones_like(images[:,0,0])
            psf = np.ones_like(images)
            for l, _ in enumerate(images):
                mask_l = (lam[l] == lam_fl)
                
                self.m_psf_scaling[l] = filter_scaling_calc(science_time=science_time[0],
                                                            flux_time=flux_time[0],
                                                            wavelength=lam[l],
                                                            delta_wavelength=lamd[0],
                                                            flux_filter=self.m_flux_filter).nominal_value
                                  
                # directly applying psf correction
                psf[l] = -1*np.median(psf_pre[mask_l], axis=0) * self.m_psf_scaling[l]
                                  
        else:
            psf = -1 * psf_pre * self.m_psf_scaling
        

        if psf.shape[0] != 1 and psf.shape[0] != images.shape[0]:
            raise ValueError('The number of frames in psf_in_tag does not match with the number '
                             'of frames in image_in_tag. The DerotateAndStackModule can be '
                             'used to average the PSF frames (without derotating) before applying '
                             'the SimplexMinimizationModule.')

        center = center_subpixel(psf)

        if self.m_reference_in_port is not None and self.m_merit != 'poisson':
            raise NotImplementedError('The reference_in_tag can only be used in combination with '
                                      'the \'poisson\' figure of merit.')

        
        
        

        pos_init = rotate_coordinates(center,
                                      (self.m_position[1], self.m_position[0]),  # (y, x)
                                      self.m_extra_rot)


        #Set up of pool
        lam_set = np.sort(list(set(lam)))
        if not i_want_to_seperate_wavelengths(self.m_processing_type):
                lam_set = [0]
        
        
        
        # Create temporary files
        tmp_im_str = os.path.join(working_place, 'tmp_simplex_images.npy')
        tmp_psf_str = os.path.join(working_place, 'tmp_simplex_psf.npy')

        np.save(tmp_im_str, images)
        np.save(tmp_psf_str, psf)
        
        # pool parameters
        result_res = []
        result_flux = []
        async_results = []
        count = 0
        
        pool = mp.Pool(cpu)
        
        #looping over all multiporcessable functions
        for lum_i in lam_set:
            if i_want_to_seperate_wavelengths(self.m_processing_type):
                mask_l = (lum_i == lam)
            else:
                mask_l = (np.ones_like(lam) == 1) 
            
            for i, n_components in enumerate(self.m_pca_number):
    
#                if self.m_reference_in_port is None:
                sklearn_pca = None
                im_shape = None
    
#                else:
#                    ref_data = self.m_reference_in_port.get_all()
#    
#                    im_shape = images[mask_l].shape
#                    ref_shape = ref_data[mask_l].shape
#    
#                    if ref_shape[1:] != im_shape[1:]:
#                        raise ValueError('The image size of the science data and the reference data '
#                                         'should be identical.')
#    
#                    # reshape reference data and select the unmasked pixels
#                    ref_reshape = ref_data.reshape(ref_shape[0], ref_shape[1]*ref_shape[2])
#    
#                    mean_ref = np.mean(ref_reshape, axis=0)
#                    ref_reshape -= mean_ref
#    
#                    # create the PCA basis
#                    sklearn_pca = PCA(n_components=n_components, svd_solver='arpack')
#                    sklearn_pca.fit(ref_reshape)
#    
#                    # add mean of reference array as 1st PC and orthogonalize it to the PCA basis
#                    mean_ref_reshape = mean_ref.reshape((1, mean_ref.shape[0]))
#    
#                    q_ortho, _ = np.linalg.qr(np.vstack((mean_ref_reshape,
#                                                         sklearn_pca.components_[:-1, ])).T)
#    
#                    sklearn_pca.components_ = q_ortho.T
                
#                print(sum(mask_l))
#                print('HEY HO MO: ', count)
#                #actually fill the pool with the funciton ready to go
#                simplex_minimizer([pos_init[0], pos_init[1], self.m_magnitude], 'Nelder-Mead', 
#                                                             {'xatol': self.m_tolerance, 'fatol': float('inf')},
#                                                             mask_l, tmp_im_str, tmp_psf_str, parang, 
#                                                             self.m_extra_rot, scaling, 
#                                                             self.m_cent_size, self.m_edge_size, count, 
#                                                             n_components, sklearn_pca, im_shape, self.m_residuals,
#                                                             self.m_merit, aperture, self.m_sigma, pixscale,
#                                                             center, self.m_there_is_ref, self.m_processing_type)
#                
#                assert False
                async_results.append(pool.apply_async(simplex_minimizer,
                                                      args=([pos_init[0], pos_init[1], self.m_magnitude], 'Nelder-Mead', 
                                                             {'xatol': self.m_tolerance, 'fatol': float('inf')},
                                                             mask_l, tmp_im_str, tmp_psf_str, parang, 
                                                             self.m_extra_rot, scaling, 
                                                             self.m_cent_size, self.m_edge_size, count, 
                                                             n_components, sklearn_pca, im_shape, self.m_residuals,
                                                             self.m_merit, aperture, self.m_sigma, pixscale,
                                                             center, self.m_there_is_ref, self.m_processing_type)))
                                                            
            count += 1
                
        pool.close()
                
        start_time = time.time()
        
        # wait for all processes to finish
        while mp.active_children():
            # number of finished processes
            nfinished = sum([i.ready() for i in async_results])

            progress(nfinished, len(async_results), '\rRunning SimplexMinimizationModule...', start_time)

            # check if new processes have finished every 5 seconds
            time.sleep(5)


        pool.terminate()
        
        
         # get the results for every async_result object
        for item in async_results:
            result_res.append(item.get()[0])
            result_flux.append(item.get()[1])

        
        
        # --- convert irregular list to array
        #get lengths
        max_len = len(max(result_res, key = lambda x: len(x)))
        res_len = len(result_res)
        print('Should be 39: ',res_len)
        im_len = len(result_res[0][0])
        
        #assigne array
        array_res = np.zeros((res_len*max_len,im_len,im_len))
        array_flux = np.zeros((res_len*max_len,6))
        attribute_lambda = np.zeros((res_len*max_len))
        
        #set values in the correct order
        for i,j in enumerate(result_res):
            array_res[i*max_len : i*max_len + len(j)] = result_res[i][::-1]
            array_flux[i*max_len : i*max_len + len(j)] = result_flux[i][::-1]
            attribute_lambda[i*max_len : i*max_len + len(j)] = lam_set[i]
        
        
        #apply data
        self.m_res_out_port.set_all(array_res)
        self.m_flux_pos_port.set_all(array_flux)

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()

        history = f'merit = {self.m_merit}'
        

        self.m_flux_pos_port.copy_attributes(self.m_image_in_port)
        self.m_flux_pos_port.del_attribute('LAMBDA')
        self.m_flux_pos_port.add_attribute('LAMBDA', attribute_lambda, static=False)   
        self.m_flux_pos_port.add_history('SimplexMinimizationModule', history)

        self.m_res_out_port.copy_attributes(self.m_image_in_port)
        self.m_res_out_port.del_attribute('LAMBDA')
        self.m_res_out_port.add_attribute('LAMBDA', attribute_lambda, static=False)   
        self.m_res_out_port.add_history('SimplexMinimizationModule', history)

        self.m_res_out_port.close_port()
        self.m_flux_pos_port.close_port()





