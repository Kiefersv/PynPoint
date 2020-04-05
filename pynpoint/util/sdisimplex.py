#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 08:05:24 2019

@author: kiefer
"""


import sys

import numpy as np

from scipy.optimize import minimize

from pynpoint.util.analysis import fake_planet, merit_function
from pynpoint.util.image import create_mask, cartesian_to_polar, \
                                rotate_coordinates
from pynpoint.util.residuals import combine_residuals
from pynpoint.util.sdi import postprocessor





def simplex_minimizer (args, method, options, mask_l, tmp_im_str, tmp_psf_str, 
                       parang, extra_rot, scaling, cent_size, edge_size, 
                       count, n_components, sklearn_pca, im_shape, residuals,
                       merit, aperture, sigma, pixscale, center,
                       there_is_ref, processing_type):
    
        """
        Function to support mulitprocessing minimization. Minimizes _simplex_objective
        and returns all generated minization steps.
        """
    
        
        # resulting residuals for each minimization step [min. steps, pic_x, pic_y] 
        result_res = [] 
        # calculated flux for each minimization step [min. steps, 6]
        result_flux = []
        
        # execute minimization
        minimize(fun=_simplex_objective,
                 x0=args,
                 args=(mask_l, tmp_im_str, tmp_psf_str, parang, 
                       extra_rot, scaling, 
                       cent_size, edge_size, count, 
                       n_components, sklearn_pca, im_shape, residuals,
                       merit, aperture, sigma, pixscale,
                       center, there_is_ref, processing_type,
                       result_res, result_flux),
                 method=method,
                 tol=None,
                 options=options)    
        
        return [result_res, result_flux]




def _simplex_objective(arg, mask, path_images, path_psf, parang, extra_rot,
                      scales, cent_size, edge_size, count, n_components, 
                      sklearn_pca,im_shape, residuals, merit, aperture, sigma,
                      pixscale, center, there_is_ref, processing_type,
                      result_res, result_flux):
    
    """
    Simplex fitting function: images with injected planets get subtracted from
    original image to remove the detected planet. Minimization variable is
    defined by the parameter method.
    """
    
    #assign parameters
    pos_y = arg[0]
    pos_x = arg[1]
    mag = arg[2]
    sep_ang = cartesian_to_polar(center, pos_y, pos_x)
    
    #load images
    images = np.load(path_images)
    psf = np.load(path_psf)

    
    #inject fake planet into images
    fake = fake_planet(images=images,
                       psf=psf,
                       parang=parang,
                       position=(sep_ang[0], sep_ang[1]),
                       magnitude=mag,
                       psf_scaling=1.)

    maskk = create_mask(fake.shape[-2:], (cent_size, edge_size))
    

    # apply postprocessing
    if not there_is_ref:
        im_res_rot, im_res_derot = postprocessor(images=fake*maskk,
                                                 parang=-1.*parang+extra_rot,
                                                 scales= scales,
                                                 pca_number=n_components,
                                                 pca_sklearn=sklearn_pca,
                                                 im_shape=None,
                                                 indices=None,
                                                 processing_type=processing_type)

    else:
        im_reshape = np.reshape(fake*maskk, (im_shape[0], im_shape[1]*im_shape[2]))

        im_res_rot, im_res_derot = postprocessor(images=im_reshape,
                                                 parang=-1.*parang+extra_rot,
                                                 scales= scales,
                                                 pca_number=n_components,
                                                 pca_sklearn=sklearn_pca,
                                                 im_shape=im_shape,
                                                 indices=None,
                                                 processing_type=processing_type)
        
    # recombine residuals and prepare output
    res_stack = combine_residuals(method=residuals,
                                  res_rot=im_res_derot[mask],
                                  residuals=im_res_rot[mask],
                                  angles=parang[mask])

    result_res.append(res_stack[0])
    
    # calculate miminization parameter (not necessarily a chi squared value)
    chi_square = merit_function(residuals=res_stack[0, ],
                                merit=merit,
                                aperture=aperture,
                                sigma=sigma)
    
    # prepare output
    position = rotate_coordinates(center, (pos_y, pos_x), extra_rot)

    res = np.asarray([position[1],
                      position[0],
                      sep_ang[0]*pixscale,
                      (sep_ang[1]-extra_rot) % 360.,
                      mag,
                      chi_square])

    result_flux.append(res)            
    
    sys.stdout.write('\rRunning SimplexMinimizationModule... ')
    sys.stdout.write(f'{n_components} PC - chi^2 = {chi_square:.8E}\n')
    sys.stdout.flush()


    return chi_square








