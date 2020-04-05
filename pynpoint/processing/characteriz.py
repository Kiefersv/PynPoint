#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:24:19 2019

@author: kiefer
"""



import math
import os
import species

import numpy as np
import matplotlib.pyplot as plt

from typing import List
from typeguard import typechecked
from scipy import optimize

from pynpoint.core.processing import ProcessingModule

from astropy.io import ascii
from astropy import units as u
from astropy import constants as const





class SpectralCharModule(ProcessingModule):
    """
    Pipeline to do simple spectral charakterisation
    """

    __author__ = 'Sven Kiefer'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 method: str,
                 error_simplex: float = None,
                 error_cutting: float = None,
                 mol_fit: str = None,
                 star_model_par: dict = None,
                 companion_model: dict = None,
                 prop_factor: float = None,
                 manual_data_points_mag: List[List[float]] = None,
                 output_dir: str = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the science images that are read 
            as input.
        method : str
            Type of input data. Can either be 'aperture' or 'simplex'.
        error_simplex : float
            Tolarance parameter of the SimplexMinimizationModule
        error_cutting : float
            number of sigmas below the mean error value which should be used as 
            threshold. All errors belwo the threshold get replaced by the 
            threshold value. Not executed if set to None.
        mol_fit : str
            Type of model fitting. Can either be set to 'full' or 'H2O'.
        star_model_par : dict
            Dictionary with the Bt-Settl parameters of the host star. Only used
            if the host stars spectra is not given as 'working/star_model.txt'.
        companion_model: dict
            Dictionary wiht the starting parameters of the MCMC fit. The model can
            be set with the 'model' parameter.
        prop_factor : float
            Used to apply correction to the stelar if given and necessary.
        manual_data_points_mag : List[List[float]]
            List with additional data points to be added to the spectra (e.g. from
            SPHERE/IRDIS). The list should be of the form:
                [[mean wavelength 1, ... , mean wavelength n],
                 [magnitude value 1, ... , magnitude value n],
                 [wavlength range 1, ... , wavlength range n]]
        output_dir : str
            Path to the directory where the output should be stored
            
        Returns
        -------
        NoneType
            None
        """

        super(SpectralCharModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        
        self.m_method = method
        self.m_error_simplex = error_simplex
        self.m_error_cutting = error_cutting
        self.m_mol_fit = mol_fit
        self.m_star_model_par = star_model_par
        self.m_companion_model = companion_model
        self.m_prop_factor = prop_factor
        self.m_manual_data_points_mag = manual_data_points_mag
        self.m_output_dir = output_dir




    def _companion_fitting(self, database, lr):
        
        # parameters - needs to be changed to user input
        n_wlakers = 50
        mcmc_nr = 5000
        burn_in = 100
        
        
        # Add companion parameters
        if 'distance' not in self.m_companion_model.keys():
            self.m_companion_model['distance'] = None
        if 'distance_error' not in self.m_companion_model.keys():
            self.m_companion_model['distance_error'] = None
        if 'app_mag' not in self.m_companion_model.keys():
            self.m_companion_model['app_mag'] = None
        
        dist_we = (self.m_companion_model['distance'], 
                   self.m_companion_model['distance_error'])
        
        if dist_we[0]==None and dist_we[1]==None:
            dist_we = None
            
        
        # add companion to database
        database.add_object(object_name = 'planet',
                            distance = dist_we,
                            app_mag = self.m_companion_model['app_mag'],
                            spectrum = None)
        
        objectbox = database.get_object(object_name='planet',
                                        filter_id=None,
                                        inc_phot=True,
                                        inc_spec=True)
        
        
        # fit companion to model
        fit = species.FitModel(objname='planet',
                               filters=None,
                               model=self.m_companion_model['model'],
                               bounds=None,
                               inc_phot=True,
                               inc_spec=False)
        
        fit.run_mcmc(nwalkers=n_wlakers,
                     nsteps=mcmc_nr,
                     guess=self.m_companion_model,
                     tag='planet')
        
        species.plot_walkers(tag='planet',
                             output='walkers.pdf')

        species.plot_posterior(tag='planet',
                               burnin=burn_in,
                               title=r'',
                               output='posterior.pdf')


        # Generate sample modles
        samples = database.get_mcmc_spectra(tag='planet',
                                            burnin=burn_in,
                                            random=30,
                                            wavelength=(lr[0], lr[1]),
                                            specres=50.)
    

        # calculate median of samples
        median = database.get_median_sample(tag='planet', 
                                            burnin=burn_in)
        
        drift = species.ReadModel(model=self.m_companion_model['model'], 
                                  wavelength=(lr[0], lr[1]))
        
        model = drift.get_model(model_par=median, 
                                specres=50.)
           
        model = species.add_luminosity(model)
        
        
        # calculate residuals
        residuals = species.get_residuals(datatype='model',
                                          spectrum=self.m_companion_model['model'],
                                          parameters=median,
                                          filters=None,
                                          objectbox=objectbox,
                                          inc_phot=True,
                                          inc_spec=False)
        
        
        # create synphot
        synphot = species.multi_photometry(datatype='model',
                                           spectrum=self.m_companion_model['model'],
                                           filters=objectbox.filter,
                                           parameters=median)
        
        
        # plot the original and fitted spectra
        species.plot_spectrum(boxes=(samples, model, objectbox, synphot),
                              filters=None,#objectbox.filter,
                              residuals=residuals,
                              colors=('gray', 'tomato', ('green','blue'), 'black'),
                              xlim=(lr[0], lr[1]),
                              scale=('linear', 'linear'),
                              title=r'',
                              output='spectrum.pdf')





    def _error_cutter(self, errors):
        """
        Increses lowest error values according to the self.m_error_cutting value
        to reduce the influence of low error value during modle fitting. The self.
        m_error_cutting value defines the number of sigmas below the mean error 
        value which should be used as threshold. All errors belwo the threshold
        get replaced by the threshold value.
        """
        # Calculate mean and std of error values
        men_of_error = np.nanmean(errors)
        std_of_error = np.nanstd(errors)
        
        # Nan handling
        errors = np.nan_to_num(errors)
        
        # Calculate cutting map
        tsig_value = men_of_error - self.m_error_cutting * std_of_error
        tsig_map = (errors < tsig_value)
        
        # Apply cutting map
        errors[tsig_map] = tsig_value
        
        return errors




    
    def _model_based_rejection(self, save_data, delta_lams, delta_t = 500):
        """
        Get a rough temperature, LogG and FeH estimation using a model grid fitting 
        (using Bt-Settl) and calculates the best fit via correlation with the 
        observation. The fitting process gives an estimation of the distance.
        """
        
        # ------------------------------------------------------------------------- Preparation
        
        # copy and sort the input data 
        lin_fit_data = np.copy(save_data)
        sort = np.argsort(lin_fit_data[:,0])
        data_l = lin_fit_data[sort,0]
        data_d = lin_fit_data[sort,1]
        data_e = lin_fit_data[sort,2]
        
        # filter out nans
        nan_mask = np.nan_to_num(data_d) != 0
        data_l = data_l[nan_mask]
        data_d = data_d[nan_mask]
        data_e = data_e[nan_mask]
        delta_lams = delta_lams[nan_mask]
        
        # Create or load models: for a new run, old model files have to be deleted
        if os.path.isfile('working/params.npy') and os.path.isfile('working/model_bb.npy'):
            model_params = np.load('working/params.npy', allow_pickle=True)
            model_bb = np.load('working/model_bb.npy', allow_pickle=True)
        else:
            model_params, model_bb = Model_Simpler([data_l, delta_lams])
            
        temps = np.asarray(model_params[1], dtype=int)*100
        
        
        # ------------------------------------------------------------------------- coross corelation filter 
        ld = data_l
        
        # --- Use the full spectra for fitting and correlation calculation
        if self.m_mol_fit == 'full':
            # cor_mask = Wavelengths to calculate cross correlation
            cor_mask = (ld == ld)
            # bbfit_mask = Wavelengths to fit
            bbfit_mask = cor_mask
        
        # --- Use water features to calculate cross cor and no for model fitting
        elif self.m_mol_fit == 'H2O':
            cor_mask = (ld > 1.3) * (ld < 1.6)
            bbfit_mask = (data_l > 1.1) * (data_l < 1.25)
        
        else:
            raise ValueError(f'Unsupported molecular fitting key word {self.m_mol_fit}')



        # ------------------------------------------------------------------------- Fitted Model Correlation
        coeff_matrix = np.zeros_like(model_bb[:,:,:,0])
        
        for t, TEMP in enumerate(model_bb):
            for l, LOGG in enumerate(TEMP):
                for f, MODEL in enumerate(LOGG):
                    
                    def bb_d(mask, r):
                        # Current model with adjustable distance. See bestFit below
                        # for a more detailed describtion (same function).
                        maskk = (mask==1)
                        lin_approx = (temps[t] * 202000 - 3.85e+8) * u.m
#                        lin_approx = (temps[t] / 6000 * 696340000) * u.m
                        b =  (lin_approx/r/u.pc)**2 * model_bb[t,l,f][maskk] * u.erg/u.cm**2/u.s/u.AA
                        return np.log10(b.to(u.W / u.m**2 / u.micron)/(u.W / u.m**2 / u.micron))
                    
                    # Fit current model to data
                    dist, c_dist = optimize.curve_fit(bb_d, 
                                                      bbfit_mask, 
                                                      np.log10(data_d[bbfit_mask]), 
                                                      p0=[self.m_companion_model['distance']])
                    
                    # go back to lin space for the fitted model
                    mod_fitted = 10**bb_d((data_l==data_l), dist[0])
                    
                    # calculate corss correlation of data and distance fitted model
                    coeff_matrix[t,l,f] = np.corrcoef(mod_fitted[cor_mask],
                                                      data_d[cor_mask])[0,1]
        
        
        
        # --------------------------------------------------------------------- Best Fit
        # calculate best fit
        result = np.where(coeff_matrix == np.max(coeff_matrix))
        listOfCordinates = list(zip(result[0], result[1], result[2]))
        bf_coords = np.asarray(listOfCordinates)[0]
        
        # Function to fit the distance of the best fitting modle
        def bestFit(lam, r):
            """ 
            Blackbody as a function of wavelength (um) and distance (pc).
            Returns units of erg/s/cm^2/cm/Steradian.
            """
            # assigne unites
            r = r * u.pc
            temp = temps[bf_coords[0]] * u.K

            # Calculating object radius as described in chapter 5
            lin_approx = (temp * 203000 / u.K - 3.85e+8) * u.m
            
            # Fit best fit model
            b =  (lin_approx/r)**2 * model_bb[bf_coords[0],bf_coords[1],bf_coords[2]] * u.erg/u.cm**2/u.s/u.AA
            
            # return log of result for better fitting
            return np.log10(b.to(u.W / u.m**2 / u.micron)/(u.W / u.m**2 / u.micron))
        

        # Fit best model to spectra observed to determine distance
        dist, c_dist = optimize.curve_fit(bestFit, 
                                          data_l, 
                                          np.log10(data_d),
                                          p0=[self.m_companion_model['distance']])
        
        
        
        # --------------------------------------------------------------------- Temperature Projection
        # Sum over FeH and LogG, calculate Maximum Likelyhood
        red_c = np.sum(np.sum(coeff_matrix, axis=2), axis=1)
        mle_temp = temps[np.argmax(red_c)]
        redn_c = (red_c - np.min(red_c)) / (np.max(red_c)- np.min(red_c))
        
        # Normalisation of coeff_matrix
        coeff_matrix_norm = (coeff_matrix - np.min(coeff_matrix)) / (np.max(coeff_matrix)- np.min(coeff_matrix))
        
        # distance error according to chapter 5.2
        d_sigma = 1.25*np.abs(dist[0]) / (2.9*10**(-3)*temps[bf_coords[0]] - 5.5)



        # --------------------------------------------------------------------- Plotting and Printing
        
        # Print the parameter space
        ps = 5
        fig, axs = plt.subplots(1, 4)#, figsize=(5,4))
        fig.text(0.5, 0.1, 'LogG', ha='center', fontsize=13)
        fig.text(0.13, 0.55, 'Temp [1000K]', va='center', rotation='vertical', fontsize=13)
        plt.setp(axs, 
                 xticks=range(len(model_params[2]))[::4], 
                 xticklabels=model_params[2][::4], 
                 yticks=range(len(temps))[::ps],
                 yticklabels=temps[:][::ps]//1000)

        i = 0
        for ax, ffeehh in zip(axs, model_params[3]):
            wut = ax.imshow(coeff_matrix_norm[:,:,i],
                            vmin=np.min(coeff_matrix_norm), 
                            vmax=np.max(coeff_matrix_norm))
            ax.set_title('FeH: -' + ffeehh)
            if i != 0: ax.set_yticks([])
            i += 1
        cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.7])
        plt.subplots_adjust(right=0.8, left=0.2, bottom=0.2, top=0.9)
        cbar = fig.colorbar(wut, cax=cbar_ax)
        cbar.set_label('Normalised correlation value', rotation=270, labelpad=12)
        plt.subplots_adjust(right=0.8, left=0.2, bottom=0.2, top=0.9)
        plt.savefig(str(self.m_output_dir) + '/Parmeterspace_3D_Check.png')
        
        
        # Print the parameterspace of temperature
        plt.figure()
        plt.plot(temps, redn_c)
        plt.ylabel('Normalised correlation value', fontsize = 14)
        plt.xlabel('Temperature [K]', fontsize = 14)
        plt.xticks(temps[::5], temps[::5])
        plt.savefig(str(self.m_output_dir) + '/Parmeterspace_1D_Check.png')
        
        
        # Print the Black body fit for distance
        plt.figure()
        plt.errorbar(data_l, data_d, yerr=data_e, fmt='.', label='data')
        plt.plot(data_l, 10**bestFit(data_l, dist[0]), 'orange', label='best fit model')
        plt.xlabel(r'Wavelength [$\mu$m]', fontsize = 14)
        plt.ylabel(r'Flux [W/(m$^2 \mu$m)]', fontsize = 14)
        plt.legend()
        plt.savefig(str(self.m_output_dir) + '/bbody-fit.png')
        
        # interessting Values
        print ('==========================================================')
        print(r'Dist: ' +  str(int(round(np.abs(dist[0])//10)*10)) +
              r' +/- ' + str(d_sigma))
        print(r'MLE-Temp: ' +  str(mle_temp))
        print('Best Fit params: T = ', str(temps[bf_coords[0]]), 
              ' // LogG = ', model_params[2][bf_coords[1]], 
              ' // FeH = ', model_params[3][bf_coords[2]])
        print ('==========================================================')
        
        
        # add newly fitted information to companion parameters
        self.m_companion_model['distance'] = int(round(np.abs(dist[0])//10)*10)
        self.m_companion_model['distance_error'] = 50
        self.m_companion_model['teff'] = temps[bf_coords[0]]
        self.m_companion_model['logg'] = float(model_params[2][bf_coords[1]])
        self.m_companion_model['feh'] = float(model_params[3][bf_coords[2]])
        
        
        
        
        
        
        

    @typechecked
    def run(self) -> None:
        """
        Methode to reduce the image_in_tag with the spectral differentiated images
        created by rescaling the images according to the wavelength.
        
        Returns
        -------
        NoneType
            None
            
        [REMARK]: This module is purly educational and should not be used in a real
            analysis of data
        """
        # Get working dir
        working_place = self._m_config_port.get_attribute('WORKING_PLACE')
         
        #collect information and data
        data_read = self.m_image_in_port.get_all()  
        lam = self.m_image_in_port.get_attribute('LAMBDA')
        lam_delta = self.m_image_in_port.get_attribute('LAMBDAD')
        lam_delta = np.ones_like(lam) * lam_delta[0] # Current correction for imporper value assignment
        tempa = self.m_star_model_par['teff']
        
        # set read in parameters according to method
        if self.m_method == 'aperture':
            data_in = data_read[:,6]
            data_error = data_read[:,7]
        
        elif self.m_method == 'simplex':
            data_nr = len(data_read)//(len(list(set(lam[lam != 0]))))
            data_in = data_read[::data_nr][:,4]
            lam = lam[::data_nr]
            
            if self.m_error_simplex is None:
                data_error = None
            else:
                data_error = np.ones_like(data_in) * self.m_error_simplex
                
        else:
            raise ValueError('Invalid method selected')
        
        
        # add additional data points
        calc_aid = (lam==lam)
        if self.m_manual_data_points_mag is not None:
            add_len = len(self.m_manual_data_points_mag[1])
            # append values
            lam = np.append(lam, self.m_manual_data_points_mag[0])
            data_in = np.append(data_in, self.m_manual_data_points_mag[1])
            lam_delta = np.append(lam_delta, self.m_manual_data_points_mag[2])
            if data_error is not None:
                data_error = np.append(data_error, [np.nan for i in range(add_len)])
            
            # create calculation aid
            calc_aid = np.append(calc_aid, [False for i in range(add_len)])
            
        
        # Sort data according to Lambda
        sorted_lam = np.argsort(lam)
        lam_set = lam[sorted_lam]
        data_in = data_in[sorted_lam]
        data_error = data_error[sorted_lam]
        lamd = lam_delta[sorted_lam]
        calc_aid = calc_aid[sorted_lam]
        
        # --- Calculating the reference spectra of the Star
        species.SpeciesInit('./working')
        
        #get Vega flux
        database = species.Database()
        try:
            database.add_spectrum('vega')
        except:
            pass
        
        vega = species.ReadCalibration('vega').get_spectrum()
        
        
        # --- get Star model flux
        lr = (lam_set[0] - np.max(lamd), lam_set[-1] + np.max(lamd))
        
        # load model from txt file
        if os.path.isfile('working/star_model.txt'):
            model_full = ascii.read('working/star_model.txt')
            star_lam = model_full['BT-Settl'] * 1e-4
            star_flux = model_full["(AGSS2009)"] * 10
            star_flux_set = np.zeros_like(lam_set)
            
            if self.m_prop_factor is not None:
                star_flux *= self.m_prop_factor
                
        
        # create model using input parameters
        else:
            pass
            model = species.ReadModel(model='bt-nextgen',
                                      wavelength=(lr[0], lr[1]))
                                      #teff = (tempa-100, tempa+100))
            
            modelbox = model.get_model(model_par=self.m_star_model_par,
                                       specres=200.)
            
            star_lam = modelbox.wavelength
            star_flux = modelbox.flux
            star_flux_set = np.zeros_like(lam_set)
            
        
        #read out spectras of model and vega
        vega_lam = vega.wavelength
        vega_flux = vega.flux
        vega_flux_set = np.zeros_like(lam_set)
        
        planet_Vmag_set = np.zeros_like(lam_set)
        planet_flux_set = np.zeros_like(lam_set)
        planet_flux_errors_set = np.zeros_like(lam_set)
        
        save_data = np.zeros((len(lam_set), 3))
        
        
        # Applying correction if given and necessary
        
        
        # Calculating planet flux and magnitude for each wavelength
        for l, lums in enumerate(lam_set):
            # create star wavelength mask
            star_mask_l = (lums + lamd[l]/2 > star_lam) * (lums - lamd[l]/2 < star_lam)
            star_flux_set[l] = np.mean(star_flux[star_mask_l])
            
            # create vega wavelength mask
            vega_mask_l = (lums + lamd[l]/2 > vega_lam) * (lums - lamd[l]/2 < vega_lam)
            vega_flux_set[l] = np.mean(vega_flux[vega_mask_l])
            
            # calculate vega mag and flux of the companion
            if calc_aid[l]:
                planet_Vmag_set[l] =  -2.5*math.log10(star_flux_set[l]/vega_flux_set[l]) + data_in[l]
            else:
                planet_Vmag_set[l] = data_in[l]
                
            planet_flux_set[l] = 10**(-planet_Vmag_set[l]/2.5) * vega_flux_set[l]
            
            
            # Gaussian error propagation if possible
            if data_error is not None:
                planet_flux_errors_set[l] = (planet_flux_set[l] * np.log(10)  / 2.5) * data_error[l]
            else:
                planet_flux_errors_set[l] = np.NaN
            
            # save the calculated values into an output array
            save_data[l,0] = lums
            save_data[l,1] = planet_flux_set[l]
            save_data[l,2] = planet_flux_errors_set[l]



        # --------------------------------------------------------------------- error cutting
        if self.m_error_cutting is not None:
            save_data[:,2] = self._error_cutter(save_data[:,2])
            planet_flux_errors_set = self._error_cutter(planet_flux_errors_set)



        # --------------------------------------------------------------------- model based rejection
        if self.m_mol_fit is not None:
            self._model_based_rejection(save_data, lamd)



#        # --------------------------------------------------------------------- MCMC Model fit
#        if self.m_companion_model is not None:
#            self.m_companion_file = os.path.join(working_place, 'tmp_companion.txt')
#            np.savetxt(self.m_companion_file, save_data)
#        
#            self._companion_fitting(database, lr)



        # --------------------------------------------------------------------- plot magnitudes
        plt.figure()
        plt.gca().invert_yaxis()
        
        if data_error is not None:
            plt.errorbar(lam_set, planet_Vmag_set, yerr=data_error, fmt='.b', capsize=5, elinewidth=1)
        else:
            plt.plot(lam_set, planet_Vmag_set, '.b')
            
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel(r'Birghtness [mag$_{vega}]$')
        out_name = os.path.join(self.m_output_dir, 'Flux.png')
        plt.savefig(out_name)
        
        

        # --------------------------------------------------------------------- plot flux
        plt.figure()
        
        if data_error is not None:
            plt.errorbar(lam_set, planet_flux_set, yerr=planet_flux_errors_set, fmt='.b', capsize=5, elinewidth=1)
        else:
            plt.plot(lam_set, planet_flux_set, '.b')
            
        plt.xlabel(r'Wavelength [$\mu$m]')
        plt.ylabel(r'Flux [W/m$^2 \mu$m)]')
        out_name = os.path.join(self.m_output_dir, 'VegaMag.png')
        plt.savefig(out_name)
      
        
        
        
        
        
def Model_Simpler(lam: List[List[float]]):
    """
    Creates model gird with the same input as the data. Model: Bt-Settl
    
    Parameters
    ----------
    lam : List[List[float]]
        2D Wavelength information in microns with:
            lam[0]: wavelength of the data point
            lam[1]: width of filter
    
    """
    
    # Function to find all Bt-Settl modles by name
    def path_model(temp, logG, FeH, alp): 
        if float(temp) > 25.5: 
            string =  f"model/models_1582013898/bt-settl-agss/lte0{temp}-{logG}-{FeH}{alp}.BT-Settl.7.dat.txt"
        else:
            string = f"model/models_1582013898/bt-settl-agss/lte0{temp}-{logG}-{FeH}.BT-Settl.7.dat.txt"
        
        return string
    
    
    # ============================================================================= Preparation
    # Convert input wavelengths to Angstrom
    for l in range(len(lam)):
        lam[l] = lam[l] * 1e+4
    
    # Define Parameter space
    temp = ['20', '22', '24', '26', '28', '30', '32', '34', '36', '38',
            '40', '42', '44', '46', '48', '50', '52', '54', '56', '58',
            '60', '62', '64', '66', '68', '70', '72', '74', '76', '78', 
            '80', '82', '84', '86', '88', '90', '92', '94', '96', '98']
    logG = ['2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0']
    FeH = ['0.0', '0.5', '1.5', '2.0']
    alp = ['a+0.0', 'a+0.2', 'a+0.4', 'a+0.4']
    
    # set up output arrays
    mod_for_bb = np.zeros((len(temp), len(logG), len(FeH), len(lam[0])))
    final_params = []
    
    # ready params for saving
    final_params.append(lam[0])
    final_params.append(temp)
    final_params.append(logG)
    final_params.append(FeH)
    
    
    # ============================================================================= Model Reduction
    for t, TEMP in enumerate(temp):
        for l, LOGG in enumerate(logG):
            for f, FEH in enumerate(FeH):
                model_full = ascii.read(path_model(TEMP, LOGG, FEH, alp[f]))
                
                # reduce resolution of modle to the one of the measured data
                for i, _ in enumerate(lam[0]):
                    lam_mask = ((model_full['BT-Settl'] > lam[0][i] - lam[1][i]/2) * 
                                (model_full['BT-Settl'] < lam[0][i] + lam[1][i]/2))
                    mod_for_bb[t,l,f,i] = np.mean(model_full['(AGSS2009)'][lam_mask])


    # ============================================================================= Save Data
    np.save('working/params', final_params)
    np.save('working/model_bb', mod_for_bb)
    
    
    return final_params, mod_for_bb
    
    
    