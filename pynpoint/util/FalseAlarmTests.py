# -*- coding: utf-8 -*-
__author__ = "Alexander Bohn"

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import scipy.optimize as opt
from astropy.table import Table
from photutils.aperture import CircularAperture
import photutils as pu
from scipy.special import erf, erfinv
from astropy.modeling import fitting, models
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clip
# from PynPoint.core import ProcessingModule

# TODO: improve plot, statistics,...

def get_false_alarm_probability(image_in,
                                rough_pos,
                                aperture_radius,
                                method='exact',
                                skip_closest_apertures=False,
                                plot=False):

    # Get image shape
    im_shape = image_in.shape

    # Perform Gaussian fit, if required
    if method=="fit":

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
    pos_angle = np.arctan2(offset_vector[0],offset_vector[1])

#    print(separation, aperture_radius)

    # Calculate number of apertures
    number_of_apertures = int(np.floor(np.pi/np.arcsin(aperture_radius/separation)))

    # Azimuthal angles of the apertures
    azimuthal_angles = np.linspace(0,2.*np.pi,number_of_apertures+1)[:-1]

    # Rotation Matrix
    rot_matrx = lambda phi: np.array([[np.cos(phi),np.sin(phi)],
                                      [-np.sin(phi),np.cos(phi)]])

    # Get cartesian positions of the apertures
    aperture_pos = np.array([np.asarray(im_shape)/2.+np.dot(rot_matrx(i),offset_vector) for i in azimuthal_angles]).T

    # Define science and background apertures
    science_aperture = CircularAperture(positions=aperture_pos[:,0],
                                        r=aperture_radius)

    if skip_closest_apertures:
        bg_apertures = CircularAperture(positions=aperture_pos[:,2:-1],
                                        r=aperture_radius)
    else:
        bg_apertures = CircularAperture(positions=aperture_pos[:,1:],
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
    bg_fluxes_clipped = sigma_clip(bg_fluxes,sigma=3,maxiters=5)

    # Perform statistics
    bg_rms = np.sqrt(np.mean(bg_fluxes_clipped**2))
    bg_mean = np.average(bg_fluxes_clipped)
    bg_std = np.std(bg_fluxes_clipped)

    signal_to_noise = science_flux/bg_rms
    # Gaussain error propagation
    snr_sigma = signal_to_noise * np.sqrt((bg_std/science_flux)**2 +(bg_std/bg_rms)**2)

    sigma = np.abs((bg_mean-science_flux)/bg_std)

    signal = science_flux-bg_mean

    fap = 1. - erf(sigma/np.sqrt(2))

    if plot:
        f, ax = plt.subplots()
        plt.imshow(image_in,origin="lower left")
        science_aperture.plot(color="r",ax=ax)
        bg_apertures.plot(color="w",ax=ax)
        plt.show()

    return Table(data=[(pos[0],),(pos[1],),(signal_to_noise,),(sigma,),(fap,),(signal,),(snr_sigma,)],
                 names=("pos_x","pos_y","signal_to_noise","sigma","fap","signal","snr_sigma"))
#
# class FalseAlarmProbabilityModule(ProcessingModule):
#
#     def __init__(self,
#                  rough_pos,
#                  aperture_radius,
#                  name_in="get_false_alarm_probability",
#                  image_in_tag="res_median",
#                  method='exact',
#                  skip_closest_apertures=False):
#
#         super(FalseAlarmProbabilityModule, self).__init__(name_in=name_in)
#
#         # Variables
#         self.m_rough_pos = rough_pos
#         self.m_aperture_radius = aperture_radius
#         self.m_method = method
#         self.m_skip_closest_apertures = skip_closest_apertures
#
#         # Ports
#         self.m_image_in_port = self.add_input_port(image_in_tag)
#
#     def run(self):
#
#         image_in = self.m_image_in_port.get_all()
#
#         # Get image shape
#         im_shape = image_in.shape
#
#         # Perform Gaussian fit, if required
#         if self.m_method == "fit":
#
#             # Cut image around pos to resolution of 5 fwhm for Gaussian fit
#             image_cut = Cutout2D(data=image_in,
#                                  position=self.m_rough_pos,
#                                  size=(4 * self.m_aperture_radius, 4 * self.m_aperture_radius),
#                                  mode="partial",
#                                  fill_value=0.).data
#
#             # Fit the source in the science image using a 2D Gaussian
#             gauss_science_init = models.Gaussian2D(amplitude=image_cut[image_cut.shape[0] / 2, image_cut.shape[1] / 2],
#                                                    x_mean=image_cut.shape[0] / 2.,
#                                                    y_mean=image_cut.shape[1] / 2.,
#                                                    x_stddev=self.m_aperture_radius,
#                                                    y_stddev=self.m_aperture_radius,
#                                                    theta=0.)
#
#             fit_gauss = fitting.LevMarLSQFitter()
#
#             y, x = np.mgrid[:image_cut.shape[1], :image_cut.shape[0]]
#             gauss_science = fit_gauss(gauss_science_init, x, y, image_cut)
#
#             tmp_cc_pos = (gauss_science.x_mean.value, gauss_science.y_mean.value)
#
#             # Transform tmp_cc_pos to cc_pos
#             pos = (self.m_rough_pos[0] + (tmp_cc_pos[0] - image_cut.shape[0] / 2.) + .5,
#                    self.m_rough_pos[1] + (tmp_cc_pos[1] - image_cut.shape[1] / 2.) + .5)
#
#
#         else:
#             pos = self.m_rough_pos
#
#         # Vector from image center to point source
#         offset_vector = np.asarray(pos) - np.asarray(im_shape) / 2.
#
#         # Radial separation of point source
#         separation = np.linalg.norm(offset_vector)
#
#         # Position angle of point source
#         pos_angle = np.arctan2(offset_vector[0], offset_vector[1])
#
#         # Calculate number of apertures
#         number_of_apertures = np.floor(np.pi / np.arcsin(self.m_aperture_radius / separation))
#
#         # Azimuthal angles of the apertures
#         azimuthal_angles = np.linspace(0, 2. * np.pi, number_of_apertures + 1)[:-1]
#
#         # Rotation Matrix
#         rot_matrx = lambda phi: np.array([[np.cos(phi), np.sin(phi)],
#                                           [-np.sin(phi), np.cos(phi)]])
#
#         # Get cartesian positions of the apertures
#         aperture_pos = np.array(
#             [np.asarray(im_shape) / 2. + np.dot(rot_matrx(i), offset_vector) for i in azimuthal_angles]).T
#
#         # Define science and background apertures
#         science_aperture = CircularAperture(positions=aperture_pos[:, 0],
#                                             r=self.m_aperture_radius)
#
#         if self.m_skip_closest_apertures:
#             bg_apertures = CircularAperture(positions=aperture_pos[:, 2:-1],
#                                             r=self.m_aperture_radius)
#         else:
#             bg_apertures = CircularAperture(positions=aperture_pos[:, 1:],
#                                             r=self.m_aperture_radius)
#
#         # Determine science flux
#         science_flux = pu.aperture_photometry(data=image_in,
#                                               apertures=science_aperture,
#                                               method="exact")["aperture_sum"].data[0]
#
#         # Determine background fluxes
#         bg_fluxes = pu.aperture_photometry(data=image_in,
#                                            apertures=bg_apertures,
#                                            method="exact")["aperture_sum"].data
#
#         # Perform statistics
#         bg_rms = np.sqrt(np.mean(bg_fluxes ** 2))
#         bg_mean = np.average(bg_fluxes)
#         bg_std = np.std(bg_fluxes)
#
#         self.m_signal_to_noise = science_flux / bg_rms
#
#         self.m_sigma = np.abs((bg_mean - science_flux) / bg_std)
#
#         self.m_fap = 1. - erf(self.m_sigma / np.sqrt(2))
#
#


# if __name__=="__main__":

    # # sigma = np.zeros(83)
    #
    # for i in range(1):
    #
    #     path = "/Users/Alex/Daten/EpsEri_2015/Science/Results/Alex_new/Images/Mask009/normal/posang/subset1pca135.fits"
    #     # path = "/Users/Alex/surfdrive/PhD/Research/Data/targets/ScoCen/data_with_raw_calibs/2MASSJ13251211-6456207/2017/SPHERE/B_H/32_0/Results/2MASSJ13251211-6456207_B_H_res_median_unscaled.fits"
    #     # path = "/Users/Alex/surfdrive/PhD/Research/Data/targets/ScoCen/data_with_raw_calibs/PCA_library/2017/SPHERE/B_H/Results/2MASSJ13444279-6347495/wo_sources_small/2MASSJ13444279-6347495_B_H_res_mean_pca_%s.fits"%i
    #     pos = (50.94,12.06)
    #     # pos = (319.94655,376.08409)
    #     # pos = (34.038348,48.816322)
    #
    #     r_NACO = 3.8e-6/8.4/(2*np.pi)*360*3600/0.02719
    #     r_SPHERE = 1.625e-6/8.4/(2*np.pi)*360*3600/0.01227
    #
    #     im = fits.open(path)[0].data
    #
    #     # # Create test image
    #     # # image size
    #     # n = 201
    #     #
    #     # # Offset from center
    #     # syn_offset_x = 50
    #     # syn_offset_y = -60
    #     #
    #     # print "Offset = (%.3f,%.3f)"%(syn_offset_x,syn_offset_y)
    #     #
    #     # # Create image
    #     # gauss = models.Gaussian2D(amplitude=10,
    #     #                           x_mean=n / 2. + syn_offset_x,
    #     #                           y_mean=n / 2. + syn_offset_y,
    #     #                           x_stddev=1.5,
    #     #                           y_stddev=1.5,
    #     #                           theta=0.)
    #     # # image grid
    #     # y, x = np.mgrid[:n, :n]
    #     #
    #     # # make gaussian function noisy to break symmetry
    #     # im = gauss(x, y) + (np.random.rand(n, n)) - .5
    #     #
    #     # plt.figure()
    #     # plt.imshow(im,origin="lower")
    #     # plt.show()
    #
    #     signal_parameters = get_false_alarm_probability(image_in=im,
    #                                                     rough_pos=pos,
    #                                                     aperture_radius=r_NACO,
    #                                                     method="fit",
    #                                                     skip_closest_apertures=True,
    #                                                     plot=True)
    #
    #     print i,signal_parameters
    #
    #     # sigma[i] = signal_parameters["sigma"]
    #
    # # plt.figure()
    # # plt.plot(range(83),sigma)
    # # plt.xlabel(r"# Principal components")
    # # plt.ylabel(r"Detection significance [$\sigma$]")
    # # plt.grid("on")
    # # plt.savefig("/Users/Alex/Desktop/test.pdf")
    # # plt.show()