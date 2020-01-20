# -*- coding: utf-8 -*-
"""
Pypeline support funciton to support IFS handling

@author: Sven Kiefer
"""


import sys

from typing import List, Dict
from uncertainties import ufloat


from pynpoint.core.pypeline import Pypeline
from pynpoint.processing.badpixel import BadPixelSigmaFilterModule, BadPixelMapModule, \
                                         ReplaceBadPixelsModule
from pynpoint.processing.centering import WaffleCenteringModule, \
                                          StarAlignmentModule, FitCenterModule, \
                                          ShiftImagesModule
from pynpoint.processing.psfpreparation import AngleCalculationModule, SortParangModule, \
                                               PSFpreparationModule
from pynpoint.processing.stacksubset import DerotateAndStackModule, CombineTagsModule
from pynpoint.processing.psfsubtraction import PcaPsfSubtractionModule, ClassicalADIModule
from pynpoint.processing.resizing import CropImagesModule
from pynpoint.processing.frameselection import SelectGivenAttributesModule
from pynpoint.processing.extract import StarExtractionModule
from pynpoint.processing.fluxposition import SimplexMinimizationModule, FakePlanetModule
from pynpoint.readwrite.fitswriting import FitsWritingModule
from pynpoint.readwrite.fitsreading import FitsReadingModule

#from pynpoint.processing.ifs import IfsScalingModule, CenteringReductionModule, FrameClipModule
#from pynpoint.processing.photometry import SdiAperturePhotometryModule


def IfsSupportModule(image_in_tag: List[str],
                     image_out_tag: List[str],
                     pipe: Pypeline,
                     mod_args: List[Dict],
                     path_lam: str = None,
                     skip_split: bool = False,
                     print_out: bool = True,
                     split_argument: str = 'LAMBDA'):

    """
        This module allows to split a database entery according to the LAMBDA atatribute,
        run modules on the split sets and then recombine them under the image_out_tag.
        Only some Modules are currently supported.

        Parameters
        ----------
        image_in_tag : str
            Tag of the database entries that are read as input and then split in lambdas.
        image_out_tag : str
            Tag of the database entry that is written as output after recombined in
            all lambdas.
        pipe : Pypeline
            Pypline to which the modules should be added.
        mod_args : List[Dict]
            Dictionary with the information for the modules. An example of how to use it
            can be found below
        path_lam : str
            Directory of an example image with the necessary wavelength information.
            If None, the input directory of pipe is taken instead.
        skip_split : bool
            If True, skip the spliting of the attribute. The Pipeline will only work
            if this is either false, or the split attributes are already in the database
        print_out : bool
            If True, write all image_out_tag into fits file after combinig.


        Returns
        -------
        NoneType
            None
    """

    # ----- Step1: splitting datasets
    """
    The splitting step takes information from either the path_lam or path_in and reads
    out the LAMBDA information to set the Modules. If the input_tags were already split,
    this step can be skiped to save computaion time.
    """

    sys.stdout.write('Running IfsSupportModdule .... ')
    sys.stdout.flush()

    # check path_lam:
    if path_lam is None:
        path_lam = pipe._m_input_place

    # open path to path_lam. This step is necessary to accuratly determin the splitting
    # parameters which are used for the rest of the module.
    puplunu = Pypeline(working_place_in=pipe._m_working_place,
                       input_place_in=path_lam,
                       output_place_in=pipe._m_output_place)

    mod_pup = FitsReadingModule(name_in="Pre_Load",
                                input_dir=path_lam,
                                image_tag="pre_load")

    puplunu.add_module(mod_pup)
    puplunu.run()

    # collecting splitting arguments
    lams = puplunu.get_attribute('pre_load', split_argument, static=False)
    lams = list(set(lams))
    lams_str = [str(i) for i in lams]
    lambdas = ['_' + split_argument + i for i in lams_str]

    arg_len = len(split_argument) + 1
    if not skip_split:
        # Seperating input data
        for tag in image_in_tag:
            for lami in lambdas:
                mods = SelectGivenAttributesModule(name_in='WEIN_' + tag + lami,
                                                   image_in_tag=tag,
                                                   selected_out_tag=tag + lami,
                                                   attribute_tag=split_argument,
                                                   attribute_value=lami[arg_len:])
                pipe.add_module(mods)

    # ----- Step2: Add modules to pipeline
    """
    During this step, the modules are added to the Pypeline in order of their input.
    Not all modules are yet supported. It is important to mention that this step is
    entierly before the pipeline is actualy run.
    """

    j = 0
    for mod in mod_args:
        for lamj in lambdas:
            modi = mod_args[j].copy()

            # List of available modules
            if mod_args[j]['module'] == 'BadPixelSigmaFilterModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj
                if 'map_out_tag' not in modi.keys():
                    modi['map_out_tag'] = None
                if modi['map_out_tag'] is not None:
                    modi['map_out_tag'] = mod_args[j]['map_out_tag'] + lamj
                if 'box' not in modi.keys():
                    modi['box'] = 9
                if 'sigma' not in modi.keys():
                    modi['sigma'] = 5.0
                if 'iterate' not in modi.keys():
                    modi['iterate'] = 1

                modu = BadPixelSigmaFilterModule(name_in=modi['name_in'],
                                                 image_in_tag=modi['image_in_tag'],
                                                 image_out_tag=modi['image_out_tag'],
                                                 map_out_tag=modi['map_out_tag'],
                                                 box=modi['box'],
                                                 sigma=modi['sigma'],
                                                 iterate=modi['iterate'])

            elif mod_args[j]['module'] == 'WaffleCenteringModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['center_in_tag'] = mod_args[j]['center_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj
                if 'l_min' not in modi.keys():
                    modi['l_min'] = None
                if 'size' not in modi.keys():
                    modi['size'] = None
                if 'size' not in modi.keys():
                    modi['size'] = None
                if 'pattern' not in modi.keys():
                    modi['pattern'] = None
                if 'center' not in modi.keys():
                    modi['center'] = 45
                if 'angle' not in modi.keys():
                    modi['angle'] = 0
                if 'sigma' not in modi.keys():
                    modi['sigma'] = 0.06
                if 'dither' not in modi.keys():
                    modi['dither'] = False

                modu = WaffleCenteringModule(name_in=modi['name_in'],
                                             image_in_tag=modi['image_in_tag'],
                                             center_in_tag=modi['center_in_tag'],
                                             image_out_tag=modi['image_out_tag'],
                                             radius=modi['radius'],
                                             l_min=modi['l_min'],
                                             size=modi['size'],
                                             pattern=modi['pattern'],
                                             center=modi['center'],
                                             angle=modi['angle'],
                                             sigma=modi['sigma'],
                                             dither=modi['dither'])

            elif mod_args[j]['module'] == 'FitsWritingModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['data_tag'] = mod_args[j]['data_tag'] + lamj
                modi['file_name'] = mod_args[j]['file_name'][:-5] + lamj + '.fits'
                if 'output_dir' not in modi.keys():
                    modi['output_dir'] = None
                if 'data_range' not in modi.keys():
                    modi['data_range'] = None
                if 'overwrite' not in modi.keys():
                    modi['overwrite'] = True
                if 'subset_size' not in modi.keys():
                    modi['subset_size'] = None

                modu = FitsWritingModule(name_in=modi['name_in'],
                                         data_tag=modi['data_tag'],
                                         file_name=modi['file_name'],
                                         output_dir=modi['output_dir'],
                                         data_range=modi['data_range'],
                                         overwrite=modi['overwrite'],
                                         subset_size=modi['subset_size'])

            elif mod_args[j]['module'] == 'AngleCalculationModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['data_tag'] = mod_args[j]['data_tag'] + lamj
                if 'instrument' not in modi.keys():
                    modi['instrument'] = 'NACO'

                modu = AngleCalculationModule(name_in=modi['name_in'],
                                              data_tag=modi['data_tag'],
                                              instrument=modi['instrument'])

            elif mod_args[j]['module'] == 'BadPixelMapModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                if modi['dark_in_tag'] is not None:
                    modi['dark_in_tag'] = mod_args[j]['dark_tag_in'] + lamj
                if modi['flat_tag_in'] is not None:
                    modi['flat_tag_in'] = mod_args[j]['flat_tag_in'] + lamj
                modi['bp_map_out_tag'] = mod_args[j]['bp_map_out_tag'] + lamj
                if 'dark_threshold' not in modi.keys():
                    modi['dark_threshold'] = 0.2
                if 'flat_threshold' not in modi.keys():
                    modi['flat_threshold'] = 0.2

                modu = BadPixelMapModule(name_in=modi['name_in'],
                                         dark_in_tag=modi['dark_in_tag'],
                                         flat_in_tag=modi['flat_in_tag'],
                                         bp_map_out_tag=modi['bp_map_out_tag'],
                                         dark_threshold=modi['dark_threshold'],
                                         flat_threshold=modi['flat_threshold'])

            elif mod_args[j]['module'] == 'ReplaceBadPixelsModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['map_in_tag'] = mod_args[j]['map_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj
                if 'sigma' not in modi.keys():
                    modi['sigma'] = None
                if 'size' not in modi.keys():
                    modi['size'] = 2
                if 'replace' not in modi.keys():
                    modi['replace'] = 'median'

                modu = ReplaceBadPixelsModule(name_in=modi['name_in'],
                                              image_in_tag=modi['image_in_tag'],
                                              map_in_tag=modi['map_in_tag'],
                                              image_out_tag=modi['image_out_tag'],
                                              sigma=modi['sigma'],
                                              size=modi['size'],
                                              replace=modi['replace'])

            elif mod_args[j]['module'] == 'IfsScalingModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj
                if 'scaling' not in modi.keys():
                    modi['scaling'] = (1.0, 1.0, 1.0)
                if 'angle' not in modi.keys():
                    modi['angle'] = 0
                if 'pixscale' not in modi.keys():
                    modi['pixscale'] = False

                modu = IfsScalingModule(name_in=modi['name_in'],
                                        image_in_tag=modi['image_in_tag'],
                                        image_out_tag=modi['image_out_tag'],
                                        scaling=modi['scaling'],
                                        angle=modi['angle'],
                                        pixscale=modi['pixscale'])

            elif mod_args[j]['module'] == 'SortParangModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj

                modu = SortParangModule(name_in=modi['name_in'],
                                        image_in_tag=modi['image_in_tag'],
                                        image_out_tag=modi['image_out_tag'])

            elif mod_args[j]['module'] == 'DerotateAndStackModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj
                if 'derotate' not in modi.keys():
                    modi['derotate'] = True
                if 'stack' not in modi.keys():
                    modi['stack'] = None
                if 'extra_rot' not in modi.keys():
                    modi['extra_rot'] = 0.0

                modu = DerotateAndStackModule(name_in=modi['name_in'],
                                              image_in_tag=modi['image_in_tag'],
                                              image_out_tag=modi['image_out_tag'],
                                              derotate=modi['derotate'],
                                              stack=modi['stack'],
                                              extra_rot=modi['extra_rot'])

            elif mod_args[j]['module'] == 'ClassicalADIModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['res_out_tag'] = mod_args[j]['res_out_tag'] + lamj
                modi['stack_out_tag'] = mod_args[j]['stack_out_tag'] + lamj
                if 'threshold' not in modi.keys():
                    modi['threshold'] = None
                if 'nreference' not in modi.keys():
                    modi['nreference'] = None
                if 'residuals' not in modi.keys():
                    modi['residuals'] = 'median'
                if 'extra_rot' not in modi.keys():
                    modi['extra_rot'] = 0.0

                modu = ClassicalADIModule(name_in=modi['name_in'],
                                          image_in_tag=modi['image_in_tag'],
                                          res_out_tag=modi['res_out_tag'],
                                          stack_out_tag=modi['stack_out_tag'],
                                          threshold=modi['threshold'],
                                          nreference=modi['nreference'],
                                          residuals=modi['residuals'],
                                          extra_rot=modi['extra_rot'])

            elif mod_args[j]['module'] == 'CropImagesModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj
                if 'center' not in modi.keys():
                    modi['center'] = None

                modu = CropImagesModule(name_in=modi['name_in'],
                                        image_in_tag=modi['image_in_tag'],
                                        image_out_tag=modi['image_out_tag'],
                                        size=modi['size'],
                                        center=modi['center'])

            elif mod_args[j]['module'] == 'PSFpreparationModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj
                modi['mask_out_tag'] = mod_args[j]['mask_out_tag'] + lamj
                if 'norm' not in modi.keys():
                    modi['norm'] = False
                if 'resize' not in modi.keys():
                    modi['resize'] = None
                if 'cent_size' not in modi.keys():
                    modi['cent_size'] = None
                if 'edge_size' not in modi.keys():
                    modi['edge_size'] = None

                modu = PSFpreparationModule(name_in=modi['name_in'],
                                            image_in_tag=modi['image_in_tag'],
                                            image_out_tag=modi['image_out_tag'],
                                            mask_out_tag=modi['mask_out_tag'],
                                            norm=modi['norm'],
                                            resize=modi['resize'],
                                            cent_size=modi['cent_size'],
                                            edge_size=modi['edge_size'])

            elif mod_args[j]['module'] == 'CenteringReductionModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['center_in_tag'] = mod_args[j]['center_in_tag'] + lamj
                modi['center_out_tag'] = mod_args[j]['center_out_tag'] + lamj

                modu = CenteringReductionModule(name_in=modi['name_in'],
                                                image_in_tag=modi['image_in_tag'],
                                                center_in_tag=modi['center_in_tag'],
                                                center_out_tag=modi['center_out_tag'])

            elif mod_args[j]['module'] == 'StarExtractionModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj
                if 'index_out_tag' not in modi.keys():
                    modi['index_out_tag'] = None
                if modi['index_out_tag'] is not None:
                    modi['index_out_tag'] = mod_args[j]['index_out_tag'] + lamj
                if 'image_size' not in modi.keys():
                    modi['image_size'] = 2.0
                if 'fwhm_star' not in modi.keys():
                    modi['fwhm_star'] = 0.2
                if 'position' not in modi.keys():
                    modi['position'] = None

                modu = StarExtractionModule(name_in=modi['name_in'],
                                            image_in_tag=modi['image_in_tag'],
                                            image_out_tag=modi['image_out_tag'],
                                            index_out_tag=modi['index_out_tag'],
                                            image_size=modi['image_size'],
                                            fwhm_star=modi['fwhm_star'],
                                            position=modi['position'])

            elif mod_args[j]['module'] == 'StarAlignmentModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj
                if 'ref_image_in_tag' not in modi.keys():
                    modi['ref_image_in_tag'] = None
                if modi['ref_image_in_tag'] is not None:
                    modi['ref_image_in_tag'] = mod_args[j]['ref_image_in_tag'] + lamj
                if 'interpolation' not in modi.keys():
                    modi['interpolation'] = 'spline'
                if 'accuracy' not in modi.keys():
                    modi['accuracy'] = 10.0
                if 'resize' not in modi.keys():
                    modi['resize'] = None
                if 'num_references' not in modi.keys():
                    modi['num_references'] = 10
                if 'subframe' not in modi.keys():
                    modi['subframe'] = None

                modu = StarAlignmentModule(name_in=modi['name_in'],
                                           image_in_tag=modi['image_in_tag'],
                                           ref_image_in_tag=modi['ref_image_in_tag'],
                                           image_out_tag=modi['image_out_tag'],
                                           interpolation=modi['interpolation'],
                                           accuracy=modi['accuracy'],
                                           resize=modi['resize'],
                                           num_references=modi['num_references'],
                                           subframe=modi['subframe'])

            elif mod_args[j]['module'] == 'FitCenterModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['fit_out_tag'] = mod_args[j]['fit_out_tag'] + lamj
                if 'mask_out_tag' not in modi.keys():
                    modi['mask_out_tag'] = None
                if modi['mask_out_tag'] is not None:
                    modi['mask_out_tag'] = mod_args[j]['mask_out_tag'] + lamj
                if 'method' not in modi.keys():
                    modi['method'] = 'full'
                if 'radius' not in modi.keys():
                    modi['radius'] = 0.1
                if 'sign' not in modi.keys():
                    modi['sign'] = 'positive'
                if 'model' not in modi.keys():
                    modi['model'] = 'gaussian'
                if 'filter_size' not in modi.keys():
                    modi['filter_size'] = None

                modu = FitCenterModule(name_in=modi['name_in'],
                                       image_in_tag=modi['image_in_tag'],
                                       fit_out_tag=modi['fit_out_tag'],
                                       mask_out_tag=modi['mask_out_tag'],
                                       method=modi['method'],
                                       radius=modi['radius'],
                                       sign=modi['sign'],
                                       model=modi['model'],
                                       filter_size=modi['filter_size'])

            elif mod_args[j]['module'] == 'ShiftImagesModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj
                modi['shift_xy'] = mod_args[j]['shift_xy'] + lamj
                if 'interpolation' not in modi.keys():
                    modi['interpolation'] = 'spline'

                modu = ShiftImagesModule(name_in=modi['name_in'],
                                         image_in_tag=modi['image_in_tag'],
                                         image_out_tag=modi['image_out_tag'],
                                         shift_xy=modi['shift_xy'],
                                         interpolation=modi['interpolation'])

            elif mod_args[j]['module'] == 'SdiAperturePhotometryModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['psf_in_tag'] = mod_args[j]['psf_in_tag'] + lamj
                modi['flux_position_tag'] = mod_args[j]['flux_position_tag'] + lamj
                if 'flux_filter' not in modi.keys():
                    modi['flux_filter'] = 'ND_0.0'
                if 'psf_scaling' not in modi.keys():
                    modi['psf_scaling'] = 1
                if 'pixscale_corr' not in modi.keys():
                    modi['pixscale_corr'] = ufloat(0.01227, 0.00002)
                if 'TN' not in modi.keys():
                    modi['TN'] = ufloat(-1.75, 0.1)
                if 'cutout_size' not in modi.keys():
                    modi['cutout_size'] = 21
                if 'fit_each_image' not in modi.keys():
                    modi['fit_each_image'] = False
                if 'aperture_size' not in modi.keys():
                    modi['aperture_size'] = 'fwhm'

                modu = SdiAperturePhotometryModule(name_in=modi['name_in'],
                                                   image_in_tag=modi['image_in_tag'],
                                                   psf_in_tag=modi['psf_in_tag'],
                                                   flux_position_tag=modi['flux_position_tag'],
                                                   rough_position=modi['rough_position'],
                                                   flux_filter=modi['flux_filter'],
                                                   psf_scaling=modi['psf_scaling'],
                                                   pixscale_corr=modi['pixscale_corr'],
                                                   TN=modi['TN'],
                                                   cutout_size=modi['cutout_size'],
                                                   fit_each_image=modi['fit_each_image'],
                                                   aperture_size=modi['aperture_size'])

            elif mod_args[j]['module'] == 'SimplexMinimizationModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['psf_in_tag'] = mod_args[j]['psf_in_tag'] + lamj
                modi['res_out_tag'] = mod_args[j]['res_out_tag'] + lamj
                modi['flux_position_tag'] = mod_args[j]['flux_position_tag'] + lamj
                if 'psf_scaling' not in modi.keys():
                    modi['psf_scaling'] = None
                if 'flux_filter' not in modi.keys():
                    modi['flux_filter'] = 'ND_0.0'
                if 'merit' not in modi.keys():
                    modi['merit'] = 'hessian'
                if 'aperture' not in modi.keys():
                    modi['aperture'] = 0.1
                if 'sigma' not in modi.keys():
                    modi['sigma'] = 0.0
                if 'tolerance' not in modi.keys():
                    modi['tolerance'] = 0.1
                if 'pca_number' not in modi.keys():
                    modi['pca_number'] = 10
                if 'cent_size' not in modi.keys():
                    modi['cent_size'] = None
                if 'edge_size' not in modi.keys():
                    modi['edge_size'] = None
                if 'extra_rot' not in modi.keys():
                    modi['extra_rot'] = 0.0
                if 'residuals' not in modi.keys():
                    modi['residuals'] = 'median'
                if 'reference_in_tag' not in modi.keys():
                    modi['reference_in_tag'] = None
                if modi['reference_in_tag'] is not None:
                    modi['reference_in_tag'] = mod_args[j]['reference_in_tag'] + lamj
                if 'processing_type' not in modi.keys():
                    modi['processing_type'] = 'Cadi'

                modu = SimplexMinimizationModule(name_in=modi['name_in'],
                                                 image_in_tag=modi['image_in_tag'],
                                                 psf_in_tag=modi['psf_in_tag'],
                                                 res_out_tag=modi['res_out_tag'],
                                                 flux_position_tag=modi['flux_position_tag'],
                                                 position=modi['position'],
                                                 magnitude=modi['magnitude'],
                                                 psf_scaling=modi['psf_scaling'],
                                                 flux_filter=modi['flux_filter'],
                                                 merit=modi['merit'],
                                                 aperture=modi['aperture'],
                                                 sigma=modi['sigma'],
                                                 tolerance=modi['tolerance'],
                                                 pca_number=modi['pca_number'],
                                                 cent_size=modi['cent_size'],
                                                 edge_size=modi['edge_size'],
                                                 extra_rot=modi['extra_rot'],
                                                 residuals=modi['residuals'],
                                                 reference_in_tag=modi['reference_in_tag'],
                                                 processing_type=modi['processing_type'])

            elif mod_args[j]['module'] == 'FrameClipModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj
                if 'clip_length' not in modi.keys():
                    modi['clip_length'] = 1
                if 'clip_mode' not in modi.keys():
                    modi['clip_mode'] = 'end'

                modu = FrameClipModule(name_in=modi['name_in'],
                                       image_in_tag=modi['image_in_tag'],
                                       image_out_tag=modi['image_out_tag'],
                                       clip_length=modi['clip_length'],
                                       clip_mode=modi['clip_mode'])

            elif mod_args[j]['module'] == 'PcaPsfSubtractionModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['images_in_tag'] = mod_args[j]['images_in_tag'] + lamj
                modi['reference_in_tag'] = mod_args[j]['reference_in_tag'] + lamj
                if 'res_mean_tag' not in modi.keys():
                    modi['res_mean_tag'] = None
                if modi['res_mean_tag'] is not None:
                    modi['res_mean_tag'] = mod_args[j]['res_mean_tag'] + lamj
                if 'res_median_tag' not in modi.keys():
                    modi['res_median_tag'] = None
                if modi['res_median_tag'] is not None:
                    modi['res_median_tag'] = mod_args[j]['res_median_tag'] + lamj
                if 'res_weighted_tag' not in modi.keys():
                    modi['res_weighted_tag'] = None
                if modi['res_weighted_tag'] is not None:
                    modi['res_weighted_tag'] = mod_args[j]['res_weighted_tag'] + lamj
                if 'res_stim_tag' not in modi.keys():
                    modi['res_stim_tag'] = None
                if modi['res_stim_tag'] is not None:
                    modi['res_stim_tag'] = mod_args[j]['res_stim_tag'] + lamj
                if 'res_rot_mean_clip_tag' not in modi.keys():
                    modi['res_rot_mean_clip_tag'] = None
                if modi['res_rot_mean_clip_tag'] is not None:
                    modi['res_rot_mean_clip_tag'] = mod_args[j]['res_rot_mean_clip_tag']+lamj
                if 'res_arr_out_tag' not in modi.keys():
                    modi['res_arr_out_tag'] = None
                if modi['res_arr_out_tag'] is not None:
                    modi['res_arr_out_tag'] = mod_args[j]['res_arr_out_tag'] + lamj
                if 'basis_out_tag' not in modi.keys():
                    modi['basis_out_tag'] = None
                if modi['basis_out_tag'] is not None:
                    modi['basis_out_tag'] = mod_args[j]['basis_out_tag'] + lamj
                if 'pca_numbers' not in modi.keys():
                    modi['pca_numbers'] = range(1, 21)
                if 'extra_rot' not in modi.keys():
                    modi['extra_rot'] = 0
                if 'subtract_mean' not in modi.keys():
                    modi['subtract_mean'] = None
                if 'processing_type' not in modi.keys():
                    modi['processing_type'] = 'Tadi'

                modu = PcaPsfSubtractionModule(name_in=modi['name_in'],
                                               images_in_tag=modi['images_in_tag'],
                                               reference_in_tag=modi['reference_in_tag'],
                                               res_mean_tag=modi['res_mean_tag'],
                                               res_median_tag=modi['res_median_tag'],
                                               res_weighted_tag=modi['res_weighted_tag'],
                                               res_stim_tag=modi['res_stim_tag'],
                                               res_arr_out_tag=modi['res_arr_out_tag'],
                                               basis_out_tag=modi['basis_out_tag'],
                                               pca_numbers=modi['pca_numbers'],
                                               extra_rot=modi['extra_rot'],
                                               subtract_mean=modi['subtract_mean'],
                                               processing_type=modi['processing_type'])

            elif mod_args[j]['module'] == 'FakePlanetModule':

                modi['name_in'] = mod_args[j]['name_in'] + lamj
                modi['image_in_tag'] = mod_args[j]['image_in_tag'] + lamj
                modi['psf_in_tag'] = mod_args[j]['psf_in_tag'] + lamj
                modi['image_out_tag'] = mod_args[j]['image_out_tag'] + lamj
                if 'psf_scaling' not in modi.keys():
                    modi['psf_scaling'] = 1.
                if 'interpolation' not in modi.keys():
                    modi['clip_mode'] = 'spline'

                modu = FakePlanetModule(name_in=modi['name_in'],
                                        image_in_tag=modi['image_in_tag'],
                                        psf_in_tag=modi['psf_in_tag'],
                                        image_out_tag=modi['image_out_tag'],
                                        position=modi['position'],
                                        magnitude=modi['magnitude'],
                                        psf_scaling=modi['psf_scaling'],
                                        interpolation=modi['interpolation'])

            else:
                raise ValueError('The module '+mod_args[j]['module']+' is not implemented')

            # Add the selected and prepared module
            pipe.add_module(modu)

        j += 1

    # ----- Step3: Recombine all splitted tags
    """
    Recombines all image_out_tag after recombining them. The output_tag is called
    WEOUT_ before the image tag to avoid ambiguty.
    """

    for imgs in image_out_tag:

        # Recombine all wavelengths
        names = [imgs + lum for lum in lambdas]
        mods = CombineTagsModule(name_in='WEOUT_' + imgs,
                                 image_in_tags=names,
                                 image_out_tag='WEOUT_' + imgs)
        pipe.add_module(mods)

        # Add fits writing modules if printing is wished
        if print_out:
            modk = FitsWritingModule(file_name='WEOUT_' + imgs + '.fits',
                                     name_in='write_WEOUT_' + imgs,
                                     data_tag='WEOUT_' + imgs)
            pipe.add_module(modk)

    sys.stdout.write('[Done]\n')
    sys.stdout.flush()
