import os
import warnings

import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.psfpreparation import AngleInterpolationModule, PSFpreparationModule
from pynpoint.processing.psfsubtraction import PcaPsfSubtractionModule
from pynpoint.util.tests import create_config, create_ifs_fake, remove_test_data

warnings.simplefilter('always')

limit = 1e-10


class TestPsfSubtractionSdi:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_ifs_fake(path=self.test_dir+'science')

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=['science', ])

    def test_read_data(self):

        read = FitsReadingModule(name_in='read',
                                 image_tag='science',
                                 input_dir=self.test_dir+'science')

        self.pipeline.add_module(read)
        self.pipeline.run_module('read')

    def test_angle_interpolation(self):

        angle = AngleInterpolationModule(name_in='angle',
                                         data_tag='science')

        self.pipeline.set_attribute('science', 'LAMBDA',
                                    tuple([0.953 + i*0.0190526315789474 for i in range(6)])*20,
                                    static=False)

        self.pipeline.add_module(angle)
        self.pipeline.run_module('angle')

        data = self.pipeline.get_data('header_science/PARANG')
        assert np.allclose(data[0], 0., rtol=limit, atol=0.)
        assert np.allclose(data[91], 78.94736842105263, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 50.0, rtol=limit, atol=0.)
        assert data.shape == (120, )

    def test_psf_preparation(self):

        prep = PSFpreparationModule(name_in='prep',
                                    image_in_tag='science',
                                    image_out_tag='science_prep',
                                    mask_out_tag=None,
                                    norm=False,
                                    resize=None,
                                    cent_size=0.2,
                                    edge_size=1.0)

        self.pipeline.add_module(prep)
        self.pipeline.run_module('prep')

        data = self.pipeline.get_data('science_prep')
        assert np.allclose(data[0, 0, 0], 0.0, rtol=limit, atol=0.)
        assert np.allclose(data[0, 8, 8], 0.0003621069828250913, rtol=limit, atol=0.)
        assert np.allclose(data[0, 29, 29], 0.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 4.156820430599551e-06, rtol=limit, atol=0.)
        assert data.shape == (120, 30, 30)

    def test_psf_subtraction_pca_sdi(self):

        processing_types = ['SDI', 'SDI+ADI', 'ADI+SDI']

        expected = [[1.5502068572032085e-08, -1.224854217408181e-07, 7.054529192798392e-07,
                     3.464083328067893e-08, 1.6513116426974567e-22, 7.601169269101861e-05],
                    [9.478509300769893e-09, -1.1539356955933606e-07, -5.28457028293079e-07,
                     5.650095065537256e-09, 1.779170017385036e-23, 7.601169269101619e-05],
                    [1.939796067613671e-08, -1.1398642151499491e-07, -3.0519309870026413e-06,
                     -4.185023101449001e-08, -3.7806203118993437e-22, 7.601169269099827e-05]]

        for i, p_type in enumerate(processing_types):
            pca = PcaPsfSubtractionModule(pca_numbers=range(1, 6),
                                          name_in='pca_single_sdi_' + p_type,
                                          images_in_tag='science_prep',
                                          reference_in_tag='science_prep',
                                          res_mean_tag='res_mean_single_sdi_' + p_type,
                                          res_median_tag='res_median_single_sdi_' + p_type,
                                          res_weighted_tag='res_weighted_single_sdi_' + p_type,
                                          res_rot_mean_clip_tag='res_clip_single_sdi_' + p_type,
                                          res_arr_out_tag='res_arr_single_sdi_' + p_type,
                                          basis_out_tag='basis_single_sdi_' + p_type,
                                          extra_rot=-15.,
                                          subtract_mean=True,
                                          processing_type=p_type)

            self.pipeline.add_module(pca)
            self.pipeline.run_module('pca_single_sdi_' + p_type)

            data = self.pipeline.get_data('res_mean_single_sdi_' + p_type)
            assert np.allclose(np.mean(data), expected[i][0], rtol=limit, atol=0.)
            assert data.shape == (30, 30, 30)

            data = self.pipeline.get_data('res_median_single_sdi_' + p_type)
            assert np.allclose(np.mean(data), expected[i][1], rtol=limit, atol=0.)
            assert data.shape == (30, 30, 30)

            data = self.pipeline.get_data('res_weighted_single_sdi_' + p_type)
            assert np.allclose(np.mean(data), expected[i][2], rtol=limit, atol=0.)
            assert data.shape == (30, 30, 30)

            data = self.pipeline.get_data('res_clip_single_sdi_' + p_type)
            assert np.allclose(np.mean(data), expected[i][3], rtol=limit, atol=0.)
            assert data.shape == (30, 30, 30)

            data = self.pipeline.get_data('res_arr_single_sdi_' + p_type + '5')
            assert np.allclose(np.mean(data), expected[i][4], rtol=limit, atol=0.)
            assert data.shape == (120, 30, 30)

            data = self.pipeline.get_data('basis_single_sdi_' + p_type)
            assert np.allclose(np.mean(data), expected[i][5], rtol=limit, atol=0.)
            assert data.shape == (5, 30, 30)
