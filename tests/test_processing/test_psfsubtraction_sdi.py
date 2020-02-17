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
        assert np.allclose(np.mean(data), 4.16290438829479e-06, rtol=limit, atol=0.)
        assert data.shape == (120, 30, 30)

    def test_psf_subtraction_pca_sdi(self):

        processing_types = ['SDI', 'SDI+ADI', 'ADI+SDI']

        expected = [[1.2921764456313175e-08, -5.74900984529506e-08, 9.27143802109514e-07,
                     5.474132485333015e-08, -1.5527246879617045e-22, -0.00021029055979012687],
                    [1.0031072003694141e-08, -7.50122846446626e-08, 1.6283280901117542e-07,
                     2.0254713965788303e-09, 2.1114469577930802e-23, -0.0002102905597901257],
                    [8.749478614194426e-09, -5.290578240963157e-08, 3.7528498011771007e-07,
                     -2.9705610911195494e-08, 6.994826528812051e-22, -0.0002102905597901452]]

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
