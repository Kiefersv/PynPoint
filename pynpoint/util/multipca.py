"""
Capsule for multiprocessing of the PSF subtraction with PCA. Residuals are created in parallel for
a range of principal components for which the PCA basis is required as input.
"""

import sys
import multiprocessing

from typing import Union, Tuple, List

import numpy as np

from typeguard import typechecked
from sklearn.decomposition.pca import PCA

from pynpoint.core.dataio import OutputPort
from pynpoint.util.multiproc import TaskProcessor, TaskCreator, TaskWriter, TaskResult, \
                                    TaskInput, MultiprocessingCapsule, to_slice
from pynpoint.util.sdi import postprocessor
from pynpoint.util.ifs import i_want_to_seperate_wavelengths
from pynpoint.util.residuals import combine_residuals


class PcaTaskCreator(TaskCreator):
    """
    The TaskCreator of the PCA multiprocessing. Creates one task for each principal component
    number. Does not require an input port since the data is directly given to the task processors.
    """

    @typechecked
    def __init__(self,
                 tasks_queue_in: multiprocessing.JoinableQueue,
                 num_proc: np.int64,
                 pca_numbers: np.ndarray) -> None:
        """
        Parameters
        ----------
        tasks_queue_in : multiprocessing.queues.JoinableQueue
            Input task queue.
        num_proc : int
            Number of processors.
        pca_numbers : numpy.ndarray
            Principal components for which the residuals are computed.

        Returns
        -------
        NoneType
            None
        """

        super(PcaTaskCreator, self).__init__(None, tasks_queue_in, None, num_proc)

        self.m_pca_numbers = pca_numbers

    @typechecked
    def run(self) -> None:
        """
        Run method of PcaTaskCreator.

        Returns
        -------
        NoneType
            None
        """

        for i, pca_number in enumerate(self.m_pca_numbers):
            parameters = (((i, i+1, None), (None, None, None), (None, None, None)), )
            self.m_task_queue.put(TaskInput(pca_number, parameters))

        self.create_poison_pills()


class PcaTaskProcessor(TaskProcessor):
    """
    The TaskProcessor of the PCA multiprocessing is the core of the parallelization. An instance
    of this class will calculate one forward and backward PCA transformation given the pre-trained
    scikit-learn PCA model. It does not get data from the TaskCreator but uses its own copy of the
    input data, which are the same and independent for each task. The following residuals can be
    created:

    * Mean residuals -- requirements[0] = True
    * Median residuals -- requirements[1] = True
    * Noise-weighted residuals -- requirements[2] = True
    * stim residuals -- requirements[3] = True
    * Clipped mean of the residuals -- requirements[4] = True
    """

    @typechecked
    def __init__(self,
                 tasks_queue_in: multiprocessing.JoinableQueue,
                 result_queue_in: multiprocessing.JoinableQueue,
                 star_reshape: np.ndarray,
                 angles: np.ndarray,
                 scales: np.ndarray,
                 pca_model: PCA,
                 im_shape: Tuple[int, int, int],
                 indices: np.ndarray,
                 requirements: Tuple[bool, bool, bool, bool, bool],
                 processing_type) -> None:
        """
        Parameters
        ----------
        tasks_queue_in : multiprocessing.queues.JoinableQueue
            Input task queue.
        result_queue_in : multiprocessing.queues.JoinableQueue
            Input result queue.
        star_reshape : numpy.ndarray
            Reshaped (2D) stack of images.
        angles : numpy.ndarray
            Derotation angles (deg).
        scales : numpy.ndarray
            Rescaling values
        pca_model : sklearn.decomposition.pca.PCA
            PCA object with the basis.
        im_shape : tuple(int, int, int)
            Original shape of the stack of images.
        indices : numpy.ndarray
            Non-masked image indices.
        requirements : tuple(bool, bool, bool, bool)
            Required output residuals.

        Returns
        -------
        NoneType
            None
        """

        super(PcaTaskProcessor, self).__init__(tasks_queue_in, result_queue_in)

        self.m_star_reshape = star_reshape
        self.m_pca_model = pca_model
        self.m_angles = angles
        self.m_scales = scales
        self.m_im_shape = im_shape
        self.m_indices = indices
        self.m_requirements = requirements
        self.m_processing_type = processing_type

    @typechecked
    def run_job(self,
                tmp_task: TaskInput) -> TaskResult:
        """
        Run method of PcaTaskProcessor.

        Parameters
        ----------
        tmp_task : pynpoint.util.multiproc.TaskInput
            Input task.

        Returns
        -------
        pynpoint.util.multiproc.TaskResult
            Output residuals.
        """

        residuals, res_rot = postprocessor(images=self.m_star_reshape,
                                           parang=self.m_angles,
                                           scales=self.m_scales,
                                           pca_number=tmp_task.m_input_data,
                                           pca_sklearn=self.m_pca_model,
                                           im_shape=self.m_im_shape,
                                           indices=self.m_indices,
                                           processing_type=self.m_processing_type)

        # Prepare either single or wavelength output
        if i_want_to_seperate_wavelengths(self.m_processing_type):
            res_output = np.zeros((5, len(list(set(self.m_scales))), res_rot.shape[1], res_rot.shape[2]))
        else:
            res_output = np.zeros((5, 1, res_rot.shape[1], res_rot.shape[2]))

        if self.m_requirements[0]:
            res_output[0, ] = combine_residuals(method='mean',
                                                res_rot=res_rot,
                                                lam=1/self.m_scales,
                                                processing_type=self.m_processing_type)

        if self.m_requirements[1]:
            res_output[1, ] = combine_residuals(method='median',
                                                res_rot=res_rot,
                                                lam=1/self.m_scales,
                                                processing_type=self.m_processing_type)

        if self.m_requirements[2]:
            res_output[2, ] = combine_residuals(method='weighted',
                                                res_rot=res_rot,
                                                residuals=residuals,
                                                angles=self.m_angles,
                                                lam=1/self.m_scales,
                                                processing_type=self.m_processing_type)

        if self.m_requirements[3]:
            res_output[3, ] = combine_residuals(method='stim',
                                                res_rot=res_rot,
                                                residuals=residuals,
                                                angles=self.m_angles,
                                                lam=1/self.m_scales,
                                                processing_type=self.m_processing_type)

        if self.m_requirements[4]:
            res_output[4, ] = combine_residuals(method='clipped',
                                                res_rot=res_rot,
                                                lam=1/self.m_scales,
                                                processing_type=self.m_processing_type)

        sys.stdout.write('.')
        sys.stdout.flush()

        return TaskResult(res_output, tmp_task.m_job_parameter[0])


class PcaTaskWriter(TaskWriter):
    """
    The TaskWriter of the PCA parallelization. Four different ports are used to save the
    results of the task processors (mean, median, weighted, stim, and clipped).
    """

    @typechecked
    def __init__(self,
                 result_queue_in: multiprocessing.JoinableQueue,
                 mean_out_port: Union[OutputPort, None],
                 median_out_port: Union[OutputPort, None],
                 weighted_out_port: Union[OutputPort, None],
                 stim_out_port: Union[OutputPort, None],
                 clip_out_port: Union[OutputPort, None],
                 data_mutex_in: multiprocessing.Lock,
                 requirements: Tuple[bool, bool, bool, bool, bool]) -> None:
        """
        Constructor of PcaTaskWriter.

        Parameters
        ----------
        result_queue_in : multiprocessing.queues.JoinableQueue
            Input result queue.
        mean_out_port : pynpoint.core.dataio.OutputPort
            Output port with the mean residuals. Not used if set to None.
        median_out_port : pynpoint.core.dataio.OutputPort
            Output port with the median residuals. Not used if set to None.
        weighted_out_port : pynpoint.core.dataio.OutputPort
            Output port with the noise-weighted residuals. Not used if set to None.
        stim_out_port : pynpoint.core.dataio.OutputPort
            Output port with the stim residuals. Not used if set to None.
        clip_out_port : pynpoint.core.dataio.OutputPort
            Output port with the clipped mean residuals. Not used if set to None.
        data_mutex_in : multiprocessing.synchronize.Lock
            A mutual exclusion variable which ensure that no read and write simultaneously occur.
        requirements : tuple(bool, bool, bool, bool)
            Required output residuals.

        Returns
        -------
        NoneType
            None
        """

        super(PcaTaskWriter, self).__init__(result_queue_in, None, data_mutex_in)

        self.m_mean_out_port = mean_out_port
        self.m_median_out_port = median_out_port
        self.m_weighted_out_port = weighted_out_port
        self.m_stim_out_port = stim_out_port
        self.m_clip_out_port = clip_out_port
        self.m_requirements = requirements

    @typechecked
    def run(self) -> None:
        """
        Run method of PcaTaskWriter. Writes the residuals to the output ports.

        Returns
        -------
        NoneType
            None
        """

        while True:
            next_result = self.m_result_queue.get()
            poison_pill_case = self.check_poison_pill(next_result)

            if poison_pill_case == 1:
                break

            elif poison_pill_case == 2:
                continue

            with self.m_data_mutex:
                res_slice = to_slice(next_result.m_position)
                pca_nr = next_result.m_position[0][0]
                ll = len(next_result.m_data_array[0, :, 0, 0])

                if self.m_requirements[0]:
                    self.m_mean_out_port._check_status_and_activate()
                    self.m_mean_out_port[pca_nr*ll:(pca_nr+1)*ll, :, :] = next_result.m_data_array[0, :, :, :]
                    self.m_mean_out_port.close_port()

                if self.m_requirements[1]:
                    self.m_median_out_port._check_status_and_activate()
                    self.m_median_out_port[pca_nr*ll:(pca_nr+1)*ll, :, :] = next_result.m_data_array[1, :, :, :]
                    self.m_median_out_port.close_port()

                if self.m_requirements[2]:
                    self.m_weighted_out_port._check_status_and_activate()
                    self.m_weighted_out_port[pca_nr*ll:(pca_nr+1)*ll, :, :] = next_result.m_data_array[2, :, :, :]
                    self.m_weighted_out_port.close_port()

                if self.m_requirements[3]:
                    self.m_stim_out_port._check_status_and_activate()
                    self.m_stim_out_port[pca_nr*ll:(pca_nr+1)*ll, :, :] = next_result.m_data_array[3, :, :, :]
                    self.m_stim_out_port.close_port()

                if self.m_requirements[4]:
                    self.m_clip_out_port._check_status_and_activate()
                    self.m_clip_out_port[pca_nr*ll:(pca_nr+1)*ll, :, :] = next_result.m_data_array[4, :, :, :]
                    self.m_clip_out_port.close_port()

            self.m_result_queue.task_done()


class PcaMultiprocessingCapsule(MultiprocessingCapsule):
    """
    Capsule for PCA multiprocessing with the poison pill pattern.
    """

    @typechecked
    def __init__(self,
                 mean_out_port: Union[OutputPort, None],
                 median_out_port: Union[OutputPort, None],
                 weighted_out_port: Union[OutputPort, None],
                 stim_out_port: Union[OutputPort, None],
                 clip_out_port: Union[OutputPort, None],
                 num_proc: np.int64,
                 pca_numbers: np.ndarray,
                 pca_model: PCA,
                 star_reshape: np.ndarray,
                 angles: np.ndarray,
                 scales: np.ndarray,
                 im_shape: Tuple[int, int, int],
                 indices: np.ndarray,
                 processing_type: str) -> None:
        """
        Constructor of PcaMultiprocessingCapsule.

        Parameters
        ----------
        mean_out_port : pynpoint.core.dataio.OutputPort
            Output port for the mean residuals.
        median_out_port : pynpoint.core.dataio.OutputPort
            Output port for the median residuals.
        weighted_out_port : pynpoint.core.dataio.OutputPort
            Output port for the noise-weighted residuals.
        stim_out_port : pynpoint.core.dataio.OutputPort
            Output port for the stim residuals.
        clip_out_port : pynpoint.core.dataio.OutputPort
            Output port for the mean clipped residuals.
        num_proc : int
            Number of processors.
        pca_numbers : numpy.ndarray
            Number of principal components.
        pca_model : sklearn.decomposition.pca.PCA
            PCA object with the basis.
        star_reshape : numpy.ndarray
            Reshaped (2D) input images.
        angles : numpy.ndarray
            Derotation angles (deg).
        scales : numpy.ndarray
            Rescaling values
        im_shape : tuple(int, int, int)
            Original shape of the input images.
        indices : numpy.ndarray
            Non-masked pixel indices.

        Returns
        -------
        NoneType
            None
        """

        self.m_mean_out_port = mean_out_port
        self.m_median_out_port = median_out_port
        self.m_weighted_out_port = weighted_out_port
        self.m_stim_out_port = stim_out_port
        self.m_clip_out_port = clip_out_port
        self.m_pca_numbers = pca_numbers
        self.m_pca_model = pca_model
        self.m_star_reshape = star_reshape
        self.m_angles = angles
        self.m_scales = scales
        self.m_im_shape = im_shape
        self.m_indices = indices
        self.m_processing_type = processing_type

        self.m_requirements = [False, False, False, False, False]

        if self.m_mean_out_port is not None:
            self.m_requirements[0] = True

        if self.m_median_out_port is not None:
            self.m_requirements[1] = True

        if self.m_weighted_out_port is not None:
            self.m_requirements[2] = True

        if self.m_stim_out_port is not None:
            self.m_requirements[3] = True

        if self.m_clip_out_port is not None:
            self.m_requirements[4] = True

        self.m_requirements = tuple(self.m_requirements)

        super(PcaMultiprocessingCapsule, self).__init__(None, None, num_proc)

    @typechecked
    def create_writer(self,
                      image_out_port: None) -> PcaTaskWriter:
        """
        Method to create an instance of PcaTaskWriter.

        Parameters
        ----------
        image_out_port : None
            Output port, not used.

        Returns
        -------
        pynpoint.util.multipca.PcaTaskWriter
            PCA task writer.
        """

        return PcaTaskWriter(self.m_result_queue,
                             self.m_mean_out_port,
                             self.m_median_out_port,
                             self.m_weighted_out_port,
                             self.m_stim_out_port,
                             self.m_clip_out_port,
                             self.m_data_mutex,
                             self.m_requirements)

    @typechecked
    def init_creator(self,
                     image_in_port: None) -> PcaTaskCreator:
        """
        Method to create an instance of PcaTaskCreator.

        Parameters
        ----------
        image_in_port : None
            Input port, not used.

        Returns
        -------
        pynpoint.util.multipca.PcaTaskCreator
            PCA task creator.
        """

        return PcaTaskCreator(self.m_tasks_queue,
                              self.m_num_proc,
                              self.m_pca_numbers)

    @typechecked
    def create_processors(self) -> List[PcaTaskProcessor]:
        """
        Method to create a list of instances of PcaTaskProcessor.

        Returns
        -------
        list(pynpoint.util.multipca.PcaTaskProcessor, )
            PCA task processors.
        """

        processors = []

        for _ in range(self.m_num_proc):

            processors.append(PcaTaskProcessor(self.m_tasks_queue,
                                               self.m_result_queue,
                                               self.m_star_reshape,
                                               self.m_angles,
                                               self.m_scales,
                                               self.m_pca_model,
                                               self.m_im_shape,
                                               self.m_indices,
                                               self.m_requirements,
                                               self.m_processing_type))

        return processors
