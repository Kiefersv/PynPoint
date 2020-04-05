# -*- coding: utf-8 -*-
"""
Various function to support IFS reduction

@author: Sven Kiefer
"""

import sys
import time

import numpy as np

from typing import Tuple, Union

from typeguard import typechecked
from scipy.ndimage import rotate

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress, memory_frames
from pynpoint.util.image import scale_image







class IfsScalingModule (ProcessingModule):
    """
    Pipeline module for rescaling images in given direction
    
    """
    __author__ = 'Sven Kiefer'

    @typechecked
    def __init__(self,
                 name_in: str, 
                 image_in_tag: str, 
                 image_out_tag: str,
                 scaling: Union[Tuple[float, float, float],
                                Tuple[None, None, float],
                                Tuple[float, float, None]] = (1.0,1.0,1.0), 
                 angle: float = 0, 
                 pixscale: bool = False) -> None:
        
        """
            This module allows to correct for distortion effects caused by the instrument
            
            Parameters
            ----------
            image_in_tag : str
                Tag of the database entries that are read as input and then 
                split in lambdas.
            image_out_tag : str
                Tag of the database entry that is written as output after recombined 
                in all lambdas.
            scaling : tupel(float, float, float)
                Scaling values for x, y and flux direction. This value should correspond 
                to the distortion
                of the instrument in angle direction.
            angle : float
                The angle of rotation of the distortion
            pixscale
                Adjust the pixel scale by the average scaling in x and y direction
                
            Returns
            -------
            NoneType
                None
            
        """
    
        super(IfsScalingModule, self).__init__(name_in)
    
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
    
        if scaling[0] is not None: self.m_x_scaling = scaling[0]
        else: self.m_x_scaling = 1.0
        
        if scaling[1] is not None: self.m_y_scaling = scaling[1]
        else: self.m_y_scaling = 1.0
        
        if scaling[2] is not None: self.m_f_scaling = scaling[2]
        else: self.m_f_scaling = 1.0
        
        self.m_angle = angle
        self.m_pixscale = pixscale
    
    
    
    def run(self) -> None:
        """
        Run method of the module. Rotates all images by a constant angle, sclaes them
        and rotates them back.

        Returns
        -------
        NoneType
            None
        """

        # Collecting information and preparing the database
        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        
        memory = self._m_config_port.get_attribute('MEMORY')
        nimages = self.m_image_in_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        start_time = time.time()

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), 'Running IfsScalingModule...', start_time)

            images = self.m_image_in_port[frames[i]:frames[i+1], ]

            for j in range(frames[i+1]-frames[i]):
                im_tmp = images[j, ]

                # ndimage.rotate first rotates in clockwise direction for positive angles
                im_tmp = rotate(im_tmp, self.m_angle, reshape=False)
                
                # Scaling of image
                im_tmp = self.m_f_scaling * scale_image(im_tmp, 
                                                        self.m_y_scaling, 
                                                        self.m_y_scaling)
                
                # ndimage.rotate first rotates in clockwise direction for positive angles
                im_tmp = rotate(im_tmp, -self.m_angle, reshape=False)
                

                self.m_image_out_port.append(im_tmp, data_dim=3)
                
        # Set approximate new scaling
        if self.m_pixscale:
            mean_scaling = (self.m_x_scaling+self.m_y_scaling)/2.
            self.m_image_out_port.add_attribute('PIXSCALE', pixscale/mean_scaling)
            
        
        # Appling the changes to the database
        sys.stdout.write('Running IfsScalingModule... [DONE]\n')
        sys.stdout.flush()

        history = f'scaling = ({self.m_x_scaling:.2f}, {self.m_y_scaling:.2f}, ' \
                  f'{self.m_f_scaling:.2f})'
        self.m_image_out_port.add_history('IfsScalingModule', history)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()
        
        
        
        
    
    
class CenteringReductionModule(ProcessingModule):
    """
    Pipeline module for enhanced Centering frame reduction using science images
    
    """
    __author__ = 'Sven Kiefer'

    @typechecked
    def __init__(self,
                 name_in: str, 
                 image_in_tag: str, 
                 center_in_tag: str, 
                 center_out_tag: str) -> None:
            
        """
            This Module
        
            Parameters
            ----------
            image_in_tag : str
                Tag of the database entries that are read as input and 
                then split in lambdas.
            center_in_tag : str
                Tag of the center images which should be reduced
            center_out_tag : str
                Output tag of the reduced center frames
                
            Returns
            -------
            NoneType
                None
                
        """
        
        super(CenteringReductionModule, self).__init__(name_in)
    
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_center_in_port = self.add_output_port(center_in_tag)
        self.m_center_out_port = self.add_output_port(center_out_tag)
    
    
    def run(self) -> None:
        """
        Run method of the module. Rotates all images by a constant angle, sclaes them
        and rotates them back.

        Returns
        -------
        NoneType
            None
        """

        # Collecting information and preparing the database
        self.m_center_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute('MEMORY')
        nimages = self.m_center_in_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        master = self.m_image_in_port.get_all()[0]

        start_time = time.time()

        for i in range(len(frames[:-1])):
            progress(i, len(frames[:-1]), 'Running CenteringReduction...', start_time)

            images = self.m_center_in_port[frames[i]:frames[i+1], ]

            self.m_center_out_port.append(images - master, data_dim=3)

        sys.stdout.write('Running CenteringReduction... [DONE]\n')
        sys.stdout.flush()

        history = f'dark_in_tag = {self.m_image_in_port.tag}'
        self.m_center_out_port.add_history('CenteringReduction', history)
        self.m_center_out_port.copy_attributes(self.m_center_in_port)
        self.m_center_out_port.close_port()
        
        
   
    



class FrameClipModule(ProcessingModule):
    """
    Pipeline to do simple spectral charakterisation
    """

    __author__ = 'Sven Kiefer'

    @typechecked
    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 clip_length: int = 1,
                 clip_mode: str = 'end') -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the science images that are read as input.
        image_out_tag : str
            Tag of the database entry with the science images that are writen as output.
        clip_length : int
            Number of return pictures (might not be used depending on clip_mode)
        clip_mode : str
            How to select frames. Possible modes: end, start
        

        Returns
        -------
        NoneType
            None
        """

        super(FrameClipModule, self).__init__(name_in)
        

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        
        self.m_clip_length = clip_length
        self.m_clip_mode = clip_mode



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
        
        
        sys.stdout.write("Running FrameClipModule ....")
        sys.stdout.flush()
        
        #Prepare for read in
        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()
        
        #gather data
        data_in = self.m_image_in_port.get_all()
        attt_in = self.m_image_in_port.get_all_non_static_attributes()
        attt_st = self.m_image_in_port.get_all_static_attributes()
        
        #Prepare lambda
        lam = self.m_image_in_port.get_attribute('LAMBDA')
        
        
        #Do clipping data
        if self.m_clip_mode == 'end':
            data_cut = data_in[-self.m_clip_length:,]
        
        elif self.m_clip_mode == 'start':
            data_cut = data_in[:self.m_clip_length,]
            
        else:
            raise ValueError('Wrong key word clip_mode')
            
        
        self.m_image_out_port.append(data_cut, data_dim=3)
            
        
        
        #Do clipping attributes
        if self.m_clip_mode == 'end':
            attt_in = self.m_image_in_port.get_all_non_static_attributes()
            
            for at in attt_in:
                da = self.m_image_in_port.get_attribute(at)[-self.m_clip_length:]
                self.m_image_out_port.add_attribute(at, da, static = False)
                
            self.m_image_out_port.del_attribute('LAMBDA')
            self.m_image_out_port.add_attribute('LAMBDA', np.array([lam[-self.m_clip_length:]]), static=False)   
        
        
        elif self.m_clip_mode == 'start':
            attt_in = self.m_image_in_port.get_all_non_static_attributes()
            
            for at in attt_in:
                da = self.m_image_in_port.get_attribute(at)[:self.m_clip_length]
                self.m_image_out_port.add_attribute(at, da, static = False)
                
            self.m_image_out_port.del_attribute('LAMBDA')
            self.m_image_out_port.add_attribute('LAMBDA', np.array([lam[:self.m_clip_length]]), static=False)   
            
        
        else:
            raise ValueError('Wrong key word clip_mode')
            
        
        for st in attt_st:
            dat = self.m_image_in_port.get_attribute(st)
            self.m_image_out_port.add_attribute(st, dat, static = True)
        
        
        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()



