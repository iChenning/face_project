# reference: https://github.com/aqeelanwar/MaskTheFace
import dlib
import os
import cv2
import glob
import random
import numpy as np
from s_data.MaskTheFace.utils.aux_functions import mask_image2


class AugmentMask(object):
    def __init__(self, root_='./s_data/MaskTheFace', mask_rate=0.3):
        self.root_ = root_
        self.mask_rate = mask_rate

        self.mask_types = ['N95', 'cloth', 'KN95', 'surgical', 'gas']
        self._get_pattern()

        self.dlib_detector = dlib.get_frontal_face_detector()
        self.dlib_predictor = dlib.shape_predictor(os.path.join(root_, 'dlib_models/shape_predictor_68_face_landmarks.dat'))

    def _get_pattern(self):
        pattern = []
        for dir_ in os.listdir(os.path.join(self.root_, 'masks/textures')):
            pattern.extend(glob.glob(os.path.join(self.root_, 'masks/textures', dir_) + '/*'))
        self.pattern = pattern

    def mask(self, sample):
        if random.random() <= self.mask_rate:
            try:
                sample_new = np.asarray(sample)
                sample_new = cv2.cvtColor(sample_new, cv2.COLOR_RGB2BGR)
                sample_new = mask_image2(self.dlib_detector, self.dlib_predictor, sample_new,
                                    mask_type=random.choice(self.mask_types), pattern=random.choice(self.pattern))
                sample_new = cv2.cvtColor(sample_new, cv2.COLOR_BGR2RGB)
                return sample_new
            except:
                return sample
        else:
            return sample