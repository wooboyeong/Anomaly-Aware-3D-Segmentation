import tensorflow as tf
import numpy as np
import nibabel as nib
import math
import random

# Here, `x_set` is list of path to the images
# and `y_set` is list of path to the target images.

class Sequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size=1, shuffle=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_idxs = list(range(len(self.x)))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data_idxs)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, step):
        batch_data_idxs = self.data_idxs[step * self.batch_size:(step + 1) * self.batch_size]

        batch_images = []
        batch_target_images = []
        for idx in batch_data_idxs:
            image = self.load_image(self.x[idx])
            target_image = self.load_target_image(self.y[idx])
            target_image = np.concatenate((target_image, image), axis=-1)
            batch_images.append(image)
            batch_target_images.append(target_image)
        return np.array(batch_images), np.array(batch_target_images)
    
    def load_image(self, filename):
        image = nib.load(filename).get_fdata().astype(np.float32)
        image = np.squeeze(image)
        image = (image - image.mean()) / image.std()
        image = np.clip(image, -5.0, 5.0)
        image = (image - image.min()) / (image.max() - image.min())
        return image[..., np.newaxis]

    def load_target_image(self, filename):
        target_image = nib.load(filename).get_fdata().astype(np.float32)
        target_image = np.squeeze(target_image)
        return target_image[..., np.newaxis]
