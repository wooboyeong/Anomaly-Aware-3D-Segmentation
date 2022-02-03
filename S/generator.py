import tensorflow as tf
import numpy as np
import nibabel as nib
import math
import random

# Here, `x_set` is list of path to the images
# and `y_set` is list of path to the segmentation masks.

class Sequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size=1, shuffle=True):
        self.xs, self.y = x_set, y_set
        self.x, self.xa = self.xs # images, anomaly images
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
        batch_masks = []
        for idx in batch_data_idxs:
            image = self.load_image(self.x[idx])
            anomaly_image = self.load_anomaly_image(self.xa[idx])
            mask = self.load_mask(self.y[idx])
            image = np.concatenate((image, anomaly_image), axis=-1)
            mask = np.concatenate((mask, anomaly_image), axis=-1)
            batch_images.append(image)
            batch_masks.append(mask)
        return np.array(batch_images), np.array(batch_masks)
    
    def load_image(self, filename):
        image = nib.load(filename).get_fdata().astype(np.float32)
        image = np.squeeze(image)
        image = (image - image.mean()) / image.std()
        return image[..., np.newaxis]

    def load_anomaly_image(self, filename):
        anomaly_image = nib.load(filename).get_fdata().astype(np.float32)
        anomaly_image = np.squeeze(anomaly_image)
        return anomaly_image[..., np.newaxis]

    def load_mask(self, filename):
        mask = nib.load(filename).get_fdata().astype(np.uint8)
        mask = np.squeeze(mask)
        return np.eye(np.unique(mask).size)[mask]
