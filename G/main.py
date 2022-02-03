import argparse
parser = argparse.ArgumentParser()
parser.add_argument('image_dir', type=str, help='directory containing the images')
parser.add_argument('mask_dir', type=str, help='directory containing the segmentation masks')
parser.add_argument('out_dir', type=str, help='output directory')
parser.add_argument('--ext', type=str, default='.nii.gz',
                    help='filename extension (default: .nii.gz)')
parser.add_argument('--batch_size', type=int, default=1,
                    help='number of samples per gradient update (default: 1)')
parser.add_argument('--base_filters', type=int, default=8,
                    help='number of filters in the full-resolution conv layers (default: 8)')
parser.add_argument('--learning_rate', type=float, default=0.0005,
                    help='learning rate (default: 0.0005)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train the model (default: 50)')
parser.add_argument('--initial_epoch', type=int, default=0,
                    help='epoch at which to start training (for resuming a previous training run) (default: 0)')
parser.add_argument('--mixed_precision', action='store_true',
                    help='set mixed precision policy')
args = parser.parse_args()

## Import packages and modules.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import glob
import sys
import os
import time
import generator # See generator.py
import model # See model.py

## Set mixed precision policy.

if args.mixed_precision:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Set mixed precision policy.')

## Load and preprocess data.

### 1. Load all the filenames into a list.

image_files = sorted(glob.glob(args.image_dir + '/*' + args.ext))
mask_files = sorted(glob.glob(args.mask_dir + '/*' + args.ext))

if len(image_files) != len(mask_files):
    sys.exit('Error: Number of images and number of masks do not match.')

print('Number of images:', len(image_files))

### 2. Create generators from the filenames.

train_ds = generator.Sequence(image_files, mask_files, args.batch_size)
test_ds = generator.Sequence(image_files, mask_files, args.batch_size, shuffle=False)

### 3. Check image dimensions (sanity check).

input_batch, output_batch = next(iter(train_ds))
masked_image_batch, image_batch = input_batch
print('Image shape:', image_batch[0].shape)
print('Masked Image shape:', masked_image_batch[0].shape)

## Build and train the model.

if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

print('Size of training set:', len(train_ds.x))

# Create a new instance of the model.
G = model.G(args.base_filters)

if args.initial_epoch != 0:
    # Load the weights from the last checkpoint.
    checkpoint_dir = os.path.join(args.out_dir, 'training')
    checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
    G.load_weights(checkpoint_path.format(epoch=args.initial_epoch))
    print('Resuming training from epoch', args.initial_epoch)

# Compile the model.
G.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
          loss='mse')

# Create a callback that saves the model's weights every epoch.
checkpoint_dir = os.path.join(args.out_dir, 'training')
checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model.
start = time.time()
history = G.fit(train_ds,
                epochs=args.epochs, initial_epoch=args.initial_epoch,
                callbacks=[cp_callback])
print('Training took {t} seconds.'.format(t=time.time() - start))

G.C.save_weights(os.path.join(checkpoint_dir, 'C.ckpt'))

## Make predictions.

recon_dir = os.path.join(args.out_dir, 'recon')
error_dir = os.path.join(args.out_dir, 'error')
if not os.path.isdir(recon_dir):
    os.makedirs(recon_dir)
if not os.path.isdir(error_dir):
    os.makedirs(error_dir)
print('Saving reconstructions to', recon_dir)
print('Saving error images to', error_dir)

t = 0.0

# Load the weights from the last checkpoint.
checkpoint_dir = os.path.join(args.out_dir, 'training')
checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
G.load_weights(checkpoint_path.format(epoch=args.epochs))

for step, (input_batch, output_batch) in enumerate(test_ds):
    start = time.time()
    reconstructed_images = G.predict(input_batch)
    t += time.time() - start

    error_images = tf.square(output_batch - reconstructed_images)

    for i in range(args.batch_size):
        data_idx = step * args.batch_size + i
        print('Progress: {idx}/{n}'.format(idx=data_idx, n=len(test_ds.x)), end='\r')

        img = nib.Nifti1Image(reconstructed_images[i, ..., 0], affine=nib.load(test_ds.x[data_idx]).affine,
                              header=nib.load(test_ds.x[data_idx]).header)
        nib.save(img, os.path.join(recon_dir, os.path.basename(test_ds.x[data_idx])))

        img = nib.Nifti1Image(error_images[i, ..., 0], affine=nib.load(test_ds.x[data_idx]).affine,
                              header=nib.load(test_ds.x[data_idx]).header)
        nib.save(img, os.path.join(error_dir, os.path.basename(test_ds.x[data_idx])))

print('Progress: {n}/{n}'.format(n=len(test_ds.x)))

n = len(image_files)
print('Inference time per image was {t} seconds.'.format(t=t/n))
