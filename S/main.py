import argparse
parser = argparse.ArgumentParser()
parser.add_argument('image_dir', type=str, help='directory containing the images')
parser.add_argument('anomaly_image_dir', type=str, help='directory containing the anomaly images')
parser.add_argument('mask_dir', type=str, help='directory containing the segmentation masks')
parser.add_argument('out_dir', type=str, help='output directory')
parser.add_argument('--ext', type=str, default='.nii.gz',
                    help='filename extension (default: .nii.gz)')
parser.add_argument('--test_split', type=float, default=0.2,
                    help='fraction of data to be used as test set (default: 0.2)')
parser.add_argument('--cross_validation', action='store_true',
                    help='do a k-fold cross-validation (k will be 1/test_split)')
parser.add_argument('--batch_size', type=int, default=1,
                    help='number of samples per gradient update (default: 1)')
parser.add_argument('--base_filters', type=int, default=16,
                    help='number of filters in the full-resolution conv layers (default: 16)')
parser.add_argument('--learning_rate', type=float, default=0.0005,
                    help='learning rate (default: 0.0005)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train the model (default: 50)')
parser.add_argument('--initial_epoch', type=int, default=0,
                    help='epoch at which to start training (for resuming a previous training run) (default: 0)')
parser.add_argument('--mixed_precision', action='store_true',
                    help='set mixed precision policy')
args = parser.parse_args()

# Import packages and modules.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import glob
import sys
import os
import time
import pandas as pd
import seaborn as sns
import utils # See utils.py
import generator # See generator.py
import model # See model.py
import metrics # See metrics.py
import losses # See losses.py

## Set mixed precision policy.

if args.mixed_precision:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Set mixed precision policy.')

## Load and preprocess data.

### 1. Load all the filenames into a list.

image_files = sorted(glob.glob(args.image_dir + '/*' + args.ext))
anomaly_image_files = sorted(glob.glob(args.anomaly_image_dir + '/*' + args.ext))
mask_files = sorted(glob.glob(args.mask_dir + '/*' + args.ext))

if len(image_files) != len(mask_files):
    sys.exit('Error: Number of images and number of masks do not match.')

print('Number of images:', len(image_files))

### 2. Split the dataset.

train_test_sets = utils.train_test_split(image_files, anomaly_image_files, mask_files, args.test_split, args.cross_validation)

### 3. Create generators from the filenames.

datasets = []
for sets in train_test_sets:
    ds = {}
    ds['train_ds'] = generator.Sequence((sets['train_images'], sets['train_anomaly_images']), sets['train_masks'], args.batch_size)
    ds['test_ds'] = generator.Sequence((sets['test_images'], sets['test_anomaly_images']), sets['test_masks'], args.batch_size, shuffle=False)
    datasets.append(ds)

### 4. Check image dimensions (sanity check).

image_batch, mask_batch = next(iter(datasets[0]['train_ds']))
print('Image shape:', image_batch[0].shape)
print('Mask shape:', mask_batch[0].shape)

## Build and train the model.

output_channels = mask_batch.shape[-1] - 1

if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

for set_idx in range(len(datasets)):
    if args.cross_validation:
        print('CV set', set_idx)
    
    train_ds = datasets[set_idx]['train_ds']
    print('Size of training set:', len(train_ds.x))
    
    # Create a new instance of the model.
    unet = model.UNET(output_channels, args.base_filters)
    
    if args.initial_epoch != 0:
        # Load the weights from the last checkpoint.
        checkpoint_dir = os.path.join(args.out_dir, 'training_{idx}'.format(idx=set_idx))
        checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
        unet.load_weights(checkpoint_path.format(epoch=args.initial_epoch))
        print('Resuming training from epoch', args.initial_epoch)

    # Compile the model.
    unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                 loss=losses.S_loss,
                 metrics=[metrics.multiclass_dice(i) for i in range(output_channels)])
    
    # Create a callback that saves the model's weights every epoch.
    checkpoint_dir = os.path.join(args.out_dir, 'training_{idx}'.format(idx=set_idx))
    checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    
    # Train the model.
    start = time.time()
    history = unet.fit(train_ds,
                       epochs=args.epochs, initial_epoch=args.initial_epoch,
                       callbacks=[cp_callback])
    print('Training took {t} seconds.'.format(t=time.time() - start))

## Make predictions.

if args.test_split == 0:
    sys.exit()

pred_dir = os.path.join(args.out_dir, 'pred_masks')
if not os.path.isdir(pred_dir):
    os.makedirs(pred_dir)
print('Saving predictions to', pred_dir)

t = 0.0
for set_idx in range(len(datasets)):
    if args.cross_validation:
        print('CV set', set_idx)
    
    test_ds = datasets[set_idx]['test_ds']
    print('Size of test set:', len(test_ds.x))
    
    # Load the weights from the last checkpoint.
    checkpoint_dir = os.path.join(args.out_dir, 'training_{idx}'.format(idx=set_idx))
    checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
    unet.load_weights(checkpoint_path.format(epoch=args.epochs))
    
    for step, (image_batch, mask_batch) in enumerate(test_ds):
        start = time.time()
        predictions = unet.predict(image_batch)
        t += time.time() - start

        for i in range(args.batch_size):
            data_idx = step * args.batch_size + i
            print('Progress: {idx}/{n}'.format(idx=data_idx, n=len(test_ds.x)), end='\r')

            pred_mask = np.argmax(predictions[i], axis=-1)
            img = nib.Nifti1Image(pred_mask.astype(np.uint8), affine=nib.load(test_ds.x[data_idx]).affine,
                                  header=nib.load(test_ds.x[data_idx]).header)
            nib.save(img, os.path.join(pred_dir, os.path.basename(test_ds.y[data_idx])))

    print('Progress: {n}/{n}'.format(n=len(test_ds.x)))

n = len(image_files) if args.cross_validation else len(test_ds.x)
print('Inference time per image was {t} seconds.'.format(t=t/n))

## Evaluate predictions.

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

pred_masks = sorted(glob.glob(pred_dir + '/*' + args.ext))
true_masks = [os.path.join(args.mask_dir, os.path.basename(fn)) for fn in pred_masks]
classes = ['BG', 'FB', 'FC', 'TB', 'TC']

eval_file = os.path.join(args.out_dir, 'eval.txt')
print('Saving results to', eval_file)

df = []
for i in range(len(pred_masks)):
    print('Progress: {i}/{n}'.format(i=i, n=len(pred_masks)), end='\r')
    
    y_true = test_ds.load_mask(true_masks[i])
    y_pred = test_ds.load_mask(pred_masks[i])
    sampling = nib.load(pred_masks[i]).header.get_zooms()

    for channel in range(output_channels):
        dsc, voe, vd = metrics.volume_metrics(y_true, y_pred, channel)
        asd, rsd, msd = metrics.surface_distance(y_true, y_pred, channel, sampling)

        entry = {}
        entry['Case'] = os.path.basename(pred_masks[i]).split('_')[0]
        entry['Label'] = classes[channel]
        entry['DSC'] = dsc
        entry['VOE'] = voe
        entry['VD'] = vd
        entry['ASD'] = asd
        entry['RSD'] = rsd
        entry['MSD'] = msd
        df.append(entry)

print('Progress: {n}/{n}'.format(n=len(pred_masks)))

df = pd.DataFrame(df)
df.to_csv(eval_file, sep='\t', columns=['Case', 'Label', 'DSC', 'VOE', 'VD', 'ASD', 'RSD', 'MSD'])

with open(os.path.join(args.out_dir, 'eval_summary.txt'), 'w') as file:
    dsc_summary = df['DSC'].groupby(df['Label'])
    file.write('DSC summary:\n{}\n'.format(dsc_summary.describe()))
    voe_summary = df['VOE'].groupby(df['Label'])
    file.write('\nVOE summary:\n{}\n'.format(voe_summary.describe()))
    vd_summary = df['VD'].groupby(df['Label'])
    file.write('\nVD summary:\n{}\n'.format(vd_summary.describe()))
    asd_summary = df['ASD'].groupby(df['Label'])
    file.write('\nASD summary:\n{}\n'.format(asd_summary.describe()))
    rsd_summary = df['RSD'].groupby(df['Label'])
    file.write('\nRSD summary:\n{}\n'.format(rsd_summary.describe()))
    msd_summary = df['MSD'].groupby(df['Label'])
    file.write('\nMSD summary:\n{}\n'.format(msd_summary.describe()))

def save_boxplot(df, metric):
    boxplot = plt.figure()
    sns.boxenplot(data=df, x='Label', y=metric, order=classes)
    boxplot.savefig(os.path.join(args.out_dir, 'boxplot_' + metric + '.png'))

save_boxplot(df, 'DSC')
save_boxplot(df, 'VOE')
save_boxplot(df, 'VD')
save_boxplot(df, 'ASD')
save_boxplot(df, 'RSD')
save_boxplot(df, 'MSD')
