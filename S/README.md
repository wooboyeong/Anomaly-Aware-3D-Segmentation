# Part 3: Downstream task: Anomaly-aware segmentation

## Modules
* utils.py -- creates training/test sets
* generator.py -- data generator
* model.py -- builds the model
* losses.py -- defines loss function
* metrics.py -- defines evaluation metrics

## Usage
1. To run the script with default arguments, run:
```
python main.py image_dir anomaly_image_dir mask_dir out_dir
```
2. There are some optional arguments, e.g.
```
python main.py image_dir anomaly_image_dir mask_dir out_dir --learnng_rate 5e-4 --epochs 50
```
3. To see all the arguments available, run:
```
python main.py --help
```