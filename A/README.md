# Part 2: Anomaly detection using the original images

## Modules
* utils.py -- creates training/test sets
* generator.py -- data generator
* model.py -- builds the model
* losses.py -- defines loss function

## Usage
1. To run the script with default arguments, run:
```
python main.py image_dir target_image_dir out_dir
```
2. There are some optional arguments, e.g.
```
python main.py image_dir target_image_dir out_dir --learnng_rate 5e-4 --epochs 50
```
3. To see all the arguments available, run:
```
python main.py --help
```
