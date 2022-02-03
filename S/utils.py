def train_test_split(image_files, anomaly_image_files, mask_files, test_split=0.2, cross_validation=False):
    train_test_sets = []
    if cross_validation:
        k = int(1 / test_split)
        print('Doing a {k}-fold cross-validation.'.format(k=k))
        for set_idx in range(k):
            sets = {}
            sets['train_images'] = []
            sets['train_anomaly_images'] = []
            sets['train_masks'] = []
            sets['test_images'] = []
            sets['test_anomaly_images'] = []
            sets['test_masks'] = []
            for i in range(len(image_files)):
                if i % k == set_idx:
                    sets['test_images'].append(image_files[i])
                    sets['test_anomaly_images'].append(anomaly_image_files[i])
                    sets['test_masks'].append(mask_files[i])
                else:
                    sets['train_images'].append(image_files[i])
                    sets['train_anomaly_images'].append(anomaly_image_files[i])
                    sets['train_masks'].append(mask_files[i])
            train_test_sets.append(sets)
    else:
        test_size = int(len(image_files) * test_split)
        print('Setting aside {n} images as the test set.'.format(n=test_size))
        sets = {}
        sets['test_images'] = image_files[:test_size]
        sets['test_anomaly_images'] = anomaly_image_files[:test_size]
        sets['test_masks'] = mask_files[:test_size]
        sets['train_images'] = image_files[test_size:]
        sets['train_anomaly_images'] = anomaly_image_files[test_size:]
        sets['train_masks'] = mask_files[test_size:]
        train_test_sets.append(sets)
    return train_test_sets
