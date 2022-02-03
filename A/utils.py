def train_test_split(image_files, target_image_files, test_split=0.2, cross_validation=False):
    train_test_sets = []
    if cross_validation:
        k = int(1 / test_split)
        print('Doing a {k}-fold cross-validation.'.format(k=k))
        for set_idx in range(k):
            sets = {}
            sets['train_images'] = []
            sets['train_target_images'] = []
            sets['test_images'] = []
            sets['test_target_images'] = []
            for i in range(len(image_files)):
                if i % k == set_idx:
                    sets['test_images'].append(image_files[i])
                    sets['test_target_images'].append(target_image_files[i])
                else:
                    sets['train_images'].append(image_files[i])
                    sets['train_target_images'].append(target_image_files[i])
            train_test_sets.append(sets)
    else:
        test_size = int(len(image_files) * test_split)
        print('Setting aside {n} images as the test set.'.format(n=test_size))
        sets = {}
        sets['test_images'] = image_files[:test_size]
        sets['test_target_images'] = target_image_files[:test_size]
        sets['train_images'] = image_files[test_size:]
        sets['train_target_images'] = target_image_files[test_size:]
        train_test_sets.append(sets)
    return train_test_sets
