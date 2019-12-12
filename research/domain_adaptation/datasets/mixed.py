from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Dependency imports
import tensorflow as tf

import numpy as np
from research.slim.datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN_ONE_CLASS = {'mnist': 'mnist_%s_%s.tfrecord',
                           'mnist_m': 'mnist_m_%s_%s.tfrecord'}

_SPLITS_TO_SIZES = {'train': 58001, 'valid': 1000, 'test': 9001}

_NUM_CLASSES = 10

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [28 x 28 x 1] grayscale image.',
    'label': 'A single integer between 0 and 9',
}

_ITEMS_TO_DESCRIPTIONS_MNIST_M = {
    'image': 'A [32 x 32 x 1] RGB image.',
    'label': 'A single integer between 0 and 9',
}


def get_split(split_name, dataset_dir, labels_one, labels_two, file_pattern=None):
    # get tf.data.Dataset
    # tf.enable_eager_execution()
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    file_list_mnist, file_list_mnist_m = get_file_list(dataset_dir, file_pattern, split_name, labels_one, labels_two)

    dataset_mnist = tf.data.TFRecordDataset(file_list_mnist)
    dataset_mnist_m = tf.data.TFRecordDataset(file_list_mnist_m)
    dataset_mnist = dataset_mnist.map(decode_gray)
    dataset_mnist_m = dataset_mnist_m.map(decode_rgb)

    # tf.data.Dataset
    return dataset_mnist, dataset_mnist_m


# def get_split(split_name, dataset_dir, labels_one, labels_two, file_pattern=None, reader=None):
#     """Gets a dataset tuple with instructions for reading MNIST.
#
#     Args:
#       split_name: A train/test split name.
#       dataset_dir: The base directory of the dataset sources.
#       file_pattern: The file pattern to use when matching the dataset sources.
#         It is assumed that the pattern contains a '%s' string so that the split
#         name can be inserted.
#       reader: The TensorFlow reader type.
#
#     Returns:
#       A `Dataset` namedtuple.
#
#     Raises:
#       ValueError: if `split_name` is not a valid train/test split.
#     """
#     if split_name not in _SPLITS_TO_SIZES:
#         raise ValueError('split name %s was not recognized.' % split_name)
#
#     if not file_pattern:
#         file_list = get_file_list(dataset_dir, file_pattern, split_name, labels_one, labels_two)
#
#     # Allowing None in the signature so that dataset_factory can use the default.
#     if reader is None:
#         reader = tf.TFRecordReader
#
#     keys_to_features = {
#         'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
#         'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
#         'image/class/label': tf.FixedLenFeature(
#             [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
#     }
#
#     items_to_handlers = {
#         'image': slim.tfexample_decoder.Image(shape=[28, 28, 1], channels=1),
#         'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
#     }
#
#     decoder = slim.tfexample_decoder.TFExampleDecoder(
#         keys_to_features, items_to_handlers)
#
#     labels_to_names = None
#     if dataset_utils.has_labels(dataset_dir):
#         labels_to_names = dataset_utils.read_label_file(dataset_dir)
#
#     return slim.dataset.Dataset(
#         data_sources=file_list,
#         reader=reader,
#         decoder=decoder,
#         num_samples=_SPLITS_TO_SIZES[split_name],
#         num_classes=_NUM_CLASSES,
#         items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
#         labels_to_names=labels_to_names)


def decode_gray(serialized_example):
    """
    Parses an image and label from the given `serialized_example`.
    It is used as a map function for `dataset.map`
    """

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
        'image/class/label': tf.FixedLenFeature(
            [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }

    # 1. define a parser
    parsed_dataset = tf.io.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features=keys_to_features)
    # #
    # 2. Convert the data
    image = tf.io.decode_png(parsed_dataset['image/encoded'], channels=0, dtype=tf.uint8)
    label = tf.cast(parsed_dataset['image/class/label'], tf.uint8)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize_images(image, [32, 32])
    label = tf.reshape(label, shape=[])

    return image, label


def decode_rgb(serialized_example):
    """
    Parses an image and label from the given `serialized_example`.
    It is used as a map function for `dataset.map`
    """

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
        'image/class/label': tf.FixedLenFeature(
            [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }

    # 1. define a parser
    parsed_dataset = tf.io.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features=keys_to_features)
    # #
    # 2. Convert the data
    image = tf.io.decode_png(parsed_dataset['image/encoded'], channels=0, dtype=tf.uint8)
    label = tf.cast(parsed_dataset['image/class/label'], tf.uint8)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [32, 32])
    image = tf.reshape(image, [32, 32, 3])
    label = tf.reshape(label, shape=[])

    return image, label


def get_class_labels(num_classes):
    # get lists of labels for source and target domains
    # labels one
    if num_classes < 5:
        raise ValueError('Number of classes must be between 5 to 10')

    class_labels_source_one = []
    fill_list(class_labels_source_one, num_classes)

    # labels two
    class_labels_source_two = []
    class_labels_test_one = []
    for x in range(0, 10):
        append = not (has_number(class_labels_source_one, x))
        if append:
            class_labels_source_two.append(x)
            class_labels_test_one.append(x)
    fill_list(class_labels_source_two, num_classes)

    class_labels_test_two = []
    for x in range(0, 10):
        append = not (has_number(class_labels_source_two, x))
        if append:
            class_labels_test_two.append(x)

    labels = {
        'labels1': class_labels_source_one,
        'labels2': class_labels_source_two,
        'labels3': class_labels_test_one,
        'labels4': class_labels_test_two
    }

    return labels


def fill_list(list, num_classes):
    # fill a list with numbers until it has 'num_classes' members
    while len(list) < num_classes:
        randnum = np.random.randint(0, 10)
        append = not (has_number(list, randnum))
        if append:
            list.append(randnum)
    return list


def has_number(list, num):
    # check if a list contains number num
    check = False
    for member in list:
        if member == num:
            check = True
    return check


def get_file_list(dataset_dir, file_pattern, split_name, labels_mnist, labels_mnist_m):
    # create list of files lead to mnist with labels_one and mnist-m with labels_two
    # lists of paths to separated tfrecord for mixed-domain case
    file_list_mnist = []
    file_list_mnist_m = []

    if not file_pattern:
        for class_label in labels_mnist:
            file_pattern = _FILE_PATTERN_ONE_CLASS['mnist']
            file_pattern = os.path.join(dataset_dir, 'mix', file_pattern % (split_name, str(class_label)))
            file_list_mnist.append(file_pattern)

        for class_label in labels_mnist_m:
            file_pattern = _FILE_PATTERN_ONE_CLASS['mnist_m']
            file_pattern = os.path.join(dataset_dir, 'mix', file_pattern % (split_name, str(class_label)))
            file_list_mnist_m.append(file_pattern)

        return file_list_mnist, file_list_mnist_m

    else:
        return file_pattern


def get_train_log_dir(training_name):
    # experiment_dir = os.path.join('/home', 'runchi', 'thesis', 'graphs', 'experiment')
    experiment_dir = os.path.join('/home', 'rk64vona', 'thesis', 'graphs', 'experiment')
    train_log_dir = os.path.join(experiment_dir, '%s/' % training_name)
    if not tf.gfile.Exists(train_log_dir):
        tf.gfile.MkDir(train_log_dir)
        retrain = False
    else:
        retrain = True
    return train_log_dir, retrain


def write_label(train_log_dir, labels):
    dataset_utils.write_mixed_labels(train_log_dir, labels)


def read_mixed_labels(train_log_dir):
    return dataset_utils.read_mixed_labels(train_log_dir)
