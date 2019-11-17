from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Dependency imports
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim
from research.domain_adaptation.datasets import mixed
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'The directory where the output TFRecords and temporary files are saved.')

# The names of the classes.
_CLASS_NAMES = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'size',
    'seven',
    'eight',
    'nine',
]


def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.

    Args:
      dataset_dir: The directory where the temporary files are stored.
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return '%s/mnist_mix_%s.tfrecord' % (dataset_dir, split_name)


def run(dataset_dir):
    pull_these_numbers(dataset_dir)

    return 0


def decode(serialized_example):
    """
    Parses an image and label from the given `serialized_example`.
    It is used as a map function for `dataset.map`
    """
    IMAGE_SIZE = 28
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
        'image/class/label': tf.FixedLenFeature(
            [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }

    # items_to_handlers = {
    #     'image': slim.tfexample_decoder.Image(shape=[32, 32, 3], channels=3),
    #     'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
    # }

    # decoder = slim.tfexample_decoder.TFExampleDecoder(
    #     keys_to_features, items_to_handlers)

    # 1. define a parser
    parsed_dataset = tf.io.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features=keys_to_features)
    #
    # 2. Convert the data
    # image = tf.io.decode_png(parsed_dataset['image/encoded'], channels=0, dtype=tf.uint8)
    # label = tf.cast(parsed_dataset['image/class/label'], tf.int32)

    return parsed_dataset


def pull_these_numbers(dataset_dir):
    tf.enable_eager_execution()

    filename = os.path.join(dataset_dir, 'mnist_m_train_1.tfrecord')
    labels1, labels2, labels3, labels4 = mixed.get_class_labels(5)
    file_list = mixed.get_file_list(dataset_dir, file_pattern=None, split_name='test', labels_mnist=labels1,
                                    labels_mnist_m=labels2)

    # dataset = tf.data.TFRecordDataset(filename)
    dataset = tf.data.TFRecordDataset(file_list)

    dataset = dataset.shuffle(8000)

    # for raw in dataset.take(1):
    #     print('Raw data from dataset: \n', repr(raw))

    dataset = dataset.map(decode)
    # for decoded in dataset.take(1):
    #     print("Decoded data from dataset: \n", repr(decoded))
    # print(repr(image))

    for decoded in dataset.take(10):
        image = tf.io.decode_png(decoded['image/encoded'], channels=0, dtype=tf.uint8)
        label = tf.cast(decoded['image/class/label'], tf.int32)
        if image.get_shape() == (28, 28, 1):
            image = tf.image.grayscale_to_rgb(image)
            image = tf.image.resize(image, [32, 32])
            print('mnist')

            print(image.get_shape())
        else:
            print('mnist_m')
        plt.imshow(image.numpy())
        plt.show()
        # print(decoded[0].numpy())
        print(label.numpy())


def main(_):
    assert FLAGS.dataset_dir

    # FLAGS.dataset_dir = '/home/runchi/thesis/datasets'
    run(FLAGS.dataset_dir)



if __name__ == '__main__':
    tf.app.run()
