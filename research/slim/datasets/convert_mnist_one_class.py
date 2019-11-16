# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts MNIST data to TFRecords of TF-Example protos.

This module downloads the MNIST data, uncompresses it, reads the files
that make up the MNIST data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import numpy as np
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_dir',
    '/home/runchi/thesis/datasets',
    'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_integer('class_to_convert', None, 'Class to be converted this round')

# The URLs where the MNIST data can be downloaded.
_DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
_TRAIN_DATA_FILENAME = 'train-images-idx3-ubyte.gz'
_TRAIN_LABELS_FILENAME = 'train-labels-idx1-ubyte.gz'
_TEST_DATA_FILENAME = 't10k-images-idx3-ubyte.gz'
_TEST_LABELS_FILENAME = 't10k-labels-idx1-ubyte.gz'

_IMAGE_SIZE = 28
_NUM_CHANNELS = 1

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


def _extract_images(filename, num_images):
    """Extract the images into a numpy array.

    Args:
      filename: The path to an MNIST images file.
      num_images: The number of images in the file.

    Returns:
      A numpy array of shape [number_of_images, height, width, channels].
    """
    print('Extracting images from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            _IMAGE_SIZE * _IMAGE_SIZE * num_images * _NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
    return data


def _extract_labels(filename, num_labels):
    """Extract the labels into a vector of int64 label IDs.

    Args:
      filename: The path to an MNIST labels file.
      num_labels: The number of labels in the file.

    Returns:
      A numpy array of shape [number_of_labels]
    """
    print('Extracting labels from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def _add_to_tfrecord(data_filename, labels_filename, num_images,
                     tfrecord_writer, class_to_convert):
    """Loads data from the binary MNIST files and writes files to a TFRecord.

    Args:
      data_filename: The filename of the MNIST images.
      labels_filename: The filename of the MNIST labels.
      num_images: The number of images in the dataset.
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    images = _extract_images(data_filename, num_images)
    labels = _extract_labels(labels_filename, num_images)

    shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_png(image)

        with tf.Session('') as sess:
            for j in range(num_images):
                #sys.stdout.flush()
                print(labels[j])
                if labels[j] == class_to_convert:
                    print('\r>> Converting image %d/%d' % (j + 1, num_images))
                    png_string = sess.run(encoded_png, feed_dict={image: images[j]})

                    example = dataset_utils.image_to_tfexample(
                        png_string, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[j])
                    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(dataset_dir, split_name, class_to_convert):
    """Creates the output filename.

    Args:
      dataset_dir: The directory where the temporary files are stored.
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return '%s/mnist_%s_%s.tfrecord' % (dataset_dir, split_name, str(class_to_convert))



def run(dataset_dir, class_to_convert):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    training_filename = _get_output_filename(dataset_dir, 'train', class_to_convert)
    testing_filename = _get_output_filename(dataset_dir, 'test', class_to_convert)
    validating_filename = _get_output_filename(dataset_dir, 'valid', class_to_convert)

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # First, process the training data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        data_filename = os.path.join(dataset_dir, 'mnist', _TRAIN_DATA_FILENAME)
        labels_filename = os.path.join(dataset_dir, 'mnist', _TRAIN_LABELS_FILENAME)
        _add_to_tfrecord(data_filename, labels_filename, 60000, tfrecord_writer, class_to_convert)

    # Next, process the testing data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        data_filename = os.path.join(dataset_dir, 'mnist', _TEST_DATA_FILENAME)
        labels_filename = os.path.join(dataset_dir, 'mnist', _TEST_LABELS_FILENAME)
        _add_to_tfrecord(data_filename, labels_filename, 10000, tfrecord_writer, class_to_convert)

    # with tf.python_io.TFRecordWriter(validating_filename) as tfrecord_writer:
    #     data_filename = os.path.join(dataset_dir, 'mnist', _TRAIN_DATA_FILENAME)
    #     labels_filename = os.path.join(dataset_dir, 'mnist', _TRAIN_LABELS_FILENAME)
    #     _add_to_tfrecord(data_filename, labels_filename, 500, tfrecord_writer, class_to_convert)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MNIST dataset!')


def check_label(dataset_dir, filename, num_images, class_to_convert):
    labels_filename = os.path.join(dataset_dir, 'mnist', filename)
    labels = _extract_labels(labels_filename, num_images)
    for j in range(num_images):
        #if labels[j] == class_to_convert:
            print(labels[j])
    print('finished set')

def main(_):
    # FLAGS.dataset_dir = '/home/runchi/thesis/datasets'
    # FLAGS.class_to_convert = 9
    run(FLAGS.dataset_dir, FLAGS.class_to_convert)

    #check_label(FLAGS.dataset_dir, _TRAIN_LABELS_FILENAME, 59500)
    #check_label(FLAGS.dataset_dir, _TRAIN_LABELS_FILENAME, 500, 0)

if __name__ == '__main__':
    tf.app.run()
