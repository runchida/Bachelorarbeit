# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

# pylint: disable=line-too-long
"""Evaluation for Domain Separation Networks (DSNs)."""
# pylint: enable=line-too-long
import math

import numpy as np
from six.moves import xrange
import tensorflow as tf

from research.domain_adaptation.datasets import dataset_factory, mixed
from research.domain_adaptation.domain_separation import losses
from research.domain_adaptation.domain_separation import models

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            'The number of images in each batch.')

tf.app.flags.DEFINE_string('master', '',
                           'BNS name of the TensorFlow master to use.')

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/da/',
                           'Directory where the model was written to.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/da/',
    'Directory where we should write the tf summaries to.')

tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string('dataset', 'mnist_m',
                           'Which dataset to test on: "mnist", "mnist_m".')

tf.app.flags.DEFINE_string('split', 'valid',
                           'Which portion to test on: "valid", "test".')

tf.app.flags.DEFINE_integer('num_examples', 1000, 'Number of test examples.')

tf.app.flags.DEFINE_string('basic_tower', 'dann_mnist',
                           'The basic tower building block.')

tf.app.flags.DEFINE_bool('enable_precision_recall', False,
                         'If True, precision and recall for each class will '
                         'be added to the metrics.')

tf.app.flags.DEFINE_bool('use_logging', False, 'Debugging messages.')

tf.app.flags.DEFINE_string(
    'training_name', None, 'Name of the training scenario')

tf.app.flags.DEFINE_string(
    'checkpoint', None, 'checkpoint number to be evaluated')

def quaternion_metric(predictions, labels):
    params = {'batch_size': FLAGS.batch_size, 'use_logging': False}
    logcost = losses.log_quaternion_loss_batch(predictions, labels, params)
    return slim.metrics.streaming_mean(logcost)


def angle_diff(true_q, pred_q):
    angles = 2 * (
            180.0 /
            np.pi) * np.arccos(np.abs(np.sum(np.multiply(pred_q, true_q), axis=1)))
    return angles


def provide_batch_fn():
    """ The provide_batch function to use. """
    return dataset_factory.provide_batch


def main(_):
    g = tf.Graph()
    with g.as_default():
        # Load the data.
        FLAGS.checkpoint_dir, retrain = mixed.get_train_log_dir(FLAGS.training_name)
        if FLAGS.dataset == 'mixed':
            labels = mixed.read_mixed_labels(FLAGS.checkpoint_dir)
            target_mnist_labels = labels['labels3']
            target_mnist_m_labels = labels['labels4']
            images, labels = provide_batch_fn()(
                FLAGS.dataset, FLAGS.split, FLAGS.dataset_dir, 4, FLAGS.batch_size, 4, labels_one=target_mnist_labels,
                    labels_two=target_mnist_m_labels)
        else:
            images, labels = provide_batch_fn()(
                FLAGS.dataset, FLAGS.split, FLAGS.dataset_dir, 4, FLAGS.batch_size, 4)

        num_classes = labels['classes'].get_shape().as_list()[1]

        tf.summary.image('eval_images', images, max_outputs=10)

        # Define the model:
        with tf.variable_scope('towers'):
            basic_tower = getattr(models, FLAGS.basic_tower)
            predictions, endpoints = basic_tower(
                images,
                num_classes=num_classes,
                is_training=False,
                batch_norm_params=None)
        metric_names_to_values = {}

        # Define the metrics:
        if 'quaternions' in labels:  # Also have to evaluate pose estimation!
            quaternion_loss = quaternion_metric(labels['quaternions'],
                                                endpoints['quaternion_pred'])

            angle_errors, = tf.py_func(
                angle_diff, [labels['quaternions'], endpoints['quaternion_pred']],
                [tf.float32])

            metric_names_to_values[
                'Angular mean error'] = slim.metrics.streaming_mean(angle_errors)
            metric_names_to_values['Quaternion Loss'] = quaternion_loss

        accuracy = tf.contrib.metrics.streaming_accuracy(
            tf.argmax(predictions, 1), tf.argmax(labels['classes'], 1))

        predictions = tf.argmax(predictions, 1)
        labels = tf.argmax(labels['classes'], 1)
        metric_names_to_values['Accuracy'] = accuracy

        if FLAGS.enable_precision_recall:
            for i in xrange(num_classes):
                index_map = tf.one_hot(i, depth=num_classes)
                name = 'PR/Precision_{}'.format(i)
                metric_names_to_values[name] = slim.metrics.streaming_precision(
                    tf.gather(index_map, predictions), tf.gather(index_map, labels))
                name = 'PR/Recall_{}'.format(i)
                metric_names_to_values[name] = slim.metrics.streaming_recall(
                    tf.gather(index_map, predictions), tf.gather(index_map, labels))

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
            metric_names_to_values)

        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        metric_names_list = list(names_to_values.keys())
        metric_values_list = list(names_to_values.values())

        for metric_name, metric_value in names_to_values.items():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        # This ensures that we make a single pass over all of the data.
        num_batches = math.ceil(FLAGS.num_examples / float(FLAGS.batch_size))

        # Setup the global step.
        slim.get_or_create_global_step()

        if FLAGS.checkpoint is None:
            slim.evaluation.evaluation_loop(
                FLAGS.master,
                checkpoint_dir=FLAGS.checkpoint_dir,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
                summary_op=tf.summary.merge(summary_ops),
                eval_op=list(names_to_updates.values())
            )

        else:
            checkpoint_path = FLAGS.checkpoint_dir + 'model.ckpt-' + FLAGS.checkpoint
            slim.evaluation.evaluate_once(
                FLAGS.master,
                checkpoint_path=checkpoint_path,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
                summary_op=tf.summary.merge(summary_ops),
                eval_op=list(names_to_updates.values())
            )




if __name__ == '__main__':
    # FLAGS.dataset = 'mnist_m'
    # FLAGS.split = 'test'
    # FLAGS.num_examples = 9001
    # FLAGS.dataset_dir = '/home/runchi/thesis/datasets'

    tf.app.run()
