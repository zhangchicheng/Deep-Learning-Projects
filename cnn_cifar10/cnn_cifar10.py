#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='./cifar10_cnn_model', help='output directory')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--train_steps', type=int, default=30000, help='number of training steps')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    # Convert the inputs to a Dataset.
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Batch the examples
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layers
    net = tf.layers.batch_normalization(inputs=features, training=mode==tf.estimator.ModeKeys.TRAIN)
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[5, 5], padding='valid',activation=tf.nn.relu)
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    net = tf.layers.dropout(inputs=net, rate=0.25, training=mode==tf.estimator.ModeKeys.TRAIN)

    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    net = tf.layers.dropout(inputs=net, rate=0.25, training=mode==tf.estimator.ModeKeys.TRAIN)

    net = tf.layers.flatten(inputs=net)
    net = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu)
    net = tf.layers.dropout(inputs=net, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)

    # Compute logits (1 per class).
    logits = tf.layers.dense(inputs=net, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(params.learning_rate)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
          train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )

def main(argv):
    args = parser.parse_args(argv[1:])

    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

    config = tf.estimator.RunConfig(
        model_dir=args.out_dir,
        save_summary_steps=100,
        log_step_count_steps=100,
        save_checkpoints_steps=500,
    )
    # Create the Estimator
    model_estimator = tf.estimator.Estimator(model_fn=cnn_model_fn, config=config, params=args)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),
        max_steps=args.train_steps)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda:eval_input_fn(test_x, test_y, args.batch_size),
        steps=None,
        start_delay_secs=60,  # Start evaluating after 10 sec.
        throttle_secs=30  # Evaluate only every 30 sec
    )
    tf.estimator.train_and_evaluate(model_estimator, train_spec, eval_spec)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
