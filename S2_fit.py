#! /usr/bin/env python

from __future__ import print_function

import sys
import os
num_models = int(sys.argv[1])

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from time import time
from multiprocessing.dummy import Pool


t1 = time()

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

#print(type(train_dataset), train_dataset.nbytes, train_dataset[:10000].nbytes)
#sys.exit()


print("Init took {} seconds".format(time()-t1), num_models)
t1 = time()


train_subset = 10000
#train_subset = -1


optimizers = {}
losses = {}
train_preds = {}
valid_preds = {}
test_preds = {}


tf_train_dataset = {}
tf_train_labels = {}
tf_valid_dataset = {}
tf_test_dataset = {}

num_gpus = 4
gpus = {"gpu{}".format(x):"/gpu:{}".format(x) for x in xrange(num_gpus)}
if True:
#with tf.device('/gpu:0'):
#with tf.device('/cpu:0'):
    graph = tf.Graph()
    with graph.as_default():

      # Input data.
      # Load the training, validation and test data into constants that are
      # attached to the graph.
      for gpulabel, gpu in gpus.items():
          with tf.device(gpu):
              tf_train_dataset[gpulabel] = tf.constant(train_dataset[:train_subset, :], 
                                             name = "ds_train")
              print(tf_train_dataset[gpulabel].dtype)

              tf_train_labels[gpulabel] = tf.constant(train_labels[:train_subset],
                                            name = "labels_train")
              print(tf_train_labels[gpulabel].dtype)

              tf_valid_dataset[gpulabel] = tf.constant(valid_dataset,
                                             name = "ds_valid")
              print(tf_valid_dataset[gpulabel].dtype)

              tf_test_dataset[gpulabel] = tf.constant(test_dataset,
                                             name = "ds_test")
              print(tf_test_dataset[gpulabel].dtype)

      for imodel in xrange(num_models):
         gpulabel = gpus.keys()[imodel % num_gpus]
         gpu = gpus[gpulabel]
         model_name = "model_{}".format(imodel)
         with tf.device(gpu):
             with tf.variable_scope(model_name):
                  weights = tf.Variable(
                    tf.truncated_normal([image_size * image_size, num_labels]))
                  biases = tf.Variable(tf.zeros([num_labels]))
                  
                  logits = tf.matmul(tf_train_dataset[gpulabel], weights) + biases

                  ##
                  loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels[gpulabel]))
                  losses[model_name] = loss
                  
                  ##  
                  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
                  optimizers[model_name] = optimizer
                  
                  ##  
                  train_prediction = tf.nn.softmax(logits)
                  train_preds[model_name] = train_prediction

                  ##
                  valid_prediction = tf.nn.softmax(
                    tf.matmul(tf_valid_dataset[gpulabel], weights) + biases)

                  valid_preds[model_name] = valid_prediction

                  ##   
                  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset[gpulabel], weights) + biases)
                  test_preds[model_name] = test_prediction

print("Graph creation took {} seconds".format(time()-t1), num_models)
t1 = time()


num_steps = 801

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto#L135
config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#config.log_device_placement=True
#config.intra_op_parallelism_threads=80
#config.inter_op_parallelism_threads=80
with tf.Session(graph=graph, config=config) as session:
  print("Session creation took {} seconds".format(time()-t1), num_models)
  t1 = time()

  tf.initialize_all_variables().run()

  print("Variables initialization took {} seconds".format(time()-t1), num_models)
  t1 = time()

  todo_run = {}  
  for name_, dict_ in {"optimizers" : optimizers, 
                       #"losses": losses,
                       #"train_preds": train_preds
                       }.items():
    for k,v in dict_.items():
        todo_run[name_ + ":" + k] = v

  #  p = Pool(120)
  for step in range(num_steps):
    results = session.run(todo_run)
    #p.map(lambda x: session.run(x), todo_run.values())
    '''
    if (step % 100 == 0):
      for k,v in results.items():
          if "loss" in k:
              print('%s - Loss at step %d: %f' % (k, step, v))

          if "train_preds" in k:
              acc =  accuracy(v, train_labels[:train_subset, :])
              print('%s - Training accuracy: %.1f%%' % (k, acc))

      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      #print('Validation accuracy: %.1f%%' % accuracy(
      #  valid_prediction.eval(), valid_labels))
    '''


  #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  #for model_name, test_prediction in test_preds.items():
  #    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

print("Training took {}".format(time()-t1), num_models)
#t1 = time()
