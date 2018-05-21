# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of util functions for training and evaluating.
"""

import numpy
import tensorflow as tf
from tensorflow import logging

try:
  xrange          # Python 2
except NameError:
  xrange = range  # Python 3


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.

  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias


def MakeSummary(name, value):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  return summary


def AddGlobalStepSummary(summary_writer,
                         global_step_val,
                         global_step_info_dict,
                         summary_scope="Eval"):
  """Add the global_step summary to the Tensorboard.

  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    global_step_info_dict: a dictionary of the evaluation metrics calculated for
      a mini-batch.
    summary_scope: Train or Eval.

  Returns:
    A string of this global_step summary
  """
  this_hit_at_one = global_step_info_dict["hit_at_one"]
  this_hit_at_two = global_step_info_dict["hit_at_two"]
  this_hit_at_three = global_step_info_dict["hit_at_three"]
  this_hit_at_four = global_step_info_dict["hit_at_four"]
  this_hit_at_five = global_step_info_dict["hit_at_five"]
  this_hit_at_six = global_step_info_dict["hit_at_six"]
  this_hit_at_seven = global_step_info_dict["hit_at_seven"]
  this_hit_at_eight = global_step_info_dict["hit_at_eight"]
  this_hit_at_nine = global_step_info_dict["hit_at_nine"]
  this_hit_at_ten = global_step_info_dict["hit_at_ten"]
  this_perr = global_step_info_dict["perr"]
  this_loss = global_step_info_dict["loss"]
  examples_per_second = global_step_info_dict.get("examples_per_second", -1)

  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@1", this_hit_at_one),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@2", this_hit_at_two),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@3", this_hit_at_three),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@4", this_hit_at_four),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@5", this_hit_at_five),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@6", this_hit_at_six),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@7", this_hit_at_seven),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@8", this_hit_at_eight),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@9", this_hit_at_nine),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@10", this_hit_at_ten),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Perr", this_perr),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Loss", this_loss),
      global_step_val)

  if examples_per_second != -1:
    summary_writer.add_summary(
        MakeSummary("GlobalStep/" + summary_scope + "_Example_Second",
                    examples_per_second), global_step_val)

  summary_writer.flush()
  info = ("global_step {0} | Batch Hit@1: {1:.3f} | Batch Hit@2: {2:.3f} | Batch Hit@3: {3:.3f} | Batch Hit@4: {4:.3f} | Batch Hit@5: {5:.3f} | Batch Hit@6: {6:.3f} | Batch Hit@7: {7:.3f} | Batch Hit@8: {8:.3f} | Batch Hit@9: {9:.3f} | Batch Hit@10: {10:.3f} |  Batch PERR: {11:.3f} | Batch Loss: {12:.3f} "
          "| Examples_per_sec: {13:.3f}").format(
              global_step_val, this_hit_at_one, this_hit_at_two, this_hit_at_three, this_hit_at_four, this_hit_at_five, this_hit_at_six, this_hit_at_seven, this_hit_at_eight, this_hit_at_nine, this_hit_at_ten, this_perr, this_loss,
              examples_per_second)
  return info


def AddEpochSummary(summary_writer,
                    global_step_val,
                    epoch_info_dict,
                    summary_scope="Eval"):
  """Add the epoch summary to the Tensorboard.

  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    epoch_info_dict: a dictionary of the evaluation metrics calculated for the
      whole epoch.
    summary_scope: Train or Eval.

  Returns:
    A string of this global_step summary
  """
  epoch_id = epoch_info_dict["epoch_id"]
  avg_hit_at_one = epoch_info_dict["avg_hit_at_one"]
  avg_hit_at_two = epoch_info_dict["avg_hit_at_two"]
  avg_hit_at_three = epoch_info_dict["avg_hit_at_three"]
  avg_hit_at_four = epoch_info_dict["avg_hit_at_four"]
  avg_hit_at_five = epoch_info_dict["avg_hit_at_five"]
  avg_hit_at_six = epoch_info_dict["avg_hit_at_six"]
  avg_hit_at_seven = epoch_info_dict["avg_hit_at_seven"]
  avg_hit_at_eight = epoch_info_dict["avg_hit_at_eight"]
  avg_hit_at_nine = epoch_info_dict["avg_hit_at_nine"]
  avg_hit_at_ten = epoch_info_dict["avg_hit_at_ten"]
  avg_perr = epoch_info_dict["avg_perr"]
  avg_loss = epoch_info_dict["avg_loss"]
  aps = epoch_info_dict["aps"]
  gap = epoch_info_dict["gap"]
  mean_ap = numpy.mean(aps)

  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@1", avg_hit_at_one),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@2", avg_hit_at_two),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@3", avg_hit_at_three),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@4", avg_hit_at_four),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@5", avg_hit_at_five),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@6", avg_hit_at_six),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@7", avg_hit_at_seven),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@8", avg_hit_at_eight),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@9", avg_hit_at_nine),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@10", avg_hit_at_ten),
      global_step_val)    
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Perr", avg_perr),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Loss", avg_loss),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_MAP", mean_ap),
          global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_GAP", gap),
          global_step_val)
  summary_writer.flush()

  info = ("epoch/eval number {0} | Avg_Hit@1: {1:.3f} | Avg_Hit@2: {2:.3f} | Avg_Hit@3: {3:.3f} | Avg_Hit@4: {4:.3f} | Avg_Hit@5: {5:.3f} | Avg_Hit@6: {6:.3f} | Avg_Hit@7: {7:.3f} | Avg_Hit@8: {8:.3f} | Avg_Hit@9: {9:.3f} | Avg_Hit@10: {10:.3f} | Avg_PERR: {11:.3f} "
          "| MAP: {12:.3f} | GAP: {13:.3f} | Avg_Loss: {14:3f}").format(
          epoch_id, avg_hit_at_one, avg_hit_at_two, avg_hit_at_three, avg_hit_at_four, avg_hit_at_five, avg_hit_at_six, avg_hit_at_seven, avg_hit_at_eight, avg_hit_at_nine, avg_hit_at_ten, avg_perr, mean_ap, gap, avg_loss)
  return info

def GetListOfFeatureNamesAndSizes(feature_names, feature_sizes):
  """Extract the list of feature names and the dimensionality of each feature
     from string of comma separated values.

  Args:
    feature_names: string containing comma separated list of feature names
    feature_sizes: string containing comma separated list of feature sizes

  Returns:
    List of the feature names and list of the dimensionality of each feature.
    Elements in the first/second list are strings/integers.
  """
  list_of_feature_names = [
      feature_names.strip() for feature_names in feature_names.split(',')]
  list_of_feature_sizes = [
      int(feature_sizes) for feature_sizes in feature_sizes.split(',')]
  if len(list_of_feature_names) != len(list_of_feature_sizes):
    logging.error("length of the feature names (=" +
                  str(len(list_of_feature_names)) + ") != length of feature "
                  "sizes (=" + str(len(list_of_feature_sizes)) + ")")

  return list_of_feature_names, list_of_feature_sizes

def clip_gradient_norms(gradients_to_variables, max_norm):
  """Clips the gradients by the given value.

  Args:
    gradients_to_variables: A list of gradient to variable pairs (tuples).
    max_norm: the maximum norm value.

  Returns:
    A list of clipped gradient to variable pairs.
  """
  clipped_grads_and_vars = []
  for grad, var in gradients_to_variables:
    if grad is not None:
      if isinstance(grad, tf.IndexedSlices):
        tmp = tf.clip_by_norm(grad.values, max_norm)
        grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad = tf.clip_by_norm(grad, max_norm)
    clipped_grads_and_vars.append((grad, var))
  return clipped_grads_and_vars

def combine_gradients(tower_grads):
  """Calculate the combined gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been summed
     across all towers.
  """
  filtered_grads = [[x for x in grad_list if x[0] is not None] for grad_list in tower_grads]
  final_grads = []
  for i in xrange(len(filtered_grads[0])):
    grads = [filtered_grads[t][i] for t in xrange(len(filtered_grads))]
    grad = tf.stack([x[0] for x in grads], 0)
    grad = tf.reduce_sum(grad, 0)
    final_grads.append((grad, filtered_grads[0][i][1],))

  return final_grads
