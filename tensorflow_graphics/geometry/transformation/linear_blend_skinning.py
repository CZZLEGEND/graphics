#Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module implements TensorFlow linear blend skinning functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def lbs_transform(point, weights, transforms, name=None):
  """Transform point using Linear Blend Skinning.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible and allow transforming full 3D shapes at once.
    In the following, B1 to Bm are optional batch dimensions, which allow
    transforming multiple poses at once.

  Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point.
    weights: A tensor of shape `[A1, ..., An, W]`, where the last dimension
      represents the weights of each transform.
    transforms: A tensor of shape `[B1, ..., Bm, W, 4, 4]`, which represents the
      4x4 transforms atached to the W handles.
    name: A name for this op that defaults to "lbs_transform".

  Returns:
    A tensor of shape `[B1, ..., Bm, A1, ..., An, 3]`, where the last dimension
    represents
    a 3d point.

  Raises:
    ValueError: If the shape of the input tensors are not supported.
  """
  with tf.compat.v1.name_scope(name, "lbs_transform",
                               [point, weights, transforms]):
    point = tf.convert_to_tensor(value=point)
    weights = tf.convert_to_tensor(value=weights)
    transforms = tf.convert_to_tensor(value=transforms)

    shape.check_static(
        tensor=point, tensor_name="point", has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=transforms,
        tensor_name="transforms",
        has_rank_greater_than=2,
        has_dim_equals=((-2, 4), (-1, 4)))
    shape.compare_dimensions(
        tensors=(weights, transforms),
        tensor_names=("weights", "transforms"),
        axes=(-1, -3))
    shape.compare_batch_dimensions(
        tensors=(point, weights),
        tensor_names=("point", "weights"),
        last_axes=(-2, -2),
        broadcast_compatible=True)

    num_weights = weights.shape[-1]

    def dim_value(dim):
      return 1 if dim is None else tf.compat.v1.dimension_value(dim)

    common_batch_shape = shape.get_broadcasted_shape(point.shape[:-1],
                                                     weights.shape[:-1])
    common_batch_shape = [dim_value(dim) for dim in common_batch_shape]

    point = tf.broadcast_to(point, common_batch_shape + [3])
    weights = tf.broadcast_to(weights, common_batch_shape + [num_weights])

    hom_point = tf.concat(
        [point, tf.ones(common_batch_shape + [1], dtype=point.dtype)], axis=-1)
    transformed_point = tf.tensordot(hom_point, transforms, [[-1], [-1]])

    point_batch_dims = point.shape.ndims - 1
    transforms_batch_dims = transforms.shape.ndims - 3
    total_dims = transformed_point.shape.ndims
    assert total_dims == point_batch_dims + transforms_batch_dims + 2
    permutation = list(
        range(
            point_batch_dims, point_batch_dims + transforms_batch_dims)) + list(
                range(0, point_batch_dims)) + [total_dims - 2, total_dims - 1]

    transformed_point = tf.transpose(a=transformed_point, perm=permutation)

    weights = tf.expand_dims(weights, axis=-1)
    weighted_point = tf.multiply(weights, transformed_point)

    interpolated_point = tf.reduce_sum(input_tensor=weighted_point, axis=-2)
    interpolated_point, _ = tf.split(interpolated_point, [3, 1], axis=-1)

    return interpolated_point


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
