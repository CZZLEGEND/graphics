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
"""Tests for linear blend skinning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import linear_blend_skinning
from tensorflow_graphics.geometry.transformation.tests import test_helpers
from tensorflow_graphics.util import test_case


class LinearBlendSkinningTest(test_case.TestCase):

  # pyformat: disable
  @parameterized.parameters(
      ((3,), (7,), (7, 4, 4),),
      ((None, 3), (None, 9), (None, 9, 4, 4),),
      ((7, 1, 3), (1, 4, 11), (5, 11, 4, 4),),
      ((7, 4, 3), (4, 11), (11, 4, 4),),
      ((3,), (5, 4, 11), (11, 4, 4),),
  )
  # pyformat: enable
  def test_lbs_transform_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(linear_blend_skinning.lbs_transform,
                                        shapes)

  # pyformat: disable
  @parameterized.parameters(
      ("point must have exactly 3 dimensions in axis -1",
       (None,), (7,), (7, 4, 4)),
      ("transforms must have a rank greater than 2", (3,), (7,), (4, 4)),
      ("transforms must have exactly 4 dimensions in axis -1",
       (3,), (7,), (7, 4, None)),
      ("transforms must have exactly 4 dimensions in axis -2",
       (3,), (7,), (7, None, 4)),
      (r"Tensors \[\'weights\', \'transforms\'\] must have the same number of dimensions in axes",
       (3,), (9,), (7, 4, 4)),
      ("Not all batch dimensions are broadcast-compatible",
       (2, 3, 3), (3, 1, 7), (7, 4, 4)),
  )
  # pyformat: enable
  def test_lbs_transform_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(linear_blend_skinning.lbs_transform,
                                    error_msg, shapes)

  # pyformat: disable
  @parameterized.parameters(
      ((3,), (3,), (7,), (7, 4, 4),),
      ((5, 7, 4, 3), (7, 1, 3), (1, 4, 11), (5, 11, 4, 4),),
      ((7, 4, 3), (7, 4, 3), (4, 11), (11, 4, 4),),
      ((5, 4, 3), (3,), (5, 4, 11), (11, 4, 4),),
      ((6, 5, 8, 4, 3), (8, 1, 3), (8, 4, 11), (6, 5, 11, 4, 4),),
  )
  # pyformat: enable
  def test_lbs_transform_output_shape_is_shape(self, expected_output_shape,
                                               *input_shapes):
    """Tests that the output tensor of lbs_transform has the correct shape."""
    x_point_init = np.random.uniform(size=input_shapes[0])
    x_point = tf.convert_to_tensor(value=x_point_init)
    x_weights_init = np.random.uniform(size=input_shapes[1])
    x_weights = tf.convert_to_tensor(value=x_weights_init)
    x_transforms_init = np.random.uniform(size=input_shapes[2])
    x_transforms = tf.convert_to_tensor(value=x_transforms_init)

    y = linear_blend_skinning.lbs_transform(x_point, x_weights, x_transforms)
    self.assertAllEqual(tf.shape(input=y), expected_output_shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_lbs_transform_jacobian_random(self):
    """Test the Jacobian of the lbs_transform function."""
    x_point_init, x_weights_init, x_transforms_init = test_helpers.generate_random_test_lbs_transform()  # pylint: disable=line-too-long  # pyformat: disable
    x_point = tf.convert_to_tensor(value=x_point_init)
    x_weights = tf.convert_to_tensor(value=x_weights_init)
    x_transforms = tf.convert_to_tensor(value=x_transforms_init)

    y = linear_blend_skinning.lbs_transform(x_point, x_weights, x_transforms)

    self.assert_jacobian_is_correct(x_point, x_point_init, y)
    self.assert_jacobian_is_correct(x_weights, x_weights_init, y)
    self.assert_jacobian_is_correct(x_transforms, x_transforms_init, y)


if __name__ == "__main__":
  test_case.main()
