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
# Math functionalities for tf-graphics.

# google internal package dependency 8)
# google internal package dependency 5

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "math",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    # google internal rule 1
    visibility = ["//visibility:public"],
    deps = [
        ":math_helpers",
        ":spherical_harmonics",
        ":vector",
        "//tensorflow_graphics/math/interpolation",
        "//tensorflow_graphics/math/optimizer",
        "//tensorflow_graphics/util:export_api",
    ],
)

py_library(
    name = "vector",
    srcs = ["vector.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "spherical_harmonics",
    srcs = ["spherical_harmonics.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":math_helpers",
        ":vector",
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "math_helpers",
    srcs = ["math_helpers.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:safe_ops",
        "//tensorflow_graphics/util:shape",
    ],
)

py_test(
    name = "spherical_harmonics_test",
    timeout = "long",
    srcs = ["tests/spherical_harmonics_test.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    # google internal rule 2
    # google internal rule 3
    # google internal rule 4
    # google internal rule 5
    # google internal rule 6
    deps = [
        ":math_helpers",
        ":spherical_harmonics",
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "math_helpers_test",
    srcs = ["tests/math_helpers_test.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    # google internal rule 2
    # google internal rule 3
    # google internal rule 4
    # google internal rule 5
    # google internal rule 6
    deps = [
        ":math_helpers",
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "vector_test",
    srcs = ["tests/vector_test.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    # google internal rule 2
    # google internal rule 3
    # google internal rule 4
    # google internal rule 5
    # google internal rule 6
    deps = [
        ":vector",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/geometry/transformation:test_data",
        "//tensorflow_graphics/util:test_case",
    ],
)
