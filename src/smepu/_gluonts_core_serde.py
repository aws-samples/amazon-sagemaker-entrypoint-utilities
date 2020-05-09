# The content of this file is taken from https://github.com/awslabs/gluon-ts,
# specifically from src/gluonts/core/serde.py whose license is shown below:
#
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Placeholder."""
from pydoc import locate
from typing import Any

kind_type = "type"
kind_inst = "instance"


def decode(r: Any) -> Any:
    """Decode a value from an intermediate representation `r`.

    Parameters
    ----------
    r
        An intermediate representation to be decoded.

    Returns
    -------
    Any
        A Python data structure corresponding to the decoded version of ``r``.

    See Also
    --------
    encode
        Inverse function.
    """
    # structural recursion over the possible shapes of r
    # r = { 'class': ..., 'args': ... }
    # r = { 'class': ..., 'kwargs': ... }
    if type(r) == dict and r.get("__kind__") == kind_inst:
        cls = locate(r["class"])
        args = decode(r["args"]) if "args" in r else []
        kwargs = decode(r["kwargs"]) if "kwargs" in r else {}
        return cls(*args, **kwargs)  # type: ignore
    # r = { 'class': ..., 'args': ... }
    # r = { 'class': ..., 'kwargs': ... }
    if type(r) == dict and r.get("__kind__") == kind_type:
        return locate(r["class"])
    # r = { k1: v1, ..., kn: vn }
    elif type(r) == dict:
        return {k: decode(v) for k, v in r.items()}
    # r = ( y1, ..., yn )
    elif type(r) == tuple:
        return tuple([decode(y) for y in r])
    # r = [ y1, ..., yn ]
    elif type(r) == list:
        return [decode(y) for y in r]
    # r = { y1, ..., yn }
    elif type(r) == set:
        return {decode(y) for y in r}
    # r = a
    else:
        return r
