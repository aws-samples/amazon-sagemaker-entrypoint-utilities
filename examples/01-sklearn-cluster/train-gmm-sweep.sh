#!/usr/bin/env bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# This script must be executed from the same directory as the train.py.

declare -a ARGS=(
    --model-dir /tmp/gmm/model
    --output-data-dir /tmp/gmm/output/data/algo-1
    --train refdata
    --algo sklearn.mixture.GaussianMixture
    --sweep 1
    --sweep-start 2
    --sweep-end 4
    # Additional gmm kwargs (except n_components) can be added below.
)

python train.py "${ARGS[@]}"
