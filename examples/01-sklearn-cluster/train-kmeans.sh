#!/usr/bin/env bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# This script must be executed from the same directory as the train.py.

declare -a ARGS=(
    --model-dir /tmp/kmeans/model
    --output-data-dir /tmp/kmeans/output/data/algo-1
    --train refdata
    --algo sklearn.cluster.KMeans
    --n_clusters 3
)

python train.py "${ARGS[@]}"
