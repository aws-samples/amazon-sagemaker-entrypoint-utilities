#!/bin/bash

# Utility function to get script's directory (deal with Mac OSX quirkiness).
get_bin_dir() {
    local READLINK=readlink
    if [[ $(uname) == 'Darwin' ]]; then
        READLINK=greadlink
        if [ $(which greadlink) == '' ]; then
            echo '[ERROR] Mac OSX requires greadlink. Install with "brew install greadlink"' >&2
            exit 1
        fi
    fi

    local BIN_DIR=$(dirname "$($READLINK -f ${BASH_SOURCE[0]})")
    echo -n ${BIN_DIR}
}

echo "#####################################################"
echo "# To see the effect of notqdm, set env. var SM_HOST #"
echo "#####################################################"

# callbacks=
declare -a complex_args=(
    "--callbacks.__class__"
    "smepu.list"
    "--callbacks.0.__class__"
    "dummyest.DummyCallback"
    "--callbacks.0.name"
    "EarlyStopper"
    "--callbacks.1.__class__"
    "dummyest.DummyCallback"
    "--callbacks.1.0"
    "Checkpointer"
)
cmd="python $(get_bin_dir)/../src/entrypoint/tiny-example/entrypoint.py --epochs 2 ${complex_args[@]}"
echo $cmd
eval $cmd
