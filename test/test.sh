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

cmd="python $(get_bin_dir)/../notebooks/entrypoint.py --init xavier $@"
echo $cmd
eval $cmd
