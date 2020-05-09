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

echo "################################################################"
echo "# To run click version, run ${BASH_SOURCE[0]} -click"
echo "# To see the effect of notqdm, set env. var SM_HOST"
echo "################################################################"

export PYTHONPATH=$(get_bin_dir)/../examples:$PYTHONPATH

# The complex_args instructs the entrypoint script to instante this estimator:
#
#     DummyEstimator(epochs=2, callbacks=smepu.list(DummyCallback("EarlyStopper"), DummyCallback("Checkpointer")))
#
# Note that smepu.list is just a wrapper to Python list, however smepu.list converts all its positional arguments into
# a single list, in contrast to Python list that strictly takes just one argument.
#
# If your list is a valid JSON, then you can simply use --callbacks '[1, 2, "abcd"]'.
# However, when list members can be custom objects such as the following example, then use smepu.list.
declare -a complex_args=(
    --algo "dummyest.DummyEstimator"
    --epochs "2"
    --callbacks.__class__ "smepu.list"
    --callbacks.0.__class__ "dummyest.DummyCallback"
    --callbacks.0.name "EarlyStopper"
    --callbacks.1.__class__ "dummyest.DummyCallback"
    --callbacks.1.0 "Checkpointer"
)
cmd="python $(get_bin_dir)/../examples/entrypoint${1}.py ${complex_args[@]}"
echo $cmd
eval $cmd
