import os
from typing import Callable, List

import click


def sm_protocol(
    model: str = "model",
    output: str = "output",
    channels: List[str] = ["train", "test", "validation"],
    channel_prefix: str = "data",
) -> Callable:
    print("haha")

    def decorator(f):
        # Need to add options in reverse order than f's args.

        # CLI hyperparameters that belong to the wrapped function.
        # See https://click.palletsprojects.com/en/7.x/advanced/#forwarding-unknown-options
        opts = click.argument("train_args", nargs=-1, type=click.UNPROCESSED)(f)

        for channel in channels[::-1]:
            opts = click.option(
                f"--{channel}",
                default=os.environ.get("SM_CHANNEL_{channel.upper()}", os.path.join(channel_prefix, channel)),
                help=f"Where to read input channel {channel}",
            )(opts)

        opts = click.option(
            "--output-data-dir",
            default=os.environ.get("SM_OUTPUT_DATA_DIR", output),
            help="Where to output additional artifacts",
        )(opts)

        opts = click.option(
            "--model-dir", default=os.environ.get("SM_MODEL_DIR", model), help="Where to output model artifacts"
        )(opts)

        return click.command(context_settings={"ignore_unknown_options": True})(opts)

    return decorator
