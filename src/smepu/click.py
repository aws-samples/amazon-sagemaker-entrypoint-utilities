"""Placeholder."""
import os
from pathlib import Path
from typing import Callable, List

import click


def sm_protocol(
    model: str = "model",
    output: str = "output",
    channels: List[str] = ["train", "test", "validation"],
    channel_prefix: str = "data",
) -> Callable:
    """Create an arg parser that implements minimum SageMaker entrypoint protocol.

    Only model, output, and channel dirs are implemented, as this is typically bare minimum to run or test an
    entrypoint script locally, e.g., `python ./entrypoint.py`.

    See https://github.com/aws/sagemaker-containers/blob/master/README.rst#important-environment-variables.

    This function must be imported using as follows:

    >>> from smepu import click as smclick
    >>> @smclick.sm_protocol()
    >>> ...

    or

    >>> from smepu.click import sm_protocol
    >>> @sm_protocol()
    >>> ...

    This is intentionally done to allow smepu package to still importable even without click installed.

    Args:
        model (str, optional): Model dir when not running on SageMaker. Defaults to "model".
        output (str, optional): Output dir when not running on SageMaker. Defaults to "output".
        channels (List[str], optional): Data channels. Defaults to ["train", "test", "validation"].
        channel_prefix (str, optional): Parent directory that contains the channel dirs. Defaults to "data".

    Returns:
        Callable: the decoratee.
    """

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
                type=Path,
            )(opts)

        opts = click.option(
            "--output-data-dir",
            default=os.environ.get("SM_OUTPUT_DATA_DIR", output),
            help="Where to output additional artifacts",
            type=Path,
        )(opts)

        opts = click.option(
            "--model-dir",
            default=os.environ.get("SM_MODEL_DIR", model),
            help="Where to output model artifacts",
            type=Path,
        )(opts)

        return click.command(context_settings={"ignore_unknown_options": True})(opts)

    return decorator
