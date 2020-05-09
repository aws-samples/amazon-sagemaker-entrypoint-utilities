"""Placeholder."""
# Make sure to import smepu before you import tqdm or any other module that
# uses tqdm.
#
# NOTE: depending on how you setup your favorite editor with formatter+isort,
# a few savings may be needed to bring smepu to the near top. For e.g., on
# vscode with black + isort enabled, it may take 2x saves.
#
# The 1st save may put smepu after tqdm (or tqdm-dependant modules), and it
# takes the 2nd (or possibly more) save to rearrange smepu to the top.
import smepu

# Must use the following form. This is intentional, to allow "import smepu" to
# work even without click installed.
from smepu.click import sm_protocol

from pydoc import locate
from typing import Any, Dict, Tuple

import click

# Setup logger must be done in the entrypoint script.
logger = smepu.setup_opinionated_logger(__name__)


@sm_protocol()
@click.option(
    "--algo", type=str, default="dummyest.DummyEstimator", help="Full name of estimator class that provides .fit()"
)
def main(train_args: Tuple[str, ...], **my_kwargs) -> None:
    """Train an estimator using hyperparameters specified from cli args.

    Args:
        train_args ([type]): [description]
        my_kwargs: arguments to this entrypoint script, parsed by click.
    """
    logger.info("SageMaker matters: %s", my_kwargs)
    logger.info("train_args: %s", train_args)

    # Convert cli args / hyperparameters to the estimator's kwargs
    kwargs: Dict[str, Any] = smepu.argparse.to_kwargs(train_args)
    logger.info("kwargs: %s", kwargs)

    # Estimator is an instance of "algo" class.
    klass: Any = locate(my_kwargs["algo"])
    estimator = klass(**kwargs)
    logger.info("%s", estimator)

    # Start training, which will show tqdm bar.
    estimator.fit()


if __name__ == "__main__":
    logger.info("Entrypoint script that uses click to digest hyperparameters.")
    main()
