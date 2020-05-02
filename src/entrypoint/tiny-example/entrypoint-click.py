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

# Must use the following form. This is intentional, to allow smepu to work even without click installed.
from smepu.click import sm_protocol

from typing import Any, Dict, Tuple

from dummyest import DummyEstimator

# Setup logger must be done in the entrypoint script.
logger = smepu.setup_opinionated_logger(__name__)


@sm_protocol()
def main(train_args: Tuple[str, ...], **my_kwargs) -> None:
    """Train an estimator using hyperparameters specified from cli args.

    Args:
        train_args ([type]): [description]
        my_kwargs: arguments to this entrypoint script, parsed by click.
    """
    logger.info("SageMaker matters: %s", my_kwargs)
    logger.info("train_args: %s", train_args)

    # Convert cli args / hyperparameters to the estimator's kwargs
    kwargs: Dict[str, Any] = smepu.argparse.parse_for_func(train_args)
    logger.info("kwargs: %s", kwargs)

    # Invoke a callable using the kwargs.
    estimator = DummyEstimator(**kwargs)
    logger.info("%s", estimator)

    # Start training, which will show tqdm bar.
    estimator.fit()


if __name__ == "__main__":
    main()
