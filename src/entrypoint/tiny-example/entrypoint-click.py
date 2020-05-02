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
from smepu.click import sm_protocol

from typing import Any, Dict

from dummyest import DummyEstimator

# Setup logger must be done in the entrypoint script.
logger = smepu.setup_opinionated_logger(__name__)


@sm_protocol()
def main(model_dir: str, output_data_dir: str, train: str, test: str, validation: str, train_args) -> None:
    logger.info("model, output_dir, train, test, validation: %s", [model_dir, output_data_dir, train, test, validation])
    logger.info("train_args: %s", train_args)

    # Convert cli args / hyperparameters to kwargs
    kwargs: Dict[str, Any] = smepu.argparse.parse_for_func(train_args)
    logger.info("kwargs: %s", kwargs)

    # Invoke a callable using the kwargs.
    estimator = DummyEstimator(**kwargs)
    logger.info("%s", estimator)

    # Start training, which will show tqdm bar.
    estimator.fit()


if __name__ == "__main__":
    main()
