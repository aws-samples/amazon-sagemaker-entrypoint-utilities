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

from pydoc import locate
from typing import Any, Dict, List

# Setup logger must be done in the entrypoint script.
logger = smepu.setup_opinionated_logger(__name__)


def main(cfg: Dict[str, Any], train_args: List[str]) -> None:
    """Run the main function of this script."""
    logger.info("Entrypoint script that uses argparse to digest hyperparameters.")
    logger.info("cfg: %s", cfg)
    logger.info("train_args: %s", train_args)

    # Convert cli args / hyperparameters to kwargs
    kwargs: Dict[str, Any] = smepu.argparse.kwargs(train_args)

    # Estimator is an instance of "algo" class.
    klass: Any = locate(cfg["algo"])
    estimator = klass(**kwargs)
    logger.info("%s", estimator)

    # Start training, which will show tqdm bar.
    estimator.fit()


if __name__ == "__main__":
    # Minimal arg parsers for SageMaker protocol.
    parser = smepu.argparse.sm_protocol()
    parser.add_argument(
        "--algo", type=str, help="Full name of estimator class that provides .fit()", default="dummyest.DummyEstimator"
    )
    args, train_args = parser.parse_known_args()

    # Demonstrate SageMaker checks.
    if not smepu.is_on_sagemaker():
        # When dev/testing script locally, it's convenient to auto-create these dirs.
        logger.info("Create model & output dirs prior to underlying function.")
        smepu.mkdir(args.model_dir)
        smepu.mkdir(args.output_data_dir)

    main(vars(args), train_args)
