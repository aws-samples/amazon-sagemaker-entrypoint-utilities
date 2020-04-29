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

from typing import Any, Dict, List

from dummyest import DummyEstimator

# Setup logger must be done in the entrypoint script.
logger = smepu.setup_opinionated_logger(__name__)


def main(cfg: Dict[str, Any], train_args: List[str]) -> None:
    logger.info("cfg: %s", cfg)
    logger.info("train_args: %s", train_args)

    # Convert cli args / hyperparameters to kwargs
    kwargs: Dict[str, Any] = smepu.argparse.parse_for_func(train_args)

    # Invoke a callable using the kwargs.
    estimator = DummyEstimator(**kwargs)
    logger.info("%s", estimator)

    # Start training, which will show tqdm bar.
    estimator.fit()


if __name__ == "__main__":
    # Minimal arg parsers for SageMaker protocol.
    parser = smepu.argparse.sm_protocol()
    args, train_args = parser.parse_known_args()

    # Demonstrate SageMaker checks.
    if not smepu.is_on_sagemaker():
        # When dev/testing script locally, it's convenient to auto-create these dirs.
        logger.info("Create model & output dirs prior to underlying function.")
        smepu.mkdir(args.model_dir)
        smepu.mkdir(args.output_data_dir)

    main(vars(args), train_args)
