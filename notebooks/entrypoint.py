import logging
from typing import Any, Dict, List

import smephp

logger = smephp.setup_opinionated_logger(__name__)


def main(cfg: Dict[str, Any], train_args: List[str]):
    logger.info("cfg: %s", cfg)
    logger.info("train_args: %s", train_args)

    # Convert cli args / hyperparameters to kwargs
    kwargs: Dict[str, Any] = smephp.argparse.parse_for_func(train_args)

    # Invoke a callable using the kwargs.
    estimator = DummyEstimator(**kwargs)
    logger.info("%s", estimator)


class DummyEstimator(object):
    def __init__(self, epochs: int = 100, init: str = "uniform"):
        self.epochs = epochs
        self.init = str(init)

    def __str__(self):
        return f'{self.__class__.__name__}(epochs={self.epochs}, init="{self.init}")'


if __name__ == "__main__":
    # Demonstrate checks
    logger.info("Probably on SageMaker: %s", smephp.is_on_sagemaker())

    parser = smephp.argparse.sm_protocol()
    args, train_args = parser.parse_known_args()

    logger.info("Create model & output dirs prior to underlying function.")
    smephp.mkdir(args.model_dir)
    smephp.mkdir(args.output_data_dir)

    main(vars(args), train_args)
