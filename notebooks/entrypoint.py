# Make sure to import smephp before you import tqdm or any other module that
# uses tqdm.
#
# NOTE: depending on how you setup your favorite editor with formatter+isort,
# a few savings may be needed to get the correct order. For e.g., on vscode
# with black + isort enabled, it may take 2x saves.
#
# The 1st save may put smephp after tqdm (or tqdm-dependant modules), and it
# takes the 2nd (or possibly more) save to rearrange smephp to the top.
import smephp

from time import sleep
from typing import Any, Dict, List

from tqdm import tqdm

# Setup logger must be done in the entrypoint script.
logger = smephp.setup_opinionated_logger(__name__)


def main(cfg: Dict[str, Any], train_args: List[str]) -> None:
    logger.info("cfg: %s", cfg)
    logger.info("train_args: %s", train_args)

    # Convert cli args / hyperparameters to kwargs
    kwargs: Dict[str, Any] = smephp.argparse.parse_for_func(train_args)

    # Invoke a callable using the kwargs.
    estimator = DummyEstimator(**kwargs)
    logger.info("%s", estimator)

    # Start training, which will show tqdm bar.
    estimator.fit()


class DummyEstimator(object):
    """Simulate tqdm-dependant ML toolkits.

    Known cases: gluonts, run_ner.py (from huggingface/transformers).
    """

    def __init__(self, epochs: int = 2, init: str = "uniform") -> None:
        """Initialize a ``DummyEstimator`` instance.

        Args:
            epochs (int, optional): The number of training epochs]. Defaults to 2.
            init (str, optional): Weight initialization. Defaults to "uniform".
        """
        self.epochs = epochs
        self.init = str(init)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(epochs={self.epochs}, init="{self.init}")'

    def fit(self, *args, **kwargs) -> None:
        for epoch in range(1, 1 + self.epochs):
            for i in tqdm(range(3)):
                sleep(1.0)
            logger.info("Epoch %s", epoch)


if __name__ == "__main__":
    # Minimal arg parsers for SageMaker protocol.
    parser = smephp.argparse.sm_protocol()
    args, train_args = parser.parse_known_args()

    # Demonstrate SageMaker checks.
    if not smephp.is_on_sagemaker():
        # When dev/testing script locally, it's convenient to auto-create these dirs.
        logger.info("Create model & output dirs prior to underlying function.")
        smephp.mkdir(args.model_dir)
        smephp.mkdir(args.output_data_dir)

    main(vars(args), train_args)
