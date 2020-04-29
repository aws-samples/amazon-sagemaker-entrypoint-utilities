import logging
from time import sleep
from typing import Any, List

from tqdm import tqdm


class DummyEstimator(object):
    """Simulate tqdm-dependant ML toolkits.

    Known cases: gluonts, run_ner.py (from huggingface/transformers).
    """

    def __init__(self, epochs: int = 2, init: str = "uniform", callbacks: List[Any] = []) -> None:
        """Initialize a ``DummyEstimator`` instance.

        Args:
            epochs (int, optional): The number of training epochs]. Defaults to 2.
            init (str, optional): Weight initialization. Defaults to "uniform".
        """
        self.epochs = epochs
        self.init = str(init)
        self.callbacks = callbacks

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(epochs={self.epochs}, init="{self.init}", callbacks={self.callbacks})'

    def fit(self, *args, **kwargs) -> None:
        for epoch in range(1, 1 + self.epochs):
            for i in tqdm(range(3)):
                sleep(1.0)
            logging.info("Epoch %s", epoch)


class DummyCallback(object):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f'{self.__class__.__name__}(name="{self.name}")'

    def __repr__(self):
        return self.__str__()
