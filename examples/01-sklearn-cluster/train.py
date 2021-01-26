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

from pathlib import Path
from pydoc import locate
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score

# Setup logger must be done in the entrypoint script.
logger = smepu.setup_opinionated_logger(__name__)


class Output:
    """An output writer to save an estimator and its output to filesystems.

    Will overwrite existing files.
    """

    def __init__(self, model_dir: Path, output_data_dir: Path) -> None:
        """Create a writer.

        Args:
            model_dir (Path): model directory
            output_data_dir (Path): data output directory
        """
        self.model_dir = model_dir
        self.output_data_dir = output_data_dir

    def save_model(self, estimator: Any, metadata: Optional[Any] = None) -> None:
        """Save estimator to `model.joblib`, or `model-{metadata}.joblib` if metadata is not None.

        Args:
            estimator (Any): an object to persist.
            metadata (Any, optional): If not None, output filename is `model-{metadata}.joblib`.
                Defaults to None.
        """
        fname = "model.joblib" if metadata is None else f"model-{metadata}.joblib"
        joblib.dump(estimator, self.model_dir / fname)

    def save_labels(self, labels: pd.DataFrame, metadata: Optional[Any] = None) -> None:
        """Save cluster labels to `labels.csv`, or `labels-{metadata}.csv` if metadata is not
        None.

        Args:
            labels (pd.DataFrame): dataframe to save.
            metadata (Any, optional): If not None, output filename is `labels-{metadata}.csv.
                Defaults to None.
        """
        fname = "labels.csv" if metadata is None else f"labels-{metadata}.csv"
        labels.to_csv(self.output_data_dir / fname, index=False)

    def save_metrics(
        self, metrics: Sequence[Mapping[str, Any]], metadata: Optional[Mapping[str, Sequence[Any]]] = None
    ) -> None:
        """Save cluster metrics to `cluster.csv`, prepended by metadata columns.

        Args:
            metrics (List[Dict[str, Any]]): List of metric sets, where each metric set is {'name': value}.
            metadata (Optional[Dict[str, List[Any]]]): Metadata columns of each metric set, where
                `len(metadata) == len(metrics)`. Default to None.
        """
        df = pd.DataFrame(metrics)
        if metadata:
            header = pd.DataFrame(metadata)
            df = pd.concat([header, df], axis=1)
        df.to_csv("cluster.csv", index=False, header=True)


class MultiOutput(Output):
    """An output writer to save multiple estimators and their output to filesystems.

    Will overwrite existing files.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.opath = self.output_data_dir / "labels.csv"
        self.header = True

    def save_model(self, estimator: Any, metadata: Any) -> None:  # type: ignore
        """Save estimator to `model-{metadata}.joblib`.

        Args:
            estimator (Any): an object to persist.
            metadata (Any): Mandatory metadata to distinguish one model file to another. As as
                example, a metadata may denote the `n_clusters` hyperparameter of the estimator.
        """
        super().save_model(estimator, metadata)

    def save_labels(self, labels: pd.DataFrame, metadata: Any) -> None:  # type: ignore
        """Add column `n_clusters=metadata` to the dataframe, then append to file `labels.csv`.

        Args:
            labels (pd.DataFrame): cluster labels dataframe.
            metadata (Any): Mandatory metadata that denotes the `n_clusters` hyperparameter used to
                generate the cluster dataframe.
        """
        df = labels.copy()
        df.insert(0, "n_clusters", metadata)
        if self.header:
            df.to_csv(self.opath, mode="w", index=False)
            self.header = False
        else:
            df.to_csv(self.opath, mode="a", index=False, header=self.header)


def main(cfg: Mapping[str, Any], hyperparams: Sequence[str]) -> None:
    """Load data, train, predict, then save model, output, and reports.

    Args:
        cfg (Mapping[str, Any]): this script's configuration
        hyperparams (Sequence[str]): hyperparameters for estimator, in cli args format.
    """
    logger.info("cfg: %s", cfg)
    logger.info("hyperparams: %s", hyperparams)

    # Deserialized estimator class and hyperparameters
    klass, est_kwargs = parse_estimator_cli_args(cfg["algo"], hyperparams)
    logger.info("Estimator class: %s", klass)

    # Estimator is an instance of "algo" class. Raises an exception if override
    # n_clusters is requested but estimator's __init__() does not accept kwarg
    # `n_clusters`.
    main2(
        cfg["train"],
        cfg["model_dir"],
        cfg["output_data_dir"],
        klass,
        est_kwargs,
        cfg["sweep"],
        cfg["sweep_start"],
        cfg["sweep_end"],
    )


def main2(
    train_channel: Path,
    model_dir: Path,
    output_data_dir: Path,
    est_klass: Type,
    est_kwargs: Dict[str, Any] = {},
    sweep: bool = False,
    sweep_start: int = 2,
    sweep_end: int = 4,
) -> None:
    # Setup output writer specifically for single run vs sweeping runs.
    writer_cls = Output if not sweep else MultiOutput
    writer = writer_cls(model_dir, output_data_dir)

    # Load, fit_predict, save.
    df = load_data(train_channel)

    # Figure-out what trials to carry out.
    if not sweep:
        trial: List[Optional[int]] = [None]  # Type annotate to keep mypy happy
        metric_metadata: Optional[Dict[str, Any]] = None
    else:
        if not (0 < sweep_start <= sweep_end):
            raise ValueError(f"Invalid sweep range: {[sweep_start, sweep_end]}")

        trial = [i for i in range(sweep_start, sweep_end + 1)]
        metric_metadata = {"n_clusters": trial}

    metric_set = []
    for n_clusters in trial:
        estimator, labels, metrics = fit_predict(df, est_klass, est_kwargs, n_clusters)
        writer.save_model(estimator, n_clusters)
        writer.save_labels(labels, n_clusters)
        metric_set.append(metrics)
    writer.save_metrics(metric_set, metric_metadata)


def load_data(path: Path) -> pd.DataFrame:
    """Load all files under `path`, but skip hidden files which start with a `.`.

    Files can end in any extension that `pd.read_csv()` accept, thus compressed csv allowed.

    Args:
        path (Path): directory of files to load.

    Returns:
        pd.DataFrame: dataframe of loaded input files.
    """
    # Load all input files into a single dataframe.
    dfs = []
    for fpath in path.resolve().glob("**/*"):
        df = pd.read_csv(fpath, dtype={0: str}, low_memory=False)
        dfs.append(df)
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)

    # Treat null values in the dataframe.
    if df.isna().values.any():
        logger.warn('NA detected in input. To convert NA strings to "" and NA numbers to 0.0')
        df.iloc[:, 0].fillna("", axis=1, inplace=True)
        df.iloc[:, 1:].fillna(axis=1, inplace=True)

    return df


def fit_predict(
    df: pd.DataFrame,
    algo: Type,
    hyperparams: Dict[str, Any],
    override_n_clusters: Optional[int] = None,
) -> Tuple[ClusterMixin, pd.DataFrame, Dict[str, Any]]:
    """Cluster the dataframe.

    Args:
        df (pd.DataFrame): input dataframe.
        algo (str): estimator classname.
        hyperparams (Sequence[str]): estimator hyperparameters.
        override_n_clusters (int, optional): If int, set `n_clusters` of the
            estimator. Defaults to None.

    Returns:
        Tuple[ClusterMixin, pd.DataFrame, Dict[str, Any]]: (estimator, cluster labels, metrics)
    """
    estimator = create_estimator(algo, hyperparams, override_n_clusters)
    logger.info("estimator: %s", estimator)

    X = df.iloc[:, 1:]
    labels: np.ndarray = estimator.fit_predict(X)
    cluster_metric = {
        "calinski_harabasz_score": calinski_harabasz_score(X, labels),
        "davies_bouldin_score": davies_bouldin_score(X, labels),
        "silhouette_score": silhouette_score(X, labels),
        "aic": try_metric(estimator, X, "aic"),
        "bic": try_metric(estimator, X, "bic"),
    }

    return (
        estimator,
        dfify_clusters({"cluster_id": labels, "silhouette": silhouette_samples(X, labels)}, df),
        cluster_metric,
    )


def try_metric(estimator: ClusterMixin, X: np.ndarray, name: str) -> Optional[float]:
    try:
        f = getattr(estimator, name)
        return f(X)
    except AttributeError:
        return None


def parse_estimator_cli_args(
    clsname: str,
    hyperparams: Sequence[str],
) -> Tuple[Type, Dict[str, Any]]:
    """Deserialize estimator class name and hyperparameters cli args to `type` and kwargs dictonary.

    Args:
        clsname (str): estimator's class name.
        hyperparams (Sequence[str]): hyperparameters in the cli args format.

    Returns:
        ClusterMixin: clustering estimator.
    """
    return cast(Type, locate(clsname)), smepu.argparse.to_kwargs(hyperparams)


def create_estimator(
    klass: Any,
    est_kwargs: Dict[str, Any],
    override_n_clusters: Optional[int] = None,
) -> ClusterMixin:
    """Instantiate `clsname` initialized with`hyperparams` cli args.

    When `override_n_clusters` is an int, force `n_clusters` or `n_components` to this int.
    This raises an exception if estimator's `__init__()` does not accept `n_clusters` or
    `n_components`.

    Args:
        clsname (str): estimator's class name.
        hyperparams (Sequence[str]): hyperparameters in the cli args format.
        override_n_clusters (int, optional): If int, force set `n_clusters`. Defaults to None.

    Returns:
        ClusterMixin: clustering estimator.
    """
    if override_n_clusters is not None:
        est_kwargs["n_clusters"] = override_n_clusters
    estimator = klass(**est_kwargs)
    return estimator


def dfify_clusters(cols: Dict[str, np.ndarray], df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate cluster ids with their input features.

    Args:
        cols (Dict[str, np.ndarray]): Columns to prepend to df.
        df (pd.DataFrame): inputrecord ids and their features.

    Returns:
        pd.DataFrame: cluster labels with features.
    """
    sers = [pd.Series(a, name=c) for c, a in cols.items()]
    retval = pd.concat([*sers, df], axis=1)
    return retval


def add_argument(parser):
    parser.add_argument(
        "--algo",
        type=str,
        help="Full name of estimator class that provides .fit() and .predict()",
        default="sklearn.cluster.KMeans",
    )

    group = parser.add_argument_group("sweep")
    group.add_argument(
        "--sweep",
        type=int,
        default=0,
        help="Sweep through multiple n_clusters; will activate --min and --max.",
    )
    group.add_argument("--sweep-start", type=int, help="Start n_clusters to search (defaults=2)", default=2)
    group.add_argument("--sweep-end", type=int, help="End n_clusters to search (defaults=4)", default=4)


if __name__ == "__main__":
    logger.info("Entrypoint script that uses argparse to digest hyperparameters.")
    parser = smepu.argparse.sm_protocol()  # Minimal arg parsers for SageMaker protocol.
    add_argument(parser)  # This script's args
    args, train_args = parser.parse_known_args()

    # When dev/testing script locally, it's convenient to auto-create these dirs.
    if not smepu.is_on_sagemaker():
        logger.info("Create model & output dirs prior to underlying function.")
        smepu.mkdir(args.model_dir)
        smepu.mkdir(args.output_data_dir)

    main(vars(args), train_args)
