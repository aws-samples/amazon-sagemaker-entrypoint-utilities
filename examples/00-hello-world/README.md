Both `entrypoint.py` and `entrypoint-click.py` show minimal working examples
of how to use `smepu` to write meta-entrypoint scripts that can run as SageMaker
training jobs.

Typically, a SageMaker training job expects a training script to follow this
structure:

1. parse hyperparameters via one of these mechanics: (a) cli args, (b)
   environment variables, or (c) a json file. `smepu` is designed to simplify
   case (a). Should the script parse hyperparameters using (b) or (c), then
   please consult the SageMaker documentation.

2. Instantiate an instance of estimator with hyperparameters as the kwargs to
   the estimator class, e.g., `EstimatorClass(hyperparam_a=1, hyperparam_b=4)`.
   This is where `smepu` can help by automatically parsing the hyperparameters
   cli args and converting those cli args to a `kwargs` dictionary.

3. The rest are typical of a training script, mainly `fit()`, validation, and
   saving the fitted estimator into 1+ file(s).

Normally, for step (1a) and (2), the script hardcodes two things:

- the estimator class, and
- the translation of hyperparameters cli args to estimator's kwargs.

These are boilerplate codes, and writing these over-and-over again are mundane
tasks, especially when you want to experiment with multiple algorithms.

This is precisely what `smepu` was designed for: to help script authors save
their precious time from hardcoding similar patterns (i.e., instantiate from an
estimator class and translating `cli_args -> kwargs`) over-and-over again.
