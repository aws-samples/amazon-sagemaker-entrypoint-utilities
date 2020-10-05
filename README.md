<div style="font-size:200%;font-weight:bold">Utilities for Amazon SageMaker
Training Entrypoint</div>

Table of contents:

- [1. Overview](#1-overview)
- [2. Installation](#2-installation)
- [3. Usage](#3-usage)
- [4. Security](#4-security)
- [5. License](#5-license)

# 1. Overview

This repo hosts an example of streamlining boiler-plate codes in SageMaker's
*training* entrypoint script.

The acronym `smepu` stands for **S**age**M**aker **e**ntry **p**oint
**u**tilities.

Main features:

1. Configure logger to consistently send logs to Amazon CloudWatch log streams.

2. Passthrough CLI-args to a wrapped estimator, so the entrypoint autors do not
have to write the boiler-plate codes that *"parses those 10+ CLI args, and calls
another estimator with those args."*

3. Automatically disable fancy outputs when running as Amazon SageMaker training
jobs.
   - Silence [`tqdm`](https://tqdm.github.io/) when training on Amazon
   SageMaker, to reduce the noise of your Amazon CloudWatch logs.

   - Plain output (i.e., no color, no fancy) for
   [`wasabi`](https://github.com/ines/wasabi), and
   [`spacy`](https://github.com/explosion/spaCy) CLI (e.g., `train` or
   `convert`).

# 2. Installation

```bash
pip install \
    'git+https://github.com/aws-samples/amazon-sagemaker-entrypoint-utilities@master#egg=smepu'
```

or:

```bash
git clone \
    https://github.com/aws-samples/amazon-sagemaker-entrypoint-utilities.git

cd amazon-sagemaker-entrypoint-utilities
pip install -e .
```

# 3. Usage

**Pre-requisite**: know how to write an Amazon SageMaker training entrypoint.

A working hello-world example is provided under `examples/`. There are two
versions provided:

1. `entrypoint.py` uses `argparse` to parse hyperparameters.
2. `entrypoint-click.py` uses `click` to parse hyperparameters.

Use `examples/entrypoint.sh` to quickly observe the behavior of those train
entrypoints when they run directly on your Python environment in your machine.

> \[**NOTE**: **not to be confused** with "Amazon SageMaker local mode" which
> refers to running the script on a SageMaker container running on a SageMaker
> notebook instance.\]
>
> Running the train script directly on your Python environment is a useful trick
> to speed-up your "dev + functional-test" cycle. Typically this stage utilizes
> synthetic tiny dataset, and you heavily leverage your favorite dev tools
> (i.e., unit-test frameworks, code debuggers, etc.).
>
> After this, you can perform "compatibility test" by running your train script
> on a Amazon SageMaker training container (whether on "Amazon SageMaker local
> mode" or a training instance), to iron-out compatibilities issues.
>
> When your scripts have been fully tested, then you can start your actual,
> large-scale model training & experimentation on Amazon SageMaker training
> instances.

Sample runs:

```bash
# Run entrypoint script outside of SageMaker.
examples/entrypoint.sh

# Mimic running on Amazon SageMaker: automatically off tqdm.
SM_HOSTS=abcd examples/entrypoint.sh

# Run click-version of entrypoint
examples/entrypoint.sh click
```

To experiment with different hyperparameters, see `DummyEstimator` in
`examples/dummyest.py`, and if necessary modify accordingly `complex_args` in
`examples/entrypoint.sh`.

# 4. Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more
formation.

# 5. License

This library is licensed under the MIT-0 License. See the LICENSE file.
