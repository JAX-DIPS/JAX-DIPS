#!/usr/bin/bash
black --skip-string-normalization --line-length=119 .
isort --multi-line=3 --trailing-comma --use-parenthese --line-width=119 --ws tests jax-dips examples