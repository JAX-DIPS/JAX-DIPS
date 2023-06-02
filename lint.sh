#!bin/bash
pip install black==23.3.0 isort==5.12.0
black --skip-string-normalization --line-length=119 .
isort --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parenthese --line-width=119 --ws --skip=external .