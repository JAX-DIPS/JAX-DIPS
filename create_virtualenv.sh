#!/bin/bash

# Name of the virtual environment
ENV_NAME="env_jax_dips"

# Check if virtual environment exists
if [ -d "$ENV_NAME" ]; then
  echo "Virtual environment $ENV_NAME already exists."
  # Activate virtual environment
  source $ENV_NAME/bin/activate
else
  # Create virtual environment
  python3 -m venv $ENV_NAME
  # Activate virtual environment
  source $ENV_NAME/bin/activate
  # Install requirements
  file="requirements.txt"
  output_file="_requirements.txt"
  # Read the file line by line, starting from the third line
  while IFS= read -r line
  do
    if [[ ! -z "$line" ]]  # Ignore empty lines
    then
      echo "$line" >> "$output_file"
    fi
  done < <(tail -n +3 "$file")
  pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  pip install -r $output_file
  rm $output_file
  # Display success message
  echo "Virtual environment $ENV_NAME created, activated, and requirements installed."
fi

