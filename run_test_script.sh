#!/bin/bash

# Define the project directory
PROJ_DIRECTORY=$(pwd)

# Navigate to the project directory and pull the Docker image
cd "$PROJ_DIRECTORY" && make pull

# List Docker images and grep for the specific image
docker images | grep spacecraft-pose-object-detection

# Navigate to the project directory and pack the benchmark
cd "$PROJ_DIRECTORY" && make pack-example

# Navigate to the project directory and test the submission
cd "$PROJ_DIRECTORY" && make test-submission

# Run the scoring Python script
python3 scripts/score.py submission/submission.csv data/test_labels.csv
