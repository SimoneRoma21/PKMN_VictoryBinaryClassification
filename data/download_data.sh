#!/bin/bash

# Download the dataset from the Kaggle competition
kaggle competitions download -c fds-pokemon-battles-prediction-2025

# Unzip the zip file into the data folder
unzip fds-pokemon-battles-prediction-2025.zip -d .

# Delete the zip file
rm fds-pokemon-battles-prediction-2025.zip