# PKMN_VictoryBinaryClassification

Repo for the challenge of Foundamentals of Data Science at Sapienza University of Rome. <br>

Owners of the repo: <br>
-Simone Roma (roma.1999214@studenti.uniroma1.it) <br>
-Cristian Sirghie (sirghie.1993485@studenti.uniroma1.it) <br>
-Patryk J. Mulica (mulica.1986671@studenti.uniroma1.it) <br>

# Virtual Env
To set up the development environment, run the following script in your terminal:

```bash
source activate.sh
```

This script will:
- Create a Python virtual environment (if not already present)
- Activate the virtual environment
- Install all dependencies listed in `requirements.txt`

Make sure you have Python 3 installed before running the script.

# Dataset
To download the dataset, you can use the provided `download_data.sh` script. **Note:** You must have both `kaggle` and `unzip` installed on your system.

- To install the Kaggle CLI, follow the instructions at [https://www.kaggle.com/settings](https://www.kaggle.com/settings).
- Make sure your Kaggle API credentials (`kaggle.json`) are set up as described on the Kaggle website.

If you do not have `kaggle` or `unzip` installed, you can manually download the dataset by visiting [https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2025/data](https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2025/data) and copying the files into the `./data` directory.
