# Bank-Note Authenticator

A machine learning project that classifies bank notes as authentic or forged based on features extracted from images.

## Project Overview
This project uses a **Random Forest Classifier** to predict the authenticity of bank notes. The model is trained on data extracted from genuine and forged banknote-like specimens.

## Dataset
The dataset `BankNote_Authentication.csv` contains data extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.

### Features:
1.  **variance**: variance of Wavelet Transformed image (continuous)
2.  **skewness**: skewness of Wavelet Transformed image (continuous)
3.  **curtosis**: curtosis of Wavelet Transformed image (continuous)
4.  **entropy**: entropy of image (continuous)
5.  **class**: class (0 for authentic, 1 for forgery)

## Requirements
The project requires the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `pickle`

## Installation
Ensure you have [uv](https://github.com/astral-sh/uv) installed or use `pip`:

```bash
cd bank-note-authenticator
pip install -r requirements.txt
```

## Usage
To train the model and generate the classifier:

```bash
python main.py
```

This will:
1. Load the dataset.
2. Train a Random Forest Classifier.
3. Print the model accuracy.
4. Save the trained model as `Classifiermodel.pkl`.

## Project Structure
- `main.py`: The main script for training the model and saving it.
- `BankNote_Authentication.csv`: The dataset file.
- `Classifiermodel.pkl`: The saved trained model.
- `data.ipynb`: Jupyter notebook for data exploration.
