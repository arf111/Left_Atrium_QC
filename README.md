# Left Atrium Quality Assessment


## Setup
Project requirements/dependencies:
- PyTorch
- [Segmentation PyTorch Library](https://github.com/qubvel/segmentation_models.pytorch)
PyPI version:
`$ pip install segmentation-models-pytorch`

## Usage
Data is needed to run the model. The data should be in the following format:
```
dataset
├── afib_data
│   ├── train
│   │   ├── patient_1
│   │   │   ├── image_1.png
│   │   │   ├── image_2.png
│   │   │   ├── ...
│   │   ├── patient_2
│   │   │   ├── image_1.png
│   │   │   ├── image_2.png
│   │   │   ├── ...
│   │   ├── ...
│   ├── test
│   │   ├── patient_1
│   │   │   ├── image_1.png
│   │   │   ├── image_2.png
│   │   │   ├── ...
│   │   ├── patient_2
│   │   │   ├── image_1.png
│   │   │   ├── image_2.png
│   │   │   ├── ...
│   │   ├── ...
│   ├── All_IQ_Scores
│   │   ├── patients_scores.csv
```
Then run the following in terminal:
`$ python3 train_qc_model.py`

