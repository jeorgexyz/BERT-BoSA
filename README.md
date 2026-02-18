# BERT BoSA
### Bidirectional Encoder Representations from Transformers using Boltzmann Simulated Annealing

BERT BoSA is a quantum-inspired implementation of the Boltzmann Simulated Annealing algorithm for optimizing the configuration of the BERT (Bidirectional Encoder Representations from Transformers) model architecture. The project aims to find the optimal configuration of BERT parameters, such as the number of transformer encoder layers, attention heads, embedding dimension, and dropout rate, to achieve the highest possible accuracy on a given text classification task.

## Requirements

- Python 3.6 or later
- PyTorch
- Torchtext

You can install the required Python packages by running:

```
pip install -r requirements.txt
```

## Dataset

The project uses the AG_NEWS dataset for text classification, which is included in the `torchtext` library. The dataset will be automatically downloaded and extracted to the `data/ag_news/` directory when you run the code for the first time.

## Folder Structure

```
BERT-BoSA/
├── data/
│   └── ag_news/
│       ├── train.csv
│       └── test.csv
├── models/
│   ├── __init__.py
│   ├── bert.py
│   └── classification_head.py
├── utils/
│   ├── __init__.py
│   └── data_utils.py
├── optimization/
│   ├── __init__.py
│   └── simulated_annealing.py
├── config.py
├── train.py
├── eval.py
├── requirements.txt
└── README.md
```

## Usage

1. **Configuration**: Modify the `config.py` file to set the desired initial configuration for the BERT model and the optimization process.

2. **Training**: Run the `train.py` script to train the BERT model with the optimized configuration obtained from the Simulated Annealing algorithm:

   ```
   python train.py
   ```

   This script will perform the following steps:
   - Load the AG_NEWS dataset
   - Initialize the BERT model with the initial configuration
   - Optimize the configuration using the Simulated Annealing algorithm
   - Train the BERT model with the optimized configuration on the training dataset
   - Save the trained model

3. **Evaluation**: Run the `eval.py` script to evaluate the trained BERT model on the test dataset:

   ```
   python eval.py
   ```

   This script will load the trained BERT model and compute its accuracy on the test dataset.

## Customization

- **Dataset**: If you want to use a different dataset for text classification, modify the `data_utils.py` file accordingly.
- **Model**: You can modify the `bert.py` and `classification_head.py` files to customize the BERT model architecture or the classification head.
- **Optimization**: If you want to use a different optimization algorithm or modify the Simulated Annealing implementation, update the `simulated_annealing.py` file.


