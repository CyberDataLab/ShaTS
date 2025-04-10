# ShaTS: A Shapley values python module for PyTorch Time Series Artificial Intelligence Models

<div align='center'>
<img src="ShaTSLogo.png" alt="ShaTS logo" width="200"/>
</div>


# Description

ShaTS (Shapley values for Time Series models) is a Shapley-based xAI method specifically designed for **time series Machine Learning (ML) and Deep Learning (DL) PyTorch models**. Unlike traditional Shapley methods, ShaTS applies **a priori feature grouping strategies** to preserve the temporal dependencies inherent in time-series data, thereby delivering coherent and actionable explanations that assist in pinpointing critical time instants and identifying important features or impacted physical components.

This repository includes:
- The **ShaTS library** as Python module.
- Example Jupyter Notebooks (`\tests`) demonstrating the usage of ShaTS on SWaT, an industrial and a sintetic dataset. 


# Features

1. **Model Agnostic:**  
   ShaTS is designed to work with any time series ML/DL model, making it a versatile tool for numerous and diverse applications.

2. **Advanced Grouping Strategies:**  
   ShaTS overcomes common limitations in time series explainability by employing different a priori grouping strategies:
   - **Temporal Grouping:** Aggregates features by individual time instants to identify key moments influencing model predictions.
   - **Feature Grouping:** Evaluates the importance of all the measurements of each individual feature across the time window.
   - **Multi-Feature Grouping:** Enables custom groupings that consolidate related features (such as those representing a specific sensor/actuator or an entire industrial process) to deliver more coherent and actionable explanations.

3. **Scalability and Efficiency:**  
   By incorporating approximate methods for Shapley value computation, ShaTS effectively handles the high computational complexity associated with time series data and scales to large models and datasets without sacrificing performance.

4. **Integrated Visualization Tools:**  
   Custom plotting capabilities allow users to analyze feature contributions over time, providing intuitive visual summaries of how various groups impact the model’s output and facilitating faster decision-making in real-world applications.




# Installation & Setup

## Prerequisites

Before installing **ShaTS**, make sure your environment meets the following requirements:

+ Python 3.12 or higher
+ pip (Python package installer)
+ (Recommended) A virtual environment like venv or conda

You can check your Python version with:
```bash
python --version
```

You can set up a virtual environment with:

```bash
python -m venv .venv
source .venv/bin/activate
```


## Installation Steps

To install the library, clone the repository and ensure dependencies are installed:

```bash
git clone <repo_url>
cd <repo_directory>
pip install -r requirements.txt
```

The example notebooks located in the [tests folder](tests/) require additional dependencies. Before running the notebooks, install the extra requirements with:
```bash
pip install -r tests/requirements.txt
```



## Usage
The examples notebooks provided in the [tests folder](tests/) provides a step-by-step guide on using ShaTS. Below is a quick example:

### 1. Import ShaTS and configure the Explainer

In your Python script or notebook, begin by importing the necessary components from the ShaTS library. While the repository exposes the abstract ShaTS class, you will typically instantiate one of its concrete implementations (e.g., FastShaTS).

```python
import shats
from shats.grouping import TimeGroupingStrategy
from shats.grouping import FeaturesGroupingStrategy
from shats.grouping import MultifeaturesGroupingStrategy
```

### 2. Initialize the Model and Data

Assume you have a pre-trained time series PyTorch model and a background dataset, which should be a list of tensors representing typical data samples that the model has seen during training.

```python
model = MyTrainedModel()
support = [...]  

shapley_class = shats.FastShaTS(model, 
    support_dataset=support,
    grouping_strategy= FeaturesGroupingStrategy(names=variable_names)
)
```

### 3. Compute Shapley Values
Once the explainer is initialized, compute the ShaTS values for your test dataset. The test dataset should be formatted similarly to the support dataset.
```python
testDataset = [...] 
tsgshap_values = shaTS.compute(testDataset)
```

### 4. Visualize Results
ShaTS includes a plotting function to help visualize the Shapley value attributions. This visualization uses a heatmap to display the contribution of each group (depending on the chosen strategy) over time. The plot also overlays the model’s predicted probability to facilitate interpretation.

```python
shaTS.plot(tsgshap_values, test_dataset=testDataset, class_to_explain=1)
```




# Folder structure
The repository is organized as follows:

```
.
├── LICENSE                  # License file
├── README.md                # Documentation
├── requirements.txt         # Project dependencies
├── setup.py                 # Packaging script
├── ShaTSLogo.png            # Project logo
├── src
│   └── shats
│       ├── __init__.py      # Module initialization
│       ├── grouping.py      # Grouping logic
│       ├── shats.py         # Main ShaTS implementation
│       └── utils.py         # Utility functions and strategy enums
└── tests
    ├── example_SWaT
    │   ├── ADmodel.pt       # Trained model for SWaT
    │   └── example.ipynb    # Usage example with SWaT
    └── example_toy_dataset
        ├── SinteticModel.pt # Trained model on toy dataset
        └── example.ipynb    # Usage example with toy dataset
```


# License
This project is licensed under the [MIT License](LICENSE).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Citation



# Contact & Support
For questions or support, contact the author:
- Manuel Franco de la Peña (manuel.francop@um.es)
