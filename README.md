# ShaTS: A Shapley values python module for PyTorch Time Series Artificial Intelligence Models

<div align='center'>
<img src="ShaTSLogo.png" alt="ShaTS logo" width="200"/>
</div>


# Description

ShaTS (Shapley values for Time Series models) is a Shapley-based xAI method specifically designed for **time series Machine Learning (ML) and Deep Learning (DL) PyTorch/Keras models**. Unlike traditional Shapley methods, ShaTS applies **a priori feature grouping strategies** to preserve the temporal dependencies inherent in time-series data, thereby delivering coherent and actionable explanations that assist in pinpointing critical time instants and identifying important features or impacted physical components.

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

3. **Flexible background dataset handling**  
   ShaTS supports two ways of defining the background dataset:
   - Passing an explicit `background_dataset`.
   - Inferring the `background_dataset` from the full `train_dataset`.

   The background dataset can be inferred using different strategies:
   - **RANDOM**: random sample from the training dataset.
   - **ENTROPY**: selects the samples with the highest entropy.
   - **STRATIFIED**: preserves the class proportions of the training dataset.
   - **KMEANS**: computes `k` centroids from the training dataset and uses them as the background dataset.

4. **Scalability and Efficiency:**  
   By incorporating approximate methods for Shapley value computation, ShaTS effectively handles the high computational complexity associated with time series data and scales to large models and datasets without sacrificing performance. The library currently exposes the following explainers:

   - **ApproShaTS**
   - **FastShaTS**
   - **KernelShaTS**
   - **CachedKernelShaTS**
   - **FastShaTSIG**

5. **Integrated Visualization Tools:**  
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

The example notebooks located in the [examples folder](examples_notebooks/) require additional dependencies. Before running the notebooks, install the extra requirements with:
```bash
pip install -r tests/requirements.txt
```



## Usage
The examples notebooks provided in the [examples folder](example_notebooks/) provides a step-by-step guide on using ShaTS. Below is a quick example:

### 0. Data format

ShaTS expects datasets to be provided as **lists of tensors**, where each tensor has shape:

```python
[time, features]
```

For example, a dataset is expected to look like:

```python
dataset = [
    tensor_1,  # shape [T, F]
    tensor_2,  # shape [T, F]
    ...
]
```

### 1. Import ShaTS and configure the Explainer

In your Python script or notebook, begin by importing the necessary components from the ShaTS library. While the repository exposes the abstract ShaTS class, you will typically instantiate one of its concrete implementations (e.g., FastShaTS).

```python
import shats
from shats.grouping import TimeGroupingStrategy
from shats.grouping import FeaturesGroupingStrategy
from shats.grouping import MultifeaturesGroupingStrategy
```

### 2. Define the model wrapper

Assume you have a pre-trained time series PyTorch model (if you would like to see the steps for a Keras model go to [keras example](example_notebooks/example_keras/)). The explainer expects a callable that receives a tensor and returns a tensor of shape `[batch, nclass]`.



```python
def model_wrapper(x):
    return model(x)
```

If your model receives a single window with shape `[T, F]`, the wrapper should internally add the batch dimension when needed.


#### Option A: Use an explicit background dataset

Use this option when you already have a background dataset prepared.

```python
model = MyTrainedModel()
background_dataset = [...]  # list of tensors with shape [T, F]

explainer = shats.FastShaTS(
    model_wrapper=model_wrapper,
    background_dataset=background_dataset,
    grouping_strategy=FeaturesGroupingStrategy(names=variable_names),
)
```

#### Option B: Infer the background dataset from the training dataset

Use this option when you want ShaTS to build the background dataset automatically.

```python
model = MyTrainedModel()
train_dataset = [...]  # list of tensors with shape [T, F]

explainer = shats.FastShaTS(
    model_wrapper=model_wrapper,
    train_dataset=train_dataset,
    background_size=20,
    background_dataset_strategy=BackgroundDatasetStrategy.RANDOM,
    grouping_strategy="feature",
)
```

### 3. Compute Shapley Values
Once the explainer is initialized, compute the ShaTS values for your test dataset. The test dataset should be formatted similarly to the support dataset.
```python
testDataset = [...] 
shats_values = shaTS.compute(testDataset)
```

### 4. Plot Results
ShaTS includes a plotting function to help visualize the Shapley value attributions. This visualization uses a heatmap to display the contribution of each group (depending on the chosen strategy) over time. The plot also overlays the model’s predicted probability to facilitate interpretation.

```python
shaTS.plot(shats_values, test_dataset=testDataset, class_to_explain=1)
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
│      ├── test_background.py             # Test background
│      ├── test_igExplainer.py            # Test IG Explainer
│      └── test_shats_initialization.py   # Test ShaTS inicialization
│  
└── example_notebooks
    ├── example_SWaT
    │   ├── ADmodel.pt       # Trained model for SWaT
    │   └── example.ipynb    # Usage example with SWaT
    ├── example_toy_dataset
    │   ├── SinteticModel.pt # Trained model on toy dataset
    │   └── example.ipynb    # Usage example with toy dataset
    └── example_keras
        ├── KerasModel.h5    # Trained model on Keras
        ├── example.ipynb    # Usage example with toy dataset
        └── requirements.txt # Specific requirements

```

# Web demo
An interactive [demo](https://shats-lab.ovh/) has been developed for comparing ShaTS again post hoc SHAP on synthetic time-series scenarios.

On the site, you can compare performance and explanations between post-hoc SHAP (with grouping) and ShaTS. By grouping features a priori across time/sensors, ShaTS preserves temporal dependencies and yields clearer, time-localized attributions for anomalies.

1. 📊 Data → Generate a synthetic 3-variable time series with temporal dependencies and inject several anomaly types.

2. 💻 Train → Fit an LSTM anomaly detector on the dataset you created.

3. 🔎 Evaluate → Assess how well the model performs.

4. ⚠️ Explain → Compare ShaTS explanations against post-hoc SHAP for any selected anomaly.


<video src="./shats_webdemo.mp4" width="1280" height="720" controls></video>


# License
This project is licensed under the [MIT License](LICENSE).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Citation

If you use ShaTS in a scientific publication, we would appreciate using the following citation:

```bibtex
@article{de2025shats,
  title={ShaTS: A Shapley-based Explainability Method for Time Series Artificial Intelligence Models},
  author={Franco de la Pe{\~n}a, Manuel and Perales G{\'o}mez, {\'A}ngel Luis and Fernández Maim{\'o}, Lorenzo},
  journal={Future Generation Computer Systems},
  pages={108178},
  year={2025},
  publisher={Elsevier}
}

```
# Funding & Acknowledgements

This research stems from the Strategic Project DEFENDER (C087/23), a result of the collaboration agreement signed between the National Institute of Cybersecurity (INCIBE) and the University of Murcia. This initiative is carried out within the framework of the funds from the Recovery, Transformation, and Resilience Plan, financed by the European Union (Next Generation).

<p align="center">
  <img src="INCIBE-logos.jpg"
       alt="Funded by the European Union, Government of Spain and INCIBE"
       width="95%">
</p>


# Contact & Support
For questions or support, contact the author:
- Manuel Franco de la Peña (manuel.francop@um.es)
