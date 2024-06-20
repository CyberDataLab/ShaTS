
# TG-SHAP Package

## Introduction

TG-SHAP (TimeGroup-SHAPley) is a novel method designed to provide interpretability for Machine Learning (ML) and Deep Learning (DL) models that work with time series data in industrial environments. This method is particularly useful for models that perform Anomaly Detection (AD), allowing operators to understand the influence of different time instants on the model's predictions.

## Background

In the industrial sector, the integration of sensors and actuators into factories is a key characteristic of Industry 5.0, aiming to enhance worker safety and operational efficiency. However, this increased connectivity also introduces new vulnerabilities to cyberattacks. Traditional signature-based detection methods have become obsolete due to the sophistication of modern attacks. Thus, there is a shift towards anomaly detection, which identifies deviations from expected operational behaviors.

ML and DL models are highly effective in AD, but their adoption in critical industrial settings is hindered by their lack of interpretability. Interpretability refers to the ability of human operators to understand the reasons behind the classification of certain patterns as anomalous. This is crucial for effective intervention and mitigation.

### Shapley Values

The Shapley value method, rooted in cooperative game theory, is a model-agnostic approach that assigns a specific contribution to each input feature for individual model predictions. Despite its robustness and comprehensibility, calculating Shapley values for models with large input spaces is computationally intensive. This complexity increases for models that exploit data sequentiality, common in AD models.

### TG-SHAP

TG-SHAP addresses these challenges by grouping features by time instants, significantly reducing the computational complexity from \(2^{p \cdot w}\) to \(2^w\), where \(p\) is the number of features measured at each instant and \(w\) is the number of time instants in a window. This transformation maintains the interpretability of the model's decisions, helping identify which time instants most influence the model's predictions.

## License

This project is licensed under the MIT License.

## Acknowledgements

This project is based on the work presented in the master's thesis by Manuel Franco de la Peña, Universidad de Murcia, 2024.