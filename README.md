# Concept Drift Detection for Fraud Detection

This project implements various concept drift detection algorithms to evaluate their performance on fraud detection datasets. The goal is to identify and handle concept drift in streaming data, ensuring robust and adaptive machine learning models.


### Key Components

1. **`main.py`**: The main script that orchestrates the loading of datasets, evaluation of models, and visualization of results.

2. **`concept_drift/`**: Contains implementations of various concept drift detection algorithms, including:
   - Statistical-based detectors (e.g., ADWIN, KSWIN, EDDM, DDM)
   - Window-based detectors (e.g., FHDDM, WSTD, CUSUM)
   - Ensemble-based detectors (e.g., ARF, AUE, AWE, DWM)
   - Other advanced detectors (e.g., D3, FTDD, RDDM)

3. **`utils/`**: Utility scripts for data loading, evaluation, and visualization:
   - `load_data.py`: Functions to load datasets from Hugging Face.
   - `evaluation.py`: Functions to evaluate models and drift detectors.
   - `visualization.py`: Functions to create heatmaps, timelines, and other visualizations.

4. **`results/`**: Directory to store generated visualizations and summary results.

## Datasets

The project uses several fraud detection datasets, loaded from Hugging Face:
- Synthetic Financial Dataset
- Credit Card Fraud Dataset (Nooha)
- European Credit Card Fraud Dataset
- Thomas K Credit Card Fraud Dataset

## Drift Detectors

The project implements and evaluates the following drift detection algorithms:
- **Statistical-Based Detectors**: ADWIN, KSWIN, EDDM, DDM
- **Window-Based Detectors**: FHDDM, WSTD, CUSUM
- **Ensemble-Based Detectors**: ARF, AUE, AWE, DWM
- **Other Detectors**: D3, FTDD, RDDM, ACE

## Models

The following machine learning models are used for evaluation:
- Naive Bayes (`GaussianNB`)
- Hoeffding Tree Classifier (`HoeffdingTreeClassifier`)

## Visualizations

The project generates several visualizations to analyze the performance of drift detectors:
- Heatmaps comparing AUC scores across datasets and detectors.
- Drift detection timelines showing accuracy over time with drift points.
- Multi-dataset heatmaps for AUC performance comparison.

## How to Run

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:
    ```
    python main.py
    ```

3. Visualizations and results will be saved in the results/ directory.

## Results
The results include:

- Heatmaps summarizing the performance of drift detectors across datasets.
- Drift detection timelines for individual datasets.
- Summary statistics for each drift detector.

## Dependencies
- Python 3.8+
- Libraries:
    - river
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - datasets

## License
This project is licensed under the MIT License.