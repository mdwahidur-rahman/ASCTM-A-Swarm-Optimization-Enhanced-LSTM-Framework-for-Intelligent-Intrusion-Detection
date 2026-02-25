# Network Security – Feature Selection + LSTM (KDD tabular)

This repository contains an end-to-end Jupyter/Google Colab workflow for **network intrusion / network security classification** using:

- **Bio-inspired feature selection** (wrapper-based) with **NiaPy** (e.g., Particle Swarm Optimization / Cat Swarm Optimization)
- A **baseline classifier** (SVM) trained on the selected feature subset
- A **deep sequence model** (**LSTM**) with an optional **custom activation** function
- Evaluation utilities (learning curve, confusion matrix, and common classification metrics)

> Main notebook: **`Network_Security (1).ipynb`**

---

## 1) What the notebook does (pipeline)

1. **Load dataset** (tabular CSV)
2. **Split train/test** (default: 80/20, stratified)
3. **Feature selection (wrapper method)**
   - Defines a fitness function that combines:
     - cross-validation error of an SVM classifier
     - a penalty on the number of selected features
   - Runs a NiaPy optimizer to search for a good feature mask
4. **Train & evaluate SVM** using the selected features
5. **Train & evaluate LSTM**
   - Runs the model with and without a custom activation function (optional)
6. **Plots and reports**
   - learning curve
   - confusion matrix
   - Accuracy / Precision / Recall / F1 (macro)

---

## 2) Dataset

The notebook expects a CSV file where:

- the first **40 columns** are features
- the **41st column** is the label

In the current notebook, the example path is:

```
/content/drive/MyDrive/02. Dataset/Network security/training_kdd.csv
```

### Recommended repository structure

```
.
├── Network_Security (1).ipynb
├── data/
│   ├── training_kdd.csv
│   └── unseen_test.csv            # optional
└── README.md
```

If you use the local `data/` folder, update the notebook path to:

```python
dataset = pd.read_csv("data/training_kdd.csv")
```

> Note: Please ensure labels are encoded consistently (e.g., integers or strings). If the label is categorical text, you may need to apply label encoding before training.

---

## 3) Requirements

You can run this project in **Google Colab** (recommended) or locally.

### Core Python packages

- `python>=3.9`
- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`
- `tensorflow` / `keras`
- `xgboost` (optional; imported in notebook)
- `niapy`

Install (Colab / local):

```bash
pip install niapy scikit-learn pandas numpy matplotlib tensorflow xgboost
```

---

## 4) How to run

### Option A — Google Colab (Drive-based)

1. Upload the notebook to Colab
2. Mount Google Drive
3. Put `training_kdd.csv` in your Drive folder
4. Update the CSV path in the notebook if needed
5. Run cells top-to-bottom

### Option B — Local run (recommended repo layout)

1. Put the dataset in `./data/`
2. Open the notebook and change the CSV path
3. Run:

```bash
jupyter notebook
```

---

## 5) Outputs

The notebook prints and/or generates:

- number of selected features
- final test metrics: Accuracy, Precision (macro), Recall (macro), F1 (macro)
- confusion matrix
- learning curve plot

---

## 6) Reproducibility notes

- Set `random_state` for `train_test_split` (already included in the notebook).
- For optimizer reproducibility, keep the `seed` in the NiaPy algorithm constructor.

---

## 7) Citation

If you use this code in academic work, please cite NiaPy and the dataset source you used (e.g., KDD/NSL-KDD), and also cite this repository.

---

## 8) License

Choose a license before public release (e.g., MIT, Apache-2.0).  
If you are unsure, **MIT** is a common choice for research code.
Paper Link: https://ieeexplore.ieee.org/abstract/document/11337741/
---

## Contact

For questions or collaboration, open an issue in this repository.
