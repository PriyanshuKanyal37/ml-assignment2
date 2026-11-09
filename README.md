# ML Assignment 2 — Student Dropout & Academic Success Classifier

## a. Problem statement
Higher-education institutions lose students to dropout, and intervening early needs a way to flag at-risk students.
Using information known at enrolment plus first- and second-semester academic performance, predict whether a student
will **Drop out**, stay **Enrolled** (still studying after the normal duration), or **Graduate**: a 3-class
classification problem. Five classifiers are trained and compared on the same data, and a Streamlit app lets anyone
upload test data, pick a saved model and see its metrics, confusion matrix and classification report.

## b. Dataset description
- **Source:** UCI Machine Learning Repository, *Predict Students' Dropout and Academic Success* (ID 697),
  https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- **Instances:** 4,424 students (Polytechnic Institute of Portalegre, Portugal)
- **Features:** 36 (all numeric / integer-encoded), no missing values
  - Demographic: marital status, nationality, gender, age at enrolment, displaced, international
  - Socio-economic: mother's/father's qualification and occupation, debtor, tuition fees up to date, scholarship holder
  - Academic path: application mode/order, course, previous qualification (and grade), admission grade
  - Performance: curricular units credited / enrolled / evaluated / approved / grade / without evaluation, for the 1st and 2nd semester
  - Macro-economic: unemployment rate, inflation rate, GDP
- **Target:** `Target` ∈ {Dropout (1,421), Enrolled (794), Graduate (2,209)}, so the classes are imbalanced
- **Split:** 80 % train / 20 % test, stratified, `random_state=7` → `test_data.csv` has 885 rows (36 features + Target)

## c. GitHub repository link
https://github.com/PriyanshuKanyal37/ml-assignment2

```
├── app.py               # Streamlit app (upload CSV → choose model → metrics + confusion matrix + report)
├── train_models.py      # trains the 5 models, saves them, writes metrics + test_data.csv
├── requirements.txt
├── test_data.csv        # 20 % hold-out test set used for all reported metrics
├── raw/data.csv         # original UCI dataset (semicolon separated)
└── model/               # 5 saved .joblib models + metrics.csv
```

## d. Models used
All five models are trained on the same 80 % split. Logistic Regression, KNN, Naive Bayes and SVM are sklearn
`Pipeline`s with `StandardScaler`, so the scaler is saved inside each model and the app needs no separate
preprocessing file. Multi-class metrics: Precision / Recall / F1 are **weighted** averages, AUC is **one-vs-rest weighted**.

| Model | Settings |
|---|---|
| Logistic Regression | StandardScaler + LogisticRegression(max_iter=2000) |
| Decision Tree | DecisionTreeClassifier(max_depth=8, min_samples_leaf=10) |
| KNN | StandardScaler + KNeighborsClassifier(n_neighbors=15) |
| Naive Bayes | StandardScaler + GaussianNB |
| Support Vector Machine | StandardScaler + SVC(kernel="rbf", C=1, probability=True) |

### Comparison table (hold-out test set, 885 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.7729** | **0.9031** | **0.7591** | **0.7729** | **0.7608** | **0.6230** |
| Decision Tree | 0.7638 | 0.8720 | 0.7501 | 0.7638 | 0.7506 | 0.6075 |
| KNN | 0.7040 | 0.8413 | 0.6934 | 0.7040 | 0.6777 | 0.5058 |
| Naive Bayes | 0.6768 | 0.8037 | 0.6568 | 0.6768 | 0.6619 | 0.4588 |
| Support Vector Machine | 0.7492 | 0.8954 | 0.7365 | 0.7492 | 0.7327 | 0.5824 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.773, MCC 0.623; rank 1 of 5. Its straight-line boundary works well because more passed subjects and higher marks generally mean a better chance of graduating. Its weak point is that it identifies only 38% of Enrolled students. |
| Decision Tree | Accuracy 0.764, MCC 0.608; rank 2 of 5. It learns if-then rules, such as predicting Dropout when very few semester-2 subjects were passed. Its AUC is lower at 0.872, and deeper trees could overfit; this tree is limited to depth 8. |
| KNN | Accuracy 0.704, MCC 0.506; rank 4 of 5. It compares students by distance, but numeric codes such as course and job IDs do not represent meaningful distances. It predicts Graduate too often and identifies only 23% of Enrolled students. |
| Naive Bayes | Accuracy 0.677, MCC 0.459; rank 5 of 5. It assumes features are independent, even though semester-1 and semester-2 results are closely related. It is the fastest and simplest model, but its independence assumption is a poor fit here. |
| Support Vector Machine | Accuracy 0.749, MCC 0.582; rank 3 of 5, with the second-best AUC at 0.895. It separates classes well, but was used with default settings rather than tuned settings. Its weak point is speed: it was the slowest model. |
| **Overall, winner for this dataset** | **Logistic Regression** is the winner: it performs best across all six metrics and is also simple and fast. |

## Streamlit app
- **Live app:** https://priyanshu-ml-assignment2.streamlit.app
- Features: CSV upload (test data), model selection dropdown, the 6 evaluation metrics, confusion matrix heat-map,
  classification report, row-level predictions, a side-by-side comparison tab and a sample `test_data.csv` download.
- The app only **loads** the saved models from `model/`. It never retrains on the uploaded data.

## Run locally
```bash
pip install -r requirements.txt
python train_models.py      # optional: re-creates model/ and test_data.csv
streamlit run app.py
```
