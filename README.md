# Electrical grid stability classification

## a. Problem statement

A four-node star grid, one electricity producer and three consumers, can settle into a stable state or drift into linear instability when each node reacts locally to price. Every node is described by a reaction time, a nominal power, and a price-elasticity coefficient. The task is to predict the operating state, stable or unstable, from those twelve measurements.

The numeric stability margin is not used as an input. That margin is the quantity the label was cut from, so including it would leak the answer. The useful output is a warning that the current operating point is unstable, early enough for the controller to change course.

## b. Dataset description

The data is the Electrical Grid Stability Simulated Data set from the UCI Machine Learning Repository (donor: Vadim Arzamasov). It comes from a simulation of decentralized demand response on a four-node star network, the setting studied by Schafer, Grabow, Auer, Kurths, Witthaut, and Timme in "Taming instabilities in power grid networks by decentralized control" (2016).

The file has 10,000 rows and no missing values. 6,380 rows (63.8 percent) are unstable and 3,620 rows (36.2 percent) are stable. Twelve columns are used as inputs.

| Feature | Meaning |
| --- | --- |
| tau1 | Reaction time of the producer, in seconds |
| tau2, tau3, tau4 | Reaction time of the three consumers, in seconds |
| p1 | Nominal power of the producer. It equals the negative of p2 + p3 + p4 |
| p2, p3, p4 | Nominal power of the three consumers |
| g1, g2, g3, g4 | Price-elasticity coefficient of each node |

Two checks were made before modelling. On every row, p1 + p2 + p3 + p4 is zero up to numerical noise (the largest absolute sum is about 5.8e-15), so p1 is determined by the other three power terms. It is kept so the model still sees all twelve inputs. On every row, stab is positive exactly when stabf is unstable, so stab was removed and only stabf is kept as the target.

The split is stratified, 80 percent train and 20 percent test, with random_state 42. That is 8,000 training rows and 2,000 test rows. The test rows are stored in test_data.csv with the twelve inputs and stabf. Hyperparameters were chosen only on the training rows. The numbers below are from that held-out file.

Precision, recall, and F1 use unstable as the positive class. AUC is the ROC area for that same class. Accuracy and Matthews correlation use both classes.

## c. Github Repository Link

Add the public repository URL here after the folder is pushed:

`https://github.com/<your-username>/<your-repo>`

The repository contains app.py, train_models.py, evaluation.py, requirements.txt, runtime.txt, README.md, test_data.csv, the full data file under data/, and the five saved models under model/.

## d. Models used

All five classifiers were fit on the same training split. Logistic regression, k-nearest neighbours, Gaussian naive Bayes, and the support vector machine sit behind a StandardScaler that was fit on the training rows only. The decision tree uses the raw inputs. Logistic regression, the tree, k-nearest neighbours, and naive Bayes had their settings chosen by 3-fold stratified search on training F1. The support vector machine kernel was chosen by the same F1 score on a 2,500-row stratified subset of the training data, then the chosen kernel was refit on all 8,000 training rows. The saved files are scored as they are. The app does not fit them again.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8195 | 0.8921 | 0.8396 | 0.8864 | 0.8624 | 0.6026 |
| Decision Tree | 0.8520 | 0.8912 | 0.8907 | 0.8754 | 0.8830 | 0.6819 |
| KNN | 0.8620 | 0.9548 | 0.8401 | 0.9679 | 0.8995 | 0.7005 |
| Naive Bayes | 0.8445 | 0.9210 | 0.8395 | 0.9350 | 0.8847 | 0.6570 |
| Support Vector Machine | 0.9635 | 0.9949 | 0.9666 | 0.9765 | 0.9715 | 0.9208 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Weakest of the five, with MCC 0.6026. Training accuracy (0.8143) and test accuracy (0.8195) sit together, so the model is not memorising. The search kept the stronger penalty, C = 0.1. The largest scaled coefficients are on the reaction times and the elasticity terms, all positive and close in size (about 0.82 to 0.86). A straight cut cannot follow the stability boundary. The linear SVM probe on training rows told the same story: F1 0.8665, well behind the RBF probe. |
| Decision Tree | Depth was capped at 16 and each leaf must hold at least 5 rows. The fitted tree stopped at depth 13 with 415 leaves. Test accuracy is 0.8520, but training accuracy is 0.9510, so some of those leaves do not travel. It has the best precision of the models other than the SVM (0.8907) and the lowest recall (0.8754). Importances are spread across tau and g, each about 0.10 to 0.14, while the four power features together contribute under 0.03. AUC (0.8912) is essentially tied with logistic regression: the extra leaves improved the hard labels more than the ranking. |
| KNN | k = 15 with distance weights, after scaling. Second on accuracy (0.8620), AUC (0.9548), F1 (0.8995), and MCC (0.7005). Recall is high (0.9679) and precision is only 0.8401, so it calls unstable too often. Training accuracy is 1.0000 because a distance-weighted vote includes the point itself at distance zero. Neighbours are informative on new rows, which is why the AUC stays high, but the classifier does not generalise the way the margin model does. |
| Naive Bayes | Gaussian naive Bayes with variance smoothing 1e-9, the value the search kept. The class prior matches the base rate, about 0.362 stable and 0.638 unstable. AUC is 0.9210, better than the linear model and the tree, so the class-conditional densities still rank cases. MCC is only 0.6570. The independence assumption is false in this table: p1 is exactly determined by the other power terms, and stability depends on how reaction time and elasticity combine. Training accuracy (0.8303) and test accuracy (0.8445) stay close, so the limit is the model, not overfitting. |
| Support Vector Machine | RBF kernel, C = 1, gamma = scale. The kernel was chosen on training rows only: the RBF probe reached F1 0.9394 and the linear probe 0.8665. On the holdout it leads every column, with accuracy 0.9635, AUC 0.9949, precision 0.9666, recall 0.9765, F1 0.9715, and MCC 0.9208. Training accuracy is 0.9749, about one point above the test figure. 2,178 of the 8,000 training rows are support vectors (1,084 stable and 1,094 unstable), so the decision uses a subset of the sample rather than memorising it. |
| Overall, Winner for your dataset? | Support Vector Machine. It is first on accuracy, AUC, precision, recall, F1, and MCC. The MCC gap to the next model, KNN at 0.7005, is about 0.22, so the result does not depend on which one of these metrics is preferred. The linear models and the tree stop around MCC 0.60 to 0.68 because the stable region is curved in the reaction-time and elasticity features. The RBF kernel is what fits that shape. |

## How to run

From this folder, with the packages in requirements.txt installed:

```
python train_models.py
streamlit run app.py
```

train_models.py reads data/grid_stability.csv, refits the five models, and rewrites test_data.csv and model/. The Streamlit app loads the saved files. Upload a CSV that has the twelve inputs and stabf, or press "Score the bundled test_data.csv". If the upload still contains stab, the app drops that column and says so. Metrics, the comparison table, the confusion matrix, and the classification report are calculated on the file in front of you. The model is not trained again.

## Deploy

1. Push this folder to a GitHub repository. Put the URL in section c above.
2. Sign in at https://streamlit.io/cloud with that GitHub account.
3. Create a new app from the repository, branch main, entry point app.py.
4. Open the live URL and score test_data.csv. Paste that URL into the submission PDF.

The submission PDF also needs one screenshot of this training script running on the BITS Virtual Lab. Run `python train_models.py` there and capture the metrics table. The data file is already in data/, so the lab does not need to download it.
