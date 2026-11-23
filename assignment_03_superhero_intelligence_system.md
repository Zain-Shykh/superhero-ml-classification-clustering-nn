# Superhero Intelligence System — Machine Learning Project
**Can six ability scores tell a hero from a villain?**

---

## Overview of Sections & Marks
* **Section 1:** Instructions
* **Section 2:** Data Preparation (**10 marks**) — Clean two files with fixed, simple rules
* **Section 3 (Task 1):** Classification (**25 marks**) — Decision Tree and KNN predict hero or villain
* **Section 4 (Task 2):** Clustering (**25 marks**) — K-Means and Agglomerative find character groups
* **Section 5 (Task 3):** Neural Network (**25 marks**) — Two small networks predict hero or villain

---

## SECTION 1: Instructions
1. Submit it in the group of 2-3 people.
2. Submit your completed work by **06 December 2026 at 11:59 PM**.
3. Submissions up to 5 minutes late will incur a **10% deduction** in marks.
4. Submissions more than 5 minutes and up to 60 minutes late will incur a **20% deduction** in marks.
5. Submissions more than 1 hour and up to 12 hours late will incur a **50% deduction** in marks.
6. Submissions more than 12 hours late will receive **zero marks**.

---

## SECTION 2: Data Preparation (10 marks)
**Get the data ready before any model sees it**

### How the whole project fits together
* **`train.csv` (Training data)** $\rightarrow$ **Clean** (Same rules for both files: 1. Inspect, 2. Remove duplicates, 3. Mark missing, 4. Remove weak rows) $\rightarrow$ splits into:
  * **Clean Set A** (All characters, `train.csv` only) $\rightarrow$ **Task 2: Clustering** (K-Means and Agglomerative, compared with Silhouette, Davies-Bouldin, Calinski-Harabasz and ARI)
  * **Clean Set B** (Good & Bad only, both files) $\rightarrow$
    * **Task 1: Classification** (Baseline, Decision Tree, KNN)
    * **Task 3: Neural Network** (Network A, Network B)
* **`test.csv` (Final test only)** $\rightarrow$ **Clean** (Same rules: 1–4) $\rightarrow$ **Clean Set B** $\rightarrow$ `test.csv` is opened only once, at the very end of Task 1 and Task 3 $\rightarrow$ **Final ranking of all 5 models on `test.csv`**.

### The dataset: two files, use them as they are
* **Source:** Superheroes dataset · Hugging Face (`huggingface.co/datasets/jrtec/Superheroes`)
* Characters from many publishers (Marvel, DC and others). **Use all of them. Do not filter by publisher.**
* **`train.csv` — the training file:**
  * Train every model
  * Choose settings with cross-validation
  * Make the validation set for the networks
  * Run the clustering (Task 2)
* **`test.csv` — the final exam:**
  * Only for the final scores of Task 1 and Task 3. Open it once, at the end. Never train or tune on it.
* **Directory Structure:**
  ```text
  my_project/
  ├── superhero_project.ipynb
  └── data/
      ├── train.csv
      └── test.csv
  ```
* Read them with `pd.read_csv("data/train.csv")` and `pd.read_csv("data/test.csv")`.
* **Rules:**
  * **Do NOT** merge the two files.
  * **Do NOT** make your own train/test split.

### Which columns you may use
#### USE these columns:
| Column | Role |
| :--- | :--- |
| `intelligence_score` | Input feature (number 0–100) |
| `strength_score` | Input feature |
| `speed_score` | Input feature |
| `durability_score` | Input feature |
| `power_score` | Input feature |
| `combat_score` | Input feature |
| `alignment` | Target: Good / Bad / Neutral (some blank) |
| `name` | Label only — never an input |

#### Do NOT use:
* `overall_score`: made from the 6 scores, so it leaks the answer
* `history_text`, `powers_text`, `superpowers`: text
* `has_...`: power yes/no flags (Every `has_...` column)
* `creator`, `gender`, `height`, `weight`, `colours`, and every other column

> **Note:** Every model in this project uses exactly these 6 inputs — nothing more, nothing less.

### Cleaning steps: same rules, in this order
1. **Inspect:** For each file, show the shape, column names, data types and missing counts (**Table T1**).
2. **Remove duplicates:**
   * Same name twice in a file: keep the first row.
   * A name in both files: remove it from `train.csv`.
3. **Mark missing:** Make the 6 scores numeric. Turn blanks and `0` into `NaN`. A `0` means "unknown", not "no ability".
4. **Remove weak rows:** Drop every character that has **3 or more** of its 6 scores missing.
5. **Fill the gaps:** Fill the remaining `NaN`s with the training median, inside your `Pipeline`.

* **Steps 1–4:** Apply to `train.csv` and `test.csv` in exactly the same way.
* **Step 5:** Training data only.
* Count the rows after every step and report them in **Table T2**.
* Versions with different names (e.g., `"Captain X"` and `"Captain X (Film)"`) are different characters. Keep both.

### Missing values: a worked example
| Character (invented) | intelligence | strength | speed | durability | power | combat | Missing | What happens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Nova Knight | 75 | 40 | 35 | 60 | 55 | 70 | 0 | Keep as it is |
| Iron Moth | 50 | $0 \rightarrow \text{NaN}$ | 25 | $\text{blank} \rightarrow \text{NaN}$ | 30 | 45 | 2 | Keep. The Pipeline fills the 2 gaps with training medians |
| Shadow Pip | $0 \rightarrow \text{NaN}$ | $0 \rightarrow \text{NaN}$ | 10 | $0 \rightarrow \text{NaN}$ | $\text{blank} \rightarrow \text{NaN}$ | 20 | 4 | Remove (3 or more missing) |

* **How the fill works:** Suppose the training medians are $\text{strength} = 45$ and $\text{durability} = 50$. Iron Moth goes into the model as: `50, 45, 25, 50, 30, 45`.
* **Why these rules?**
  * **Median:** not pulled around by a few extreme scores.
  * **Training data only:** the test set must stay unseen.
  * **3+ missing $\rightarrow$ remove:** too much of the row would be guessed.

### Two clean datasets for two kinds of task
After steps 1–4 (both files):

* **Clean Set A $\rightarrow$ Task 2:**
  * **What:** cleaned `train.csv`, every character (Good, Bad, Neutral or blank alignment).
  * **Alignment:** put aside. Look at it only after clustering.
* **Clean Set B $\rightarrow$ Tasks 1 & 3:**
  * **What:** cleaned `train.csv` and `test.csv`, only rows with alignment `Good` or `Bad`.
  * **Target:** `Good` $\rightarrow 0$ (Hero), `Bad` $\rightarrow 1$ (Villain).
  * Report how many `Neutral` and blank rows were removed.
  * Also report in **T1**: how many `0`s and blanks became `NaN` in each of the 6 score columns.

#### Table T2: Rows after each step
| Step | `train.csv` | `test.csv` |
| :--- | :--- | :--- |
| Loaded | $n = ?$ | $n = ?$ |
| After step 2 (duplicates) | $n = ?$ | $n = ?$ |
| After step 4 (weak rows) | $n = ?$ | $n = ?$ |
| Clean Set A | $n = ?$ | not used |
| Clean Set B | $n = ?$ | $n = ?$ |

### Keep the test set locked
* **`train.csv` (learn here):** Fit imputer $\rightarrow$ Fit scaler $\rightarrow$ Fit model $\rightarrow$ Choose settings (CV) $\rightarrow$ produces fitted pipeline.
* **`test.csv` (once, at the end):** Fitted pipeline $\rightarrow$ Transform only (no fitting) $\rightarrow$ Predict $\rightarrow$ Score.
* **Tip:** Put the imputer, scaler and model in one scikit-learn `Pipeline` — it keeps this order safe for you.
* **Four rules:**
  1. Never call `.fit()` or `.fit_transform()` on test data.
  2. Never choose settings by looking at test scores.
  3. Never use `overall_score` — it is made from the 6 inputs.
  4. Clustering never sees `alignment`.

### Rules for every task
#### Seed and tools
* Use `random_state=42` in every function that has it.
* Networks: `tf.keras.utils.set_random_seed(42)`
* Tools: `pandas`, `scikit-learn`, `SciPy`, `matplotlib` (`seaborn` allowed), `TensorFlow`/`Keras`.

#### Fixed colours (hex codes)
* **Hero / Good:** `#1F5FBF`
* **Villain / Bad:** `#D62839`
* **Baseline:** `#9E9E9E`
* **Decision Tree:** `#2E7D32`
* **KNN:** `#F9A825`
* **Network A:** `#6A1B9A`
* **Network B:** `#00838F`
* **Clusters:** `tab10` order (Cluster 1 = 1st colour)

#### Every graph and table
* Title, labelled axes, and a legend if there is more than one colour.
* 1–2 sentences under it: what it shows and what you conclude.
* Number them **G1–G16** and **T1–T11**, exactly as in this brief.

#### Confusion matrix format
* Rows = actual class; Columns = predicted class (`Predicted` $\rightarrow$).
* Order: `Good` (Hero), then `Bad` (Villain).
* Show the count in every cell.

| Actual \ Predicted | Good | Bad |
| :--- | :---: | :---: |
| **Good** | TN | FP |
| **Bad** | FN | TP |

---

## SECTION 3: Task 1 — Classification (25 marks)
**Hero or villain? Predict alignment from the 6 ability scores**

### Task 1 at a glance
* **Inputs (6 scores):** `intelligence_score`, `strength_score`, `speed_score`, `durability_score`, `power_score`, `combat_score`
* **Target:**
  * $0 = \text{Hero}$ (`alignment == Good`)
  * $1 = \text{Villain}$ (`alignment == Bad`)
* **Three Models:**
  * **Baseline:** always predicts the most common class
  * **Decision Tree:** asks yes/no questions
  * **KNN:** votes of the most similar characters
* **Data:** Clean Set B
* **Train on:** all of `train.csv`
* **Test on:** `test.csv`, once
* **Main metric:** Macro-F1

### How the two models decide
* **Decision Tree:**
  * Asks yes/no questions about the scores until it reaches a leaf with an answer (e.g., $\text{combat\_score} \le 50? \rightarrow \text{power\_score} \le 40?$).
  * `max_depth` = the most questions in a row. Deeper trees fit training data better but can overfit.
  * No scaling needed.
* **K-Nearest Neighbours (KNN):**
  * Finds the $k$ characters with the most similar scores and takes a vote (e.g., $k=5$ nearest: 3 villains, 2 heroes $\rightarrow$ predict Villain).
  * Uses distance, so all 6 scores must be on the same scale $\rightarrow$ `StandardScaler` is required.

### Method: step by step
1. **Load:** Clean Set B rows of `train.csv`. $X = 6$ scores, $y = 0/1$.
2. **Pipeline:** Median imputer $\rightarrow$ `StandardScaler` (KNN only) $\rightarrow$ model.
3. **Grid search:** Try every setting with 5-fold stratified CV. Score = macro-F1.
4. **Refit:** Best setting is retrained on all of `train.csv` (`refit=True`).
5. **Test once:** Predict `test.csv` with the refitted model.
6. **Report:** Metrics, confusion matrix, comparison.

* **5-fold cross-validation inside `train.csv`:**
  * Round 1 to Round 5: Each setting is trained 5 times; each time a different fifth (`check`) of `train.csv` checks it while 4 fifths (`learn`) train it.
  * The setting with the highest average macro-F1 wins.
  * `test.csv` is not used at all in this step.

### Exact settings
| Model | Pipeline (in this order) | Search these values | Keep fixed |
| :--- | :--- | :--- | :--- |
| **Baseline** | `DummyClassifier` | none | `strategy="most_frequent"` |
| **Decision Tree** | `SimpleImputer(strategy="median")` $\rightarrow$ `DecisionTreeClassifier` | `max_depth`: `3, 5, 7, 10, None`<br>`min_samples_leaf`: `1, 5, 10`<br>*(15 combinations)* | `criterion="gini"`<br>`random_state=42` |
| **KNN** | `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler` $\rightarrow$ `KNeighborsClassifier` | `n_neighbors`: `3, 5, 7, 9, 11, 15`<br>`weights`: `"uniform", "distance"`<br>*(12 combinations)* | `metric="euclidean"` |

* **Search for both models:**
  ```python
  GridSearchCV(
      pipeline,
      param_grid,
      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
      scoring="f1_macro",
      refit=True
  )
  ```
* **Not allowed:** Class weights, oversampling, undersampling, extra features, tuning on `test.csv`.

### Metrics: start with the confusion matrix
| Actual \ Predicted | Hero (0) | Villain (1) |
| :--- | :--- | :--- |
| **Hero (0)** | **TN** — Hero called hero | **FP** — Hero wrongly called villain |
| **Villain (1)** | **FN** — Villain missed (called hero) | **TP** — Villain caught |

| Metric | In plain words | Formula |
| :--- | :--- | :--- |
| **Accuracy** | Share of all predictions that are right | $\frac{\text{correct}}{\text{all}}$ |
| **Precision** | Of those we called villain, how many really are | $\frac{TP}{TP + FP}$ |
| **Recall** | Of all real villains, how many we found | $\frac{TP}{TP + FN}$ |
| **F1** | One number that balances precision and recall | $\frac{2 \cdot P \cdot R}{P + R}$ |
| **Macro-F1 (MAIN)** | Average of $\text{F1(Hero)}$ and $\text{F1(Villain)}$. Treats both classes equally, so imbalance cannot fool it. | $\frac{\text{F1 Hero} + \text{F1 Villain}}{2}$ |

* Villain ($1$) is the positive class for $TP, FP, FN$ and $TN$. Hero metrics work the same way with the roles swapped.
* Get them with `classification_report(y_test, y_pred, digits=3)`.

### Comparison basis: how to pick the winner
1. **Winner:** Highest test Macro-F1.
2. **Tie-break:** Gap under $0.01$? Higher Villain recall wins.
3. **Useful?:** Only if test Macro-F1 is higher than the Baseline's.
4. **Overfitting?:** $\text{Train Macro-F1} - \text{Test Macro-F1} > 0.10 \rightarrow$ it overfits.

#### Table T5: Task 1 results (all on `test.csv` except the Train column)
| Model | Train Macro-F1 | Accuracy | Precision Hero | Recall Hero | F1 Hero | Precision Villain | Recall Villain | F1 Villain | Macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline | | | | | | | | | |
| Decision Tree | | | | | | | | | |
| KNN | | | | | | | | | |

* **Train Macro-F1** = the final model scored on its own training data (all of `train.csv`). **Macro-F1** column is the main one.

### Task 1: what to hand in
* **Graphs:**
  * **G1 Class counts:** Good and Bad bars for `train.csv` and `test.csv` side by side, with `%` labels.
  * **G2 Confusion matrix — Decision Tree:** Counts in every cell, required format.
  * **G3 Confusion matrix — KNN:** Same format as G2.
  * **G4 Model comparison:** Grouped bars: Accuracy, Macro-Precision, Macro-Recall, Macro-F1 for the 3 models ($y$ from $0$ to $1$).
* **Tables:**
  * **T3:** Class counts and `%` in each file.
  * **T4:** Best settings and best CV macro-F1.
  * **T5:** Results table.
* **Answer in the report:**
  * **Q1:** Which model wins by the rules? Quote the numbers.
  * **Q2:** Did both models beat the Baseline? By how much?
  * **Q3:** Which class is harder to predict? Use the confusion matrices.
  * **Q4:** Does either model overfit? Use the train-test gap.

---

## SECTION 4: Task 2 — Clustering (25 marks)
**Find groups of similar characters — no labels allowed**

### Task 2 at a glance
* **K-Means:** Builds $k$ groups around $k$ centre points.
* **Agglomerative (Ward):** Joins the closest groups, step by step.
* **Data:** Clean Set A (cleaned `train.csv`, all characters)
* **Inputs:** The 6 scores only
* **Never used:** `alignment`, `name` or any other column
* **No test set:** `test.csv` is not used in Task 2

### How the two algorithms build groups
* **K-Means:**
  1. Place $k$ centres.
  2. Each character joins its nearest centre.
  3. Move each centre to the middle of its group.
  4. Repeat 2–3 until nothing changes.
* **Agglomerative (Ward):**
  1. Every character starts as its own group.
  2. Join the two closest groups. Ward = the join that adds the least spread.
  3. Repeat until only one group is left.
  4. Cut the tree to get $k$ groups.

### Method: step by step
1. **Load Clean Set A:** Cleaned `train.csv`, all characters, 6 scores.
2. **Fill & scale:** Median imputer, then `StandardScaler`.
3. **Run K-Means:** For $k = 2$ to $8$, save inertia and Silhouette.
4. **Choose $k$:** Highest Silhouette. Comment on the Elbow.
5. **Agglomerative:** Ward linkage, same $k$ as K-Means.
6. **Score both:** Silhouette, Davies-Bouldin, Calinski-Harabasz, plus ARI.
7. **Renumber:** Order clusters by average score: lowest = Cluster 1.
8. **Explain:** Profiles, names, PCA plots, then the alignment mix.

> **Note:** Alignment is looked at only in step 8 after the clusters already exist.

### Choosing $k$ (the number of groups)
* **The rule:**
  * Final $k =$ the $k$ from $2$ to $8$ with the highest Silhouette Score.
  * Tie (same to 3 decimals) $\rightarrow$ the smaller $k$.
  * Describe what the Elbow shows. It does not change the choice.
  * Use the same $k$ for Agglomerative.
* *(Note on Page 24 example plots: Example data only — your curves will look different. Do not copy those numbers.)*

### Exact settings and cluster numbering
| Step | What to use |
| :--- | :--- |
| **Fill & scale** | `SimpleImputer(strategy="median")`<br>`StandardScaler()`<br>fitted on Clean Set A |
| **K-Means** | `KMeans(n_clusters=k, n_init=10, random_state=42)`<br>for $k = 2, 3, 4, 5, 6, 7, 8$ |
| **Agglomerative** | `AgglomerativeClustering(n_clusters=k, linkage="ward")`<br>$k =$ the final $k$ from K-Means |
| **PCA (plots only)** | `PCA(n_components=2, random_state=42)`<br>fit once on the scaled data; use it for both scatter plots |
| **Dendrogram** | `linkage(X, method="ward")`<br>`dendrogram(..., truncate_mode="lastp", p=30)`<br>draw a dashed line where the tree splits into $k$ groups |
| **Metrics** | all computed on the scaled data |

#### Rename the clusters
* Algorithms number clusters at random. Rename them by the average of the 6 scores ($0\text{--}100$ scale).
* Lowest average = **Cluster 1**, next = **Cluster 2**, etc. (e.g., $\text{avg } 24 \rightarrow \text{Cluster 1}$, $\text{avg } 38 \rightarrow \text{Cluster 2}$, $\text{avg } 61 \rightarrow \text{Cluster 3}$).
* Use `tab10` colour 1 for Cluster 1, colour 2 for Cluster 2 ... in every graph.

### Metrics: judge the groups without labels
* **Silhouette Score** ($-1$ to $1$, $\uparrow$ higher is better):
  * How well each character sits in its own group compared with the nearest other group (`silhouette_score`).
  * **Bands:** $0.71\text{--}1.00$ strong | $0.51\text{--}0.70$ reasonable | $0.26\text{--}0.50$ weak | $\le 0.25$ no real structure
* **Davies-Bouldin Index** ($0$ and up, $\downarrow$ lower is better):
  * How much each group looks like its most similar neighbour group (`davies_bouldin_score`).
* **Calinski-Harabasz Score** ($0$ and up, $\uparrow$ higher is better):
  * Spread between groups compared with spread inside groups. Only compare models on the same data (`calinski_harabasz_score`).
* **Adjusted Rand Index (ARI)** (about $0$ to $1$, agreement):
  * How much K-Means and Agglomerative agree. $1 =$ same groups, $0 =$ no better than chance (`adjusted_rand_score(kmeans_labels, agglo_labels)`).
  * **Bands:** $\ge 0.70$ high | $0.40\text{--}0.69$ moderate | $< 0.40$ low agreement

> **Important:** Never use accuracy, precision or recall for clustering — there are no true labels.

### Comparison basis
1. **Better separated:** The algorithm that wins at least 2 of the 3 metrics ($\text{Silhouette } \uparrow$, $\text{Davies-Bouldin } \downarrow$, $\text{Calinski-Harabasz } \uparrow$).
2. **Similar groups?:** Use the ARI bands: high, moderate or low agreement. Back it up with the profiles.
3. **Easier to interpret:** The algorithm whose profiles (G10, G11) differ more clearly. Give each cluster a short name.

#### Table T6: Clustering metrics
| Algorithm | $k$ | Silhouette $\uparrow$ | Davies-Bouldin $\downarrow$ | Calinski-Harabasz $\uparrow$ | Wins (of 3) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| K-Means | | | | | |
| Agglomerative | | | | | |
| **ARI between the two** | | | | | |

#### Table T7: Cluster profiles (one per algorithm)
| Cluster | Size | INT | STR | SPD | DUR | POW | COM | Name you give |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | | | | | | | | |
| 2 | | | | | | | | |
| 3 | | | | | | | | |

* `INT, STR, SPD, DUR, POW, COM` = averages of the six `*_score` columns on the original $0\text{--}100$ scale.
* **Table T8:** `%` Good, Bad, Neutral and blank in each cluster.

### Task 2: what to hand in
* **Graphs:**
  * **G5 Elbow curve:** inertia vs $k$ ($2\text{--}8$).
  * **G6 Silhouette score vs $k$:** final $k$ marked.
  * **G7 Dendrogram (Ward):** last 30 merges, dashed cut line at $k$.
  * **G8 PCA scatter — K-Means:** axes: PC1, PC2 with `%` variance explained.
  * **G9 PCA scatter — Agglomerative:** same projection and axes as G8.
  * **G10 Cluster profiles — K-Means:** $x = 6$ scores, one bar per cluster, $y =$ average $0\text{--}100$.
  * **G11 Cluster profiles — Agglomerative:** same format as G10.
* **Tables:**
  * **T6:** Clustering metrics + ARI.
  * **T7:** Cluster profiles with names.
  * **T8:** Alignment mix in each cluster (`%`).
* **Answer in the report:**
  * **Q5:** Which $k$ did you choose? What does the Elbow say?
  * **Q6:** Did the two algorithms find similar groups? Use ARI and profiles.
  * **Q7:** Which grouping is better separated, and which is easier to interpret?
  * **Q8:** Describe each cluster in plain words. Is any cluster mostly villains?

---

## SECTION 5: Task 3 — Neural Network (25 marks)
**Predict hero or villain with two small feed-forward networks (MLPs)**

### The two networks
* **Network A — Basic: one hidden layer**
  * **Input:** 6 scores
  * **Hidden:** 8 neurons, ReLU activation
  * **Output:** 1 neuron, sigmoid activation
  * **Total:** **65 trainable parameters**
* **Network B — Deeper: two hidden layers**
  * **Input:** 6 scores
  * **Hidden 1:** 16 neurons, ReLU activation
  * **Hidden 2:** 8 neurons, ReLU activation
  * **Output:** 1 neuron, sigmoid activation
  * **Total:** **257 trainable parameters**

* **Output rule:** Output = probability of Villain. $\ge 0.5 \rightarrow \text{Villain } (1)$, $< 0.5 \rightarrow \text{Hero } (0)$.
* 6 input nodes because there are exactly 6 input scores.

### Method: step by step
1. **Split:** Clean Set B, `train.csv` split into **80% fit part** (the network learns from this) and **20% validation** (stratified):
   ```python
   train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
   ```
2. **Fill & scale:** Fit imputer + scaler on the **80% only**. Apply to validation and test.
3. **Build:** Network A and Network B with the exact settings.
4. **Train:** Same data, same settings for both. Early stopping watches validation loss.
5. **Evaluate:** Train, validation and test results, kept separate (`test.csv` is final test only, open once).
6. **Compare:** Network A vs B, then all five models.

> **Note:** Decision Tree and KNN learned from all of `train.csv`; the networks learn from 80% of it. This is expected — mention it when you compare.

### Exact training settings
*(Same for both networks — Only the hidden layers differ)*

| Setting | Value |
| :--- | :--- |
| **Framework** | TensorFlow / Keras |
| **Seed** | `tf.keras.utils.set_random_seed(42)` |
| **Hidden activation** | ReLU |
| **Output layer** | 1 neuron, sigmoid |
| **Loss** | `binary_crossentropy` |
| **Optimizer** | Adam, learning rate `0.001` |
| **Batch size** | `32` |
| **Max epochs** | `200` |
| **Early stopping** | `EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)` |
| **Decision threshold** | `0.5` ($\text{probability} \ge 0.5 \rightarrow \text{Villain}$) |

* **Not allowed:**
  * Dropout or extra layers/neurons
  * Class weights or resampling
  * Changing any setting above
  * Tuning on `test.csv`
* Small run-to-run differences in network results are normal and accepted.

### Reading loss curves
* **Good fit:** Both lines (Training loss & Validation loss) fall and stay close together.
* **Overfitting:** Training loss keeps falling, validation loss turns up. Early stopping keeps the best epoch.
* **Underfitting:** Both stay high and flat. The network is not learning enough.
* **Requirements for G13 and G14:** $x =$ epoch, $y =$ binary cross-entropy loss, two lines (training, validation) and a dashed vertical line at the best epoch (lowest validation loss). Report the best epoch in **T9**.

### Metrics
| Metric | What it tells you | Report on |
| :--- | :--- | :--- |
| **Accuracy** | Share of predictions that are right | train, validation, test |
| **Macro-F1 (MAIN)** | Average F1 of Hero and Villain | train, validation, test |
| **Precision, Recall, F1 per class** | How well each class is handled | test |
| **Confusion matrix** | Which class gets mixed up | test |
| **Log loss** | How good the predicted probabilities are; lower is better | validation, test |

#### Table T10: Results by split
| Network | Split | Accuracy | Macro-F1 | Log loss |
| :--- | :--- | :--- | :--- | :--- |
| Network A | train | | | |
| Network A | validation | | | |
| Network A | test | | | |
| Network B | train | | | |
| Network B | validation | | | |
| Network B | test | | | |

* Same definitions as Task 1.
* Use `classification_report` and `log_loss` from `sklearn.metrics`.
* Put the per-class precision, recall and F1 on `test.csv` in **T10** or right below it.

### Comparison basis
1. **Network A vs Network B:**
   * **Winner:** higher test Macro-F1.
   * Say *"the deeper network improved results"* only if Network B's test Macro-F1 is at least $0.01$ higher.
   * **Overfitting:** $\text{train Macro-F1} - \text{test Macro-F1} > 0.10$, or validation loss rises while training loss falls.
2. **All five models on `test.csv`:**
   * Rank by test Macro-F1.
   * Gap under $0.01 \rightarrow$ higher Villain recall ranks higher.
   * *"Beats the baseline"* $=$ test Macro-F1 higher than the Baseline's.

#### Table T11: Final ranking
| Rank | Model | Test accuracy | Test Macro-F1 | Villain recall | Beats baseline? (yes/no) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

### Task 3: what to hand in
* **Graphs:**
  * **G12 Architecture diagrams — A and B:** any drawing tool; the only graph not made by code.
  * **G13 Loss curves — Network A:** training vs validation, best epoch marked.
  * **G14 Loss curves — Network B:** same format as G13.
  * **G15 Confusion matrices — A and B:** required format, counts shown.
  * **G16 Final comparison:** grouped bars: Accuracy and Macro-F1 for all 5 models on `test.csv`.
* **Tables:**
  * **T9:** Network settings + best epoch.
  * **T10:** Train / validation / test results.
  * **T11:** Final ranking of all 5 models.
* **Answer in the report:**
  * **Q9:** Did Network B beat A by at least $0.01$ Macro-F1? Why or why not?
  * **Q10:** Do the loss curves show over- or underfitting? Which epoch was best?
  * **Q11:** Did either network beat the Decision Tree and KNN? Give numbers.
  * **Q12:** Name two limitations of this dataset or experiment.