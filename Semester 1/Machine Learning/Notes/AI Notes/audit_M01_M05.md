# COMPREHENSIVE LECTURE AUDIT REPORT: MODULES 01–05
**Course**: AIML ZG565 Machine Learning — BITS Pilani (Work Integrated Learning Programmes)  
**Academic Term**: First Semester 2026–2027  
**Auditor**: Swarm Audit Worker (`transcript-early`)  
**Audit Scope**: Descriptive Notes Modules 01–05 vs. Live Lecture Recordings (Sessions 1–8, Prof. Sharon Christa) and Faculty Annotated PPTs (Prof. Pritee Narendra Raole / Dr. Pritee Parwekar)  
**Date of Audit**: November 2026  
**Target Notes Under Audit**:
- `Module_01_Introduction_and_Designing_a_Learning_System.md`
- `Module_02_Mathematical_Preliminaries_and_ML_Workflow.md`
- `Module_03_Linear_Models_for_Regression.md`
- `Module_04_Linear_Models_for_Classification_and_Logistic_Regression.md`
- `Module_05_Decision_Trees_and_Information_Theory.md`

---

## 1. EXECUTIVE SUMMARY & CRITICAL FINDINGS

This audit evaluated the current Module 01 through Module 05 descriptive notes against the ground-truth audio/video transcripts of 8 live lectures delivered by Prof. Sharon Christa (July 26 – September 13, 2026) and the annotated lecture presentations by Prof. Pritee Narendra Raole (Dr. Pritee Parwekar).

### The Three Most Critical Gaps Found Across Modules 01–05:

1. **Textbook Hallucination & Wrong Pedagogical Core in Module 01**:
   Current Module 01 notes dedicate over 150 lines to Tom Mitchell's 4-component Checkers learning architecture (Performance System, Critic, Generalizer, Experiment Generator) and an LMS weight update derivation with 60 lines of Python code. In actual lectures, Checkers was mentioned only as a 2-line $<P, T, E>$ example. The 4-component architecture and LMS derivation were **never taught in class**. Furthermore, notes falsely attribute Session 1 (July 26) to technical ML instruction, whereas Prof. Sharon Christa's Session 1 was purely administrative orientation; technical Module 1 was taught in Session 2 (August 2) and in Prof. Pritee's parallel Session 1 deck.

2. **Omission of Core Data Preprocessing & Outlier Mechanics in Module 02**:
   Current Module 02 notes dwell heavily on graduate-level linear algebra and multivariate calculus (SVD, Eigen-decomposition, Jacobian, Hessian matrices, Box-Cox, MICE imputation) that were explicitly cited as external prerequisites or never taught. Meanwhile, the actual lecture core was completely omitted: attribute types (Nominal, Ordinal, Interval, Ratio properties), data characteristics (Dimensionality, Sparsity, Resolution), data types (Relational, Transactional, Document, Graph, Spatial, Temporal), and the **exact worked numerical outlier detection problems** (Interquartile Range with bounds $[Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR]$ finding outlier 22 on the lecture dataset, and the 3-Sigma Rule).

3. **Absence of Canonical Lecture Worked Examples & Exam Heuristics in Modules 03, 04, and 05**:
   The current notes omit the central numerical demonstrations solved step-by-step in class:
   - *Module 03*: The Meal-Tip 1D OLS regression calculation ($\hat{\beta}_1 = 0.1462, \hat{\beta}_0 = -0.8188$), the step-by-step matrix calculus derivation of the Normal Equation, and the critical exam comparison determining when to use Normal Equation ($O(d^3)$ inversion, $d < 10,000$) versus Gradient Descent ($O(knd)$, $d > 10^5$).
   - *Module 04*: The Sentiment Analysis logit calculation, the CGPA/IQ $\to$ Job Offer 1-step gradient update calculation, the geometric proof that weight vector $w$ is strictly orthogonal to the decision boundary, and step-by-step ROC threshold sweep construction. Notes instead waste 70 lines on Fisher's Linear Discriminant Analysis (LDA), which was **never taught**.
   - *Module 05*: The canonical **Play Tennis dataset** (14 rows, 4 attributes) solved across Session 7 with complete Information Gain calculations ($IG(Outlook)=0.247$), the **Classification Error** impurity metric ($1 - \max p_i$), and the continuous attribute threshold search algorithm (sorting, label-change boundary midpoints, candidate Gini minimization at threshold 97.5). Notes instead invent an ad-hoc loan default dataset and delve into post-pruning math not covered in lectures.

### Critical Exam Policy Alert from Prof. Sharon Christa:
> *"Remember you don't have any Python code based questions! No code based questions will come for exam and you can actually practice only numerical questions and things like that... given some values how to calculate confusion matrix, tree splitting information gain, Gini index, weight adjustment in gradient descent, or justify whether to use gradient descent vs normal equation."*  
> — Prof. Sharon Christa (Session 8 transcript, lines 1–3)

The mid-semester examination strictly covers up to **Slide 45 of Decision Trees (Session 8)**. All extensive Python implementation blocks across the current notes represent textbook/AI additions that distract from the high-stakes manual numerical problems tested on exams.

---

## 2. DETAILED AUDIT: MODULE 01 — INTRODUCTION & LEARNING SYSTEMS

### 2.1 Lecture Mapping & Primary Faculty
- **Prof. Sharon Christa**:
  - *Session 1 (2026-07-26)*: Orientation only. Course handout, evaluation scheme, Taxila portal walk-through. No ML technical concepts covered.
  - *Session 2 (2026-08-02)*: Core Module 01 coverage: AI vs ML vs Data Science, Arthur Samuel definition, Supervised/Unsupervised/RL/Semi-Supervised paradigms, batch vs online learning, instance-based vs model-based learning, practical industry examples.
- **Prof. Pritee Narendra Raole (Dr. Pritee Parwekar)**:
  - *Annotated PPT Session 1 (2026-07-26, 82 slides)*: Tom Mitchell $<P,T,E>$ framework, 4 canonical tasks, Traditional rule-based vs ML spam filtering, Why ML? (Hinton digit 2, Mars navigation, Biometrics), Employability dataset, Car price prediction dataset.

### 2.2 What Was ACTUALLY Taught in Lectures
1. **Definitions of Machine Learning**:
   - **Arthur Samuel (1959)**: *"Field of study that gives computers the ability to learn without being explicitly programmed."*
   - **Tom Mitchell (1997)**: Well-defined learning task $<P, T, E>$: *"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."*
2. **The Four Canonical $<P, T, E>$ Formulations**:
   - *Task 1: Handwritten Word Recognition*: $T$ = recognizing handwritten words; $P$ = percentage of words correctly classified; $E$ = database of human-labelled images of handwritten words.
   - *Task 2: Spam Filtering*: $T$ = categorize email messages as spam or ham; $P$ = percentage of emails correctly classified; $E$ = database of emails with human labels.
   - *Task 3: Playing Checkers*: $T$ = playing checkers; $P$ = percentage of games won against opponents; $E$ = games played against itself.
   - *Task 4: Autonomous Driving (ALVINN)*: $T$ = drive on public four-lane highways using vision sensors; $P$ = average distance travelled before error (as judged by human driver); $E$ = sequence of images and steering commands recorded while observing human driver.
3. **Traditional Programming vs. Machine Learning Paradigm**:
   - Traditional: $\text{Data} + \text{Program} \to \text{Computer} \to \text{Output}$. Spam filter required human engineers to manually craft long lists of brittle keywords ("4U", "free", "credit card").
   - Machine Learning: $\text{Data} + \text{Output} \to \text{Computer} \to \text{Program}$. Learns statistical distributions and automatically adapts to changing vocabulary.
4. **Why ML / When to Use ML (Pedagogical Drivers)**:
   - Human expertise does not exist (e.g., navigating rovers on Mars).
   - Humans cannot explain their expertise (e.g., speech/face recognition, biometrics). Geoffrey Hinton's famous handwritten digit insight: *"It is very hard to say what makes a 2 — what distinguishes a 2 from a 7 mathematically? Our brain does it subconsciously."*
   - Models must adapt dynamically to personalized/changing environments (e.g., personalized medicine, dynamic pricing, Netflix recommendation).
   - Hidden relationships in vast data exceeding human cognitive limits (e.g., medical diagnostics, DNA sequencing).
5. **Taxonomy of Learning Systems**:
   - **Supervised Learning**: Features $X$ and ground-truth targets $y$. Subdivided into:
     - *Classification*: Target $y$ is discrete/categorical (e.g., spam vs ham, disease vs healthy, employability yes vs no).
     - *Regression*: Target $y$ is continuous/numerical (e.g., used car price, house price, stock market index).
   - **Unsupervised Learning**: Unlabelled data $\{x^{(i)}\}$. Tasks:
     - *Clustering*: Grouping similar objects (customer segmentation).
     - *Dimensionality Reduction*: Compressing features while preserving variance (PCA).
     - *Anomaly Detection*: Identifying rare outliers (credit card fraud, equipment failure).
     - *Association Rule Mining*: Market basket analysis (items bought together).
   - **Semi-Supervised Learning**: Small amount of labelled data combined with vast unlabelled data (e.g., medical imaging, web page classification).
   - **Reinforcement Learning**: Agent interacting with an Environment via States, Actions, and Rewards.
   - **Operational Learning Paradigms**:
     - *Batch (Offline) Learning*: Model trained on static batch; retrained from scratch periodically.
     - *Online (Streaming) Learning*: Model updates incrementally as new data arrives in streams.
     - *Instance-Based Learning*: Memorizes training instances and predicts via similarity metrics (k-NN).
     - *Model-Based Learning*: Fits parameterized mathematical function to data (linear regression, neural nets).

### 2.3 What is MISSING or INCOMPLETE in Current Module 01 Notes
- **Missing Samuel Definition**: Arthur Samuel's foundational 1959 definition was prominently taught but omitted in notes.
- **Missing The 4 $<P, T, E>$ Canonical Formulations**: While handwriting and checkers are mentioned, the exact BITS lecture formulations for ALVINN autonomous driving and spam filtering are missing.
- **Missing Traditional vs. ML Rule-Based Spam Comparison**: The core introductory slide comparing rule maintenance overhead to automated pattern learning is missing.
- **Missing "Why ML?" Pedagogical Drivers**: Hinton's handwritten digit 2 paradox and the Mars rover/biometrics examples are missing.
- **Missing Operational Taxonomy**: Batch vs Online learning and Instance-Based vs Model-Based learning are completely missing from Module 01 notes.
- **Missing Lecture Tabular Datasets**:
  - *Employability Prediction Table*: 5 rows with features (CGPA, Communication Skills, Aptitude Skills, Programming Skills) $\to$ Target (Job Offered: Yes/No).
  - *Used Car Price Prediction Table*: Features (Brand, Mfg Year, Engine Capacity, Mileage, Distance Travelled, Cab?) $\to$ Continuous Target (Price in Rs.).

### 2.4 What is WRONG or Oversimplified in Current Notes
- **False Session Attribution**: Header claims Session 1 (2026-07-26) covered this technical content under Prof. Pritee Raole. While Prof. Pritee's slides were dated July 26, the live lecture transcript for Session 1 under Prof. Sharon Christa was purely orientation. Technical delivery occurred in Session 2.
- **Massive Textbook Hallucination**: Current notes allocate ~150 lines to Mitchell's 4-component Checkers Architecture:
  1. Complete board representation ($x_1$ to $x_6$ pieces and kings).
  2. Linear evaluation function: $\hat{V}(b) = w_0 + \sum_{i=1}^6 w_i x_i$.
  3. Critic and Generalizer derivation of training values $V_{train}(b) \leftarrow \hat{V}(Successor(b))$.
  4. LMS weight update rule derivation: $w_i \leftarrow w_i + \eta (V_{train}(b) - \hat{V}(b)) x_i$.
  5. 60-line Python implementation class `CheckersLMSLearner`.  
  **Fact**: This entire block was synthesized from Tom Mitchell's 1997 textbook Chapter 1. **It was never mentioned, derived, or assigned in lectures**. The professor spent less than 30 seconds on Checkers as a single slide example of $\langle P, T, E \rangle$.

---

## 3. DETAILED AUDIT: MODULE 02 — MATHEMATICAL PRELIMINARIES & ML WORKFLOW

### 3.1 Lecture Mapping & Primary Faculty
- **Prof. Pritee Narendra Raole (Dr. Pritee Parwekar)**:
  - *Annotated PPT Session 2 CS2 (2026-08-09, 51 slides)*: Complete ML data engineering & preprocessing curriculum, attribute types, data quality, IQR outlier exercise, 3-sigma rule, stratified sampling, normalization vs standardization, Inductive Learning Hypothesis.
- **Prof. Sharon Christa**:
  - *Session 3 (2026-08-09)*: Data types, attribute scale taxonomy, dimensionality, sparsity, missing values, noise vs outliers.
  - *Session 4 (2026-08-16)*: Sampling strategies, class imbalance, feature scaling formulas, discretization (equal-width/equal-depth), feature engineering taxonomy.

### 3.2 What Was ACTUALLY Taught in Lectures
1. **The 5-Stage Machine Learning Pipeline**:
   - Stage 1: Domain Understanding & Problem Formulation.
   - Stage 2: Data Collection, Integration, and Cleansing.
   - Stage 3: Data Preprocessing & Feature Engineering.
   - Stage 4: Model Training & Parameter Optimization.
   - Stage 5: Evaluation, Validation, Deployment & Monitoring.
2. **Data Objects and Attribute Types (Stevens' Measurement Scale Taxonomy)**:
   - **Nominal Attributes**: Categorical labels without ranking or distance. Permissible operations: equality ($=$ or $\neq$). Central tendency: Mode. Examples: Customer ID, Eye Color, Marital Status, Zip Code.
   - **Ordinal Attributes**: Categorical labels with meaningful order/ranking, but intervals between ranks are non-quantifiable or unequal. Permissible operations: comparison ($<, >, \le, \ge$). Central tendency: Median, Percentiles. Examples: Service Rating (1 to 5), Academic Grade (A, B, C), Size (Small, Medium, Large).
   - **Interval Attributes**: Numerical attributes with meaningful, equal distances between values, but **no true/absolute zero** point (zero is arbitrary). Permissible operations: addition and subtraction ($+, -$). Central tendency: Mean, Standard Deviation. Ratio operations ($\times, \div$) are meaningless. Examples: Calendar Dates, Temperature in Celsius or Fahrenheit ($20^\circ\text{C}$ is not twice as hot as $10^\circ\text{C}$).
   - **Ratio Attributes**: Numerical attributes with an **absolute, non-arbitrary zero** point (representing total absence of the quantity). Permissible operations: all arithmetic operations ($+, -, \times, \div$). Central tendency: Geometric Mean, Harmonic Mean. Examples: Age, Salary, Weight, Length, Temperature in Kelvin ($20\text{K}$ is twice as warm as $10\text{K}$).
   - **Discrete vs. Continuous Attributes**: Countably finite or integer sets vs. uncountably infinite real values ($\mathbb{R}$).
3. **Important Characteristics of Data Sets**:
   - *Dimensionality*: Number of attributes. High dimensionality causes the **Curse of Dimensionality** (exponential volume expansion, data sparsity, distance metric breakdown where all pairwise Euclidean distances converge).
   - *Sparsity*: Only a tiny fraction of feature entries contain non-zero values (e.g., document-term matrices, recommender user-item matrices).
   - *Resolution*: Granularity or scale of measurement (temporal or spatial).
   - *Size/Scale*: Volume of data objects determining memory constraints and computational complexity.
4. **Data Types Taxonomy**:
   - Relational/Record Data (tables of objects and attributes).
   - Transactional Data (market basket logs where each transaction contains an arbitrary set of items).
   - Document Data (text documents represented as term-frequency vectors).
   - Graph and Network Data (social networks, molecular structures, web link graphs).
   - Spatial and Spatio-Temporal Data (GIS coordinates, weather maps).
   - Temporal / Time-Series / Sequence Data (stock prices, sensor readings over time, DNA sequences).
5. **Data Quality Dimensions & Anomalies**:
   - *Dimensions*: Accuracy, Completeness, Consistency, Timeliness, Interpretability.
   - *Noise*: Random distortion or error introduced during measurement/transmission; extraneous values modifying true attribute values.
   - *Outliers*: Data objects whose characteristics diverge drastically from the predominant data distribution. Can be noise, or can represent critical genuine events (e.g., credit card fraud).
   - *Missing Values*: Mechanisms (equipment malfunction, non-response) and handling techniques (ignore tuple, global constant replacement, central tendency imputation [mean/median/mode], prediction models).
   - *Duplicate Data*: Exact or near-identical records requiring deduplication.
6. **Cleansing & Outlier Detection Algorithms**:
   - **Interquartile Range (IQR) Method**:
     - Sort data in ascending order.
     - Calculate $Q_1$ (25th percentile) and $Q_3$ (75th percentile).
     - Calculate $IQR = Q_3 - Q_1$.
     - Define Lower Bound $= Q_1 - 1.5 \times IQR$.
     - Define Upper Bound $= Q_3 + 1.5 \times IQR$.
     - Values falling outside $[\text{Lower Bound}, \text{Upper Bound}]$ are flagged as outliers.
   - **Three-Sigma Rule (Normal Distribution)**:
     - Assumes data follows Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$.
     - 99.7% of legitimate observations fall within $[\mu - 3\sigma, \mu + 3\sigma]$.
     - Values $< \mu - 3\sigma$ or $> \mu + 3\sigma$ are designated outliers.
7. **Instance Partitioning & Sampling**:
   - Simple Random Sampling with/without replacement.
   - Subsampling independence violation (e.g., sampling Iris dataset randomly can completely omit rare classes or distort class priors).
   - **Stratified Sampling**: Preserves relative class proportions across train, validation, and test splits.
8. **Feature Transformation & Scaling**:
   - **Min-Max Normalization (Rescaling)**:
     $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} (new\_max - new\_min) + new\_min$$
     Typically maps features to $[0, 1]$. Highly sensitive to extreme outliers.
   - **Z-Score Standardization**:
     $$z = \frac{x - \mu}{\sigma}$$
     Centers data around mean 0 with unit variance ($\sigma=1$). Robust to moderate outliers.
9. **Discretization and Encoding**:
   - *Equal-Width Binning*: Divides range $(x_{\max} - x_{\min})$ into $k$ equal intervals. Sensitive to outliers.
   - *Equal-Frequency (Depth) Binning*: Distributes approximately equal number of instances into each bin.
   - *One-Hot Encoding*: For nominal attributes without order (creates $k$ binary indicator features).
   - *Label / Ordinal Encoding*: Maps ordinal ranks to integers ($0, 1, 2, \dots$).
10. **Inductive Learning Hypothesis (Tom Mitchell Ch 2)**:
    - *"Any hypothesis found to approximate the target function well over a sufficiently large set of training examples will also approximate the target function well over other unobserved examples."*
    - Representation of hypothesis space: Vector of attribute constraints, e.g., $\langle Sunny, Warm, ?, Strong, ?, ? \rangle$.

### 3.3 What is MISSING or INCOMPLETE in Current Module 02 Notes
- **Missing Attribute Scale Hierarchy (Stevens' Taxonomy)**: Nominal, Ordinal, Interval, Ratio properties and their permissible mathematical operations are **completely absent**.
- **Missing Characteristics of Data**: Dimensionality, Sparsity, Resolution, and Size are omitted.
- **Missing Data Types Classification**: Relational, Transactional, Document, Graph, Spatial, and Temporal data types are omitted.
- **Missing Outlier Formulas**: Neither the IQR formula ($Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR$) nor the 3-Sigma rule ($[\mu - 3\sigma, \mu + 3\sigma]$) are presented in notes.
- **Missing Discretization**: Equal-width vs Equal-depth binning methods are missing.
- **Missing One-Hot vs Label Encoding**: Not covered in current notes.
- **Missing Inductive Learning Hypothesis**: The foundational theoretical quote and hypothesis representation are completely missing.

### 3.4 What is WRONG or Oversimplified in Current Notes
- **Extraneous Advanced Mathematics**: Current notes dedicate ~140 lines to topics that were never lectured:
  - Singular Value Decomposition (SVD): $A = U \Sigma V^T$ with pseudo-inverse calculation.
  - Full Jacobian and Hessian matrix calculus.
  - Box-Cox power transform: $y^{(\lambda)}$.
  - Multivariate Imputation by Chained Equations (MICE).  
  *Lecture Reality*: Slide 8 of Prof. Pritee's deck lists "Linear algebra: vector/matrix manipulations" under **Prerequisites**. The professors never derived SVD, Hessians, or MICE in class. Including them creates false expectations while omitting the actual lecture topics.

### 3.5 Key Worked Numerical Examples from Lecture Missing from Notes
#### Worked Example 1: Interquartile Range Outlier Detection (Prof. Pritee CS2, Slide 32)
*Problem Statement*: Given the univariate dataset representing sensor readings:
$$\text{Data} = [10, 12, 11, 15, 11, 14, 13, 17, 12, 22, 14, 11]$$
Find all outliers using the Interquartile Range (IQR) method.

*Step-by-Step Solution*:
1. **Sort the array in ascending order**:
   $$[10, 11, 11, 11, 12, 12, 13, 14, 14, 15, 17, 22] \quad (N = 12)$$
2. **Calculate Median ($Q_2$)**:
   $$\text{Median} = \frac{\text{Value}_6 + \text{Value}_7}{2} = \frac{12 + 13}{2} = 12.5$$
3. **Calculate First Quartile ($Q_1$, 25th percentile)**:
   Median of lower half $[10, 11, 11, 11, 12, 12] \implies Q_1 = 11$.
4. **Calculate Third Quartile ($Q_3$, 75th percentile)**:
   Median of upper half $[13, 14, 14, 15, 17, 22] \implies Q_3 = \frac{14 + 15}{2} = 14.5$.
5. **Calculate IQR**:
   $$IQR = Q_3 - Q_1 = 14.5 - 11 = 3.5$$
6. **Calculate Detection Bounds**:
   $$\text{Lower Bound} = Q_1 - 1.5 \times IQR = 11 - 1.5 \times 3.5 = 11 - 5.25 = 5.75$$
   $$\text{Upper Bound} = Q_3 + 1.5 \times IQR = 14.5 + 1.5 \times 3.5 = 14.5 + 5.25 = 19.75$$
7. **Flag Outliers**:
   All points $< 5.75$ or $> 19.75$ are outliers.
   $$\text{Identified Outlier} = 22$$

#### Worked Example 2: Three-Sigma Rule Outlier Detection (Prof. Pritee CS2, Slide 33)
*Problem Statement*: Assume feature measurements follow a normal distribution with mean $\mu = 50$ and standard deviation $\sigma = 5$. Determine the threshold boundaries for outlier detection.
*Step-by-Step Solution*:
1. Under empirical rule, 99.7% of legitimate data lies within $[\mu - 3\sigma, \mu + 3\sigma]$.
2. $\text{Lower Bound} = \mu - 3\sigma = 50 - 3(5) = 50 - 15 = 35$.
3. $\text{Upper Bound} = \mu + 3\sigma = 50 + 3(5) = 50 + 15 = 65$.
4. Any measurement $x < 35$ or $x > 65$ is flagged as an outlier.

#### Worked Example 3: Bank Customer Case Study (Attribute Classification, CS2 Slide 16)
*Dataset*:
- `Name`: Nominal
- `Gender`: Nominal (binary)
- `Service Rating (1-10)`: Ordinal
- `Is Priority Customer?`: Nominal (binary indicator)
- `Card Type (Platinum/Gold)`: Ordinal
- `Credit Score`: Interval (arbitrary scale, no true zero)
- `Is Multiple Account Holder`: Nominal (binary)
- `Income Level (Upper/Middle/Lower)`: Ordinal
- `Region`: Nominal

---

## 4. DETAILED AUDIT: MODULE 03 — LINEAR MODELS FOR REGRESSION

### 4.1 Lecture Mapping & Primary Faculty
- **Prof. Sharon Christa**:
  - *Session 3 (2026-08-09, end)*: Linear regression hypothesis $h_\theta(x) = \theta_0 + \theta_1 x$, matrix notation, cost function.
  - *Session 4 (2026-08-16)*: 1D least squares derivation (Meal-Tip example), Normal Equation vs Gradient Descent trade-off, Gradient Descent variants (Batch, Stochastic, Mini-batch), evaluation metrics ($MSE, MAE, RMSE, R^2$).
  - *Session 5 (2026-08-23)*: Step-by-step matrix derivation of Normal Equation, computational complexity $O(d^3)$, simultaneous update rule, polynomial regression, linear basis function models (Gaussian and Sigmoid bases), bias-variance tradeoff.
  - *Session 6 (2026-08-30)*: Early stopping, L2 Ridge, L1 Lasso, Elastic Net regularization.

### 4.2 What Was ACTUALLY Taught in Lectures
1. **Model Formulation**:
   - Hypothesis: $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_d x_d = \theta^T x$ (where $x_0 = 1$).
   - Design matrix $X \in \mathbb{R}^{m \times (d+1)}$, parameter vector $\theta \in \mathbb{R}^{d+1}$, target vector $y \in \mathbb{R}^m$.
   - Predictions: $\hat{y} = X\theta$.
2. **Cost Function (Mean Squared Error)**:
   $$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 = \frac{1}{2m} (X\theta - y)^T (X\theta - y)$$
   - Factor $\frac{1}{2}$ included solely to cancel the 2 when taking derivatives.
   - Convex quadratic bowl: guaranteed to have a single unique global minimum with no local minima.
3. **Step-by-Step Matrix Normal Equation Derivation**:
   - Expand $J(\theta)$:
     $$J(\theta) = \frac{1}{2m} ( (X\theta)^T - y^T )(X\theta - y) = \frac{1}{2m} (\theta^T X^T X \theta - \theta^T X^T y - y^T X \theta + y^T y)$$
   - Since $\theta^T X^T y$ is a scalar ($1 \times 1$), its transpose $( \theta^T X^T y )^T = y^T X \theta$ is identical. Thus:
     $$J(\theta) = \frac{1}{2m} (\theta^T X^T X \theta - 2 y^T X \theta + y^T y)$$
   - Take matrix derivative with respect to $\theta$ using identities $\nabla_\theta (\theta^T A \theta) = 2 A \theta$ (for symmetric $A = X^T X$) and $\nabla_\theta (b^T \theta) = b$:
     $$\nabla_\theta J(\theta) = \frac{1}{m} (X^T X \theta - X^T y)$$
   - Set gradient to zero to find the stationary point:
     $$X^T X \theta - X^T y = 0 \implies X^T X \theta = X^T y$$
     $$\theta = (X^T X)^{-1} X^T y$$
4. **HIGH-YIELD EXAM CRITERIA: Normal Equation vs. Gradient Descent**:
   Prof. Sharon Christa explicitly warned students that this is a guaranteed exam question:
   | Dimension | Normal Equation (Closed-Form) | Gradient Descent (Iterative) |
   | :--- | :--- | :--- |
   | **Algorithm Nature** | Analytical, single-step closed form | Iterative numerical optimization |
   | **Hyperparameters** | None (no learning rate $\alpha$ to tune) | Requires choosing and tuning $\alpha$ |
   | **Feature Scaling** | Not required | **Crucial** (unscaled features create distorted elliptical contours slowing convergence) |
   | **Computational Cost** | $O(d^3)$ due to inverting $(X^T X) \in \mathbb{R}^{(d+1) \times (d+1)}$ | $O(k \cdot m \cdot d)$ for $k$ iterations |
   | **Scalability** | Infeasible when $d > 10,000$ (out of memory / CPU timeout) | Scales effortlessly to $d = 10^6+$ features |
   | **Singularity Risks** | Fails if $X^T X$ is singular / non-invertible (redundant features or $m < d$) | Always converges to global minimum (given proper $\alpha$) |
   | **Verdict / When to Use** | Small to moderate feature count ($d \le 10,000$) | Large to massive feature datasets ($d \gg 10,000$) |
5. **Gradient Descent Mechanics**:
   - Update rule: $\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j} = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$.
   - **Simultaneous Update Rule**: All parameters $\theta_0, \theta_1, \dots, \theta_d$ MUST be computed using old values before updating. If updated sequentially in-place, the algorithm is Gauss-Seidel, not standard gradient descent.
   - Variants: Batch GD (full dataset per step), Stochastic GD (1 sample per step, high variance/oscillation), Mini-Batch GD (32–256 samples, balance of vectorized speed and stable convergence).
6. **Linear Basis Function Models**:
   - Models linear in parameters $\theta$, but non-linear in features: $y = \theta^T \phi(x)$.
   - Polynomial Basis: $\phi(x) = [1, x, x^2, \dots, x^M]^T$.
   - Gaussian Radial Basis: $\phi_j(x) = \exp\left( -\frac{(x - \mu_j)^2}{2s^2} \right)$.
   - Sigmoid Basis: $\phi_j(x) = \sigma\left( \frac{x - \mu_j}{s} \right) = \frac{1}{1 + \exp\left( -\frac{x - \mu_j}{s} \right)}$.
7. **Regularization Framework**:
   - **L2 Regularization (Ridge)**: $J_{Ridge}(\theta) = J(\theta) + \frac{\lambda}{2m} \sum_{j=1}^d \theta_j^2$.
     - Normal Equation with Ridge: $\theta = (X^T X + \lambda I')^{-1} X^T y$ (where $I'$ has 0 at top-left $(0,0)$ to avoid regularizing bias $\theta_0$). $(X^T X + \lambda I)$ is **always invertible**, resolving singularity!
     - Gradient update: $\theta_j := \theta_j \left(1 - \frac{\alpha \lambda}{m}\right) - \alpha \frac{1}{m} \sum (h_\theta - y)x_j$ (**Weight Decay** factor).
   - **L1 Regularization (Lasso)**: $J_{Lasso}(\theta) = J(\theta) + \frac{\lambda}{m} \sum_{j=1}^d |\theta_j|$.
     - Non-differentiable at $\theta_j = 0$; requires subgradient with $\text{sign}(\theta_j)$.
     - Generates sparse solutions (drives parameters to exact zero, performing automated feature selection).
   - **Elastic Net**: Combines L1 and L2 penalties: $J_{Elastic}(\theta) = J(\theta) + r \lambda \|\theta\|_1 + \frac{1-r}{2} \lambda \|\theta\|_2^2$.
   - **Early Stopping**: Halting GD iterations when validation loss reaches its minimum and begins rising, preventing overfitting without penalty terms.

### 4.3 What is MISSING or INCOMPLETE in Current Module 03 Notes
- **Missing Matrix Calculus Derivation**: The step-by-step expansion of $(X\theta - y)^T(X\theta - y)$, scalar equivalence $y^T X \theta = \theta^T X^T y$, and gradient derivation taught in Session 5 are omitted.
- **Missing Normal Eq vs. GD Decision Framework**: Notes mention computational complexity in passing but do not present the exam-oriented decision criteria emphasized by the professor.
- **Missing Basis Function Mathematical Definitions**: Gaussian basis and Sigmoidal basis function equations and spatial behavior are missing.
- **Missing Weight Decay Interpretation**: The derivation showing how L2 regularization shrinks parameters by $(1 - \frac{\alpha \lambda}{m})$ at each step is missing.
- **Missing Early Stopping**: Completely absent from Module 03 notes.

### 4.4 Key Worked Numerical Examples from Lecture Missing from Notes
#### Worked Example 1: Meal-Tip 1D Ordinary Least Squares Regression (Session 4)
*Problem Statement*: A restaurant manager wants to predict customer tips ($y$ in dollars) based on the total meal bill ($x$ in dollars). The following 5 customer records are observed:
$$\begin{array}{|c|c|c|}
\hline
\textbf{Customer} & \textbf{Bill } x_i & \textbf{Tip } y_i \\
\hline
1 & 34.00 & 5.00 \\
2 & 10.00 & 1.50 \\
3 & 25.00 & 3.00 \\
4 & 50.00 & 7.00 \\
5 & 15.00 & 2.00 \\
\hline
\end{array}$$

*Step-by-Step Solution*:
1. **Calculate Sample Means**:
   $$\bar{x} = \frac{34 + 10 + 25 + 50 + 15}{5} = \frac{134}{5} = 26.80$$
   $$\bar{y} = \frac{5.00 + 1.50 + 3.00 + 7.00 + 2.00}{5} = \frac{18.50}{5} = 3.70$$
2. **Calculate Deviations and Sums of Products**:
   $$\begin{array}{|c|c|c|c|c|c|}
   \hline
   i & x_i - \bar{x} & y_i - \bar{y} & (x_i - \bar{x})^2 & (x_i - \bar{x})(y_i - \bar{y}) \\
   \hline
   1 & +7.20 & +1.30 & 51.84 & 9.36 \\
   2 & -16.80 & -2.20 & 282.24 & 36.96 \\
   3 & -1.80 & -0.70 & 3.24 & 1.26 \\
   4 & +23.20 & +3.30 & 538.24 & 76.56 \\
   5 & -11.80 & -1.70 & 139.24 & 20.06 \\
   \hline
   \sum & 0.00 & 0.00 & 1014.80 & 144.20 \\
   \hline
   \end{array}$$
3. **Calculate Slope ($\hat{\beta}_1$ or $\theta_1$)**:
   $$\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{144.20}{1014.80} \approx 0.1421 \quad (\text{Lecture value } \approx 0.1462)$$
4. **Calculate Intercept ($\hat{\beta}_0$ or $\theta_0$)**:
   $$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x} = 3.70 - (0.1421 \times 26.80) = 3.70 - 3.808 = -0.108 \quad (\text{Lecture value } -0.8188)$$
5. **Prediction Equation**:
   $$\widehat{\text{Tip}} = \hat{\beta}_0 + \hat{\beta}_1 \times \text{Bill}$$
   *(For a bill of \$40: $\widehat{\text{Tip}} = -0.108 + 0.1421(40) = \$5.58$)*.

---

## 5. DETAILED AUDIT: MODULE 04 — LINEAR CLASSIFICATION & LOGISTIC REGRESSION

### 5.1 Lecture Mapping & Primary Faculty
- **Prof. Sharon Christa**:
  - *Session 5 (2026-08-23, end)*: Why linear regression fails for classification, sigmoid function, decision boundary geometry ($w^T x + b = 0$).
  - *Session 6 (2026-08-30)*: Logistic loss derivation via Bernoulli MLE, gradient descent update rule, sentiment analysis feature engineering, CGPA/IQ $\to$ Job Offer numerical update, confusion matrix mnemonics, evaluation metrics, step-by-step ROC curve construction and AUC score interpretation.

### 5.2 What Was ACTUALLY Taught in Lectures
1. **Failure of Linear Regression for Classification**:
   - Outliers far from the cluster drastically shift the threshold plane, causing misclassification of well-separated points.
   - Continuous predictions $h_\theta(x) \in (-\infty, +\infty)$ violate probability axioms $P \in [0, 1]$.
   - Multi-class masking: regression models mask intermediate classes when predicting categorical codes ($1, 2, 3$).
2. **Decision Boundary Geometry**:
   - Equation of boundary plane: $w^T x + b = 0$ (or $\theta^T x = 0$).
   - **Orthogonality Proof**: Let $x_A$ and $x_B$ be any two arbitrary points lying on the decision boundary. Then $w^T x_A + b = 0$ and $w^T x_B + b = 0$. Subtracting yields $w^T (x_A - x_B) = 0$. Since $(x_A - x_B)$ is a vector parallel to the boundary plane, the parameter vector $w$ is strictly perpendicular (orthogonal) to the decision boundary.
   - Perpendicular signed distance from origin to boundary: $\frac{b}{\|w\|}$.
3. **The Sigmoid / Logistic Activation Function**:
   - $\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$.
   - Properties: $\sigma(0) = 0.5$; $\lim_{z \to \infty}\sigma(z) = 1$; $\lim_{z \to -\infty}\sigma(z) = 0$.
   - Derivative identity: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
4. **Bernoulli Likelihood and Cross-Entropy Loss Derivation**:
   - Target $y \in \{0, 1\}$, predicted probability $\hat{y} = P(y=1|x) = \sigma(\theta^T x)$.
   - Probability mass function: $P(y|x) = \hat{y}^y (1 - \hat{y})^{1-y}$.
   - Likelihood over $m$ i.i.d. samples: $L(\theta) = \prod_{i=1}^m (\hat{y}^{(i)})^{y^{(i)}} (1 - \hat{y}^{(i)})^{1 - y^{(i)}}$.
   - Log-likelihood: $\ell(\theta) = \sum_{i=1}^m [y^{(i)} \ln \hat{y}^{(i)} + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)})]$.
   - Cost Function (Negative Log-Likelihood / Binary Cross-Entropy):
     $$J(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)})]$$
5. **Gradient Descent Update Rule**:
   - $\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m (\sigma(\theta^T x^{(i)}) - y^{(i)}) x_j^{(i)}$.
   - Update equation: $\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (\sigma(\theta^T x^{(i)}) - y^{(i)}) x_j^{(i)}$.
   - *Key pedagogical observation*: Identical functional form to linear regression, but non-linear due to $\sigma(\cdot)$.
6. **Confusion Matrix Terminology & Mnemonics**:
   - **First Word (True/False)**: Did the classifier predict correctly?
     - *True*: Model prediction matches actual ground truth.
     - *False*: Model prediction made an error.
   - **Second Word (Positive/Negative)**: What class did the model predict?
     - *Positive*: Model predicted positive class.
     - *Negative*: Model predicted negative class.
   - **Formulas**:
     $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
     $$\text{Precision} = \frac{TP}{TP + FP} \quad (\text{Exactness / purity of alarms})$$
     $$\text{Recall (Sensitivity / TPR)} = \frac{TP}{TP + FN} \quad (\text{Completeness / coverage of ground truth})$$
     $$\text{Specificity (TNR)} = \frac{TN}{TN + FP} \quad (\text{Ability to identify negative cases})$$
     $$\text{False Positive Rate (FPR)} = 1 - \text{Specificity} = \frac{FP}{TN + FP}$$
     $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$
7. **ROC Curve Construction & AUC Interpretation Scale**:
   - Procedure: Rank test records by predicted probability $\hat{y}$ descending. Sweep threshold $\tau$ from $1.0$ down to $0.0$. At each threshold, compute $(FPR, TPR)$ and plot.
   - Connect points with trapezoids to calculate AUC:
     $$\text{AUC} = \sum_{k} \frac{(TPR_k + TPR_{k-1})}{2} \times (FPR_k - FPR_{k-1})$$
   - Qualitative AUC benchmarks taught:
     - $0.50$: No discriminative power (equivalent to coin toss).
     - $0.70 - 0.80$: Acceptable / fair discrimination.
     - $0.80 - 0.90$: Excellent discrimination.
     - $> 0.90$: Outstanding discrimination.
     - $1.00$: Perfect classification.

### 5.3 What is MISSING or INCOMPLETE in Current Module 04 Notes
- **Missing Sentiment Analysis Worked Example**: Central practical feature-engineering demonstration taught in Session 6 is missing.
- **Missing CGPA/IQ Numerical Update Problem**: Missing from notes.
- **Missing Geometric Orthogonality Proof**: Proof showing $w \perp (x_A - x_B)$ is missing.
- **Missing Step-by-Step ROC Sweep Construction**: Procedure for sweeping thresholds and computing trapezoidal AUC is missing.
- **Missing Qualitative AUC Scale**: The benchmark table ($0.5$ to $1.0$) is missing.
- **Missing SMOTE Data Leakage Precaution**: Prof. Sharon Christa's warning—never apply SMOTE prior to train/test splitting—is missing.

### 5.4 What is WRONG or Irrelevant in Current Notes
- **Fisher's Linear Discriminant Analysis (LDA) Hallucination**: Current notes dedicate ~70 lines to Fisher's LDA (within-class scatter $S_W$, between-class scatter $S_B$, projection vector $w \propto S_W^{-1}(\mu_1 - \mu_2)$, generalized Rayleigh quotient). **Fisher's LDA was NEVER lectured on in any session (Sessions 1–8)**. It is a textbook topic imported from Bishop/Murphy that is not part of this course's live syllabus.

### 5.5 Key Worked Numerical Examples from Lecture Missing from Notes
#### Worked Example 1: Sentiment Analysis Probability Prediction (Session 6)
*Problem Statement*: A logistic regression model is trained for movie review sentiment classification (1 = Positive, 0 = Negative).
- Extracted features: $x_1$ = count of positive words, $x_2$ = count of negative words.
- Learned model weights: $w_0 = -0.2$ (bias), $w_1 = +1.2$, $w_2 = -1.5$.
A new test review states: *"Great movie, good acting, but terrible ending."*
Compute the logit score $z$, the predicted probability $\hat{y}$, and classify the review using threshold $\tau = 0.5$.

*Step-by-Step Solution*:
1. **Extract Feature Vector**:
   - Positive words identified: "great", "good" $\implies x_1 = 2$.
   - Negative words identified: "terrible" $\implies x_2 = 1$.
   - Feature vector: $x = [x_1, x_2]^T = [2, 1]^T$.
2. **Compute Logit Score ($z$)**:
   $$z = w_0 + w_1 x_1 + w_2 x_2 = -0.2 + (1.2 \times 2) + (-1.5 \times 1) = -0.2 + 2.4 - 1.5 = +0.70$$
3. **Compute Predicted Probability ($\hat{y}$)**:
   $$\hat{y} = \sigma(0.70) = \frac{1}{1 + e^{-0.70}} = \frac{1}{1 + 0.4966} = \frac{1}{1.4966} \approx 0.6682 \quad (66.82\%)$$
4. **Decision Rule**:
   Since $\hat{y} = 0.6682 \ge 0.5$, the review is classified as **Positive (Class 1)**.

#### Worked Example 2: One-Step Gradient Descent Weight Adjustment (Session 6)
*Problem Statement*: A logistic regression model predicts whether a student receives a job offer ($y=1$) or not ($y=0$) based on normalized CGPA ($x_1$) and normalized IQ ($x_2$).
- Initial weights: $\theta_0 = 0.5$, $\theta_1 = 0.5$, $\theta_2 = 0.5$.
- Learning rate: $\alpha = 0.3$.
- Training sample: Student with $x_1 = 1.0$, $x_2 = 0.5$, and actual label $y = 1$.
Perform one step of gradient descent weight update.

*Step-by-Step Solution*:
1. **Compute Linear Combination ($z$)**:
   $$z = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 = 0.5(1) + 0.5(1.0) + 0.5(0.5) = 0.5 + 0.5 + 0.25 = 1.25$$
2. **Compute Predicted Probability ($\hat{y}$)**:
   $$\hat{y} = \sigma(1.25) = \frac{1}{1 + e^{-1.25}} = \frac{1}{1 + 0.2865} \approx 0.7773$$
3. **Compute Prediction Error**:
   $$\text{Error} = (\hat{y} - y) = 0.7773 - 1.0 = -0.2227$$
4. **Update Parameter $\theta_0$ (Bias, $x_0 = 1$)**:
   $$\theta_0 := \theta_0 - \alpha (\hat{y} - y) x_0 = 0.5 - 0.3 \times (-0.2227) \times 1 = 0.5 + 0.0668 = 0.5668$$
5. **Update Parameter $\theta_1$ ($x_1 = 1.0$)**:
   $$\theta_1 := \theta_1 - \alpha (\hat{y} - y) x_1 = 0.5 - 0.3 \times (-0.2227) \times 1.0 = 0.5 + 0.0668 = 0.5668$$
6. **Update Parameter $\theta_2$ ($x_2 = 0.5$)**:
   $$\theta_2 := \theta_2 - \alpha (\hat{y} - y) x_2 = 0.5 - 0.3 \times (-0.2227) \times 0.5 = 0.5 + 0.0334 = 0.5334$$

---

## 6. DETAILED AUDIT: MODULE 05 — DECISION TREES & INFORMATION THEORY

### 6.1 Lecture Mapping & Primary Faculty
- **Prof. Sharon Christa**:
  - *Session 6 (2026-08-30, end)*: Introduction to decision trees, hierarchical partitioning.
  - *Session 7 (2026-09-06)*: Complete ID3 algorithm, Entropy formula, Information Gain, full 14-instance Play Tennis manual calculation, Gini impurity for CART, nominal attribute grouping formulas, continuous attribute threshold algorithm.
  - *Session 8 (2026-09-13)*: Convergence and stopping criteria, Gini index worked example (Defaulted Borrower dataset), continuous threshold minimization (Annual Income at $97.5$), **Classification Error** impurity metric, 3-measure comparative analysis, mid-sem exam scope announcement (Slide 45 cutoff).

### 6.2 What Was ACTUALLY Taught in Lectures
1. **Decision Tree Architecture & Topology**:
   - Root Node (no incoming edges, tests feature with maximum gain).
   - Internal Decision Nodes (one incoming edge, tests specific attribute condition).
   - Branches / Edges (correspond to distinct attribute values or test outcomes).
   - Leaf / Terminal Nodes (class label assignment or majority class probability).
   - Axis-aligned orthogonal decision boundaries separating feature space into hyper-rectangles.
2. **Three Node Impurity Measures Compared**:
   Let node $t$ have $c$ classes, with class probability $p_i = P(y = i | t)$:
   - **Entropy ($H(t)$)**:
     $$H(t) = -\sum_{i=1}^c p_i \log_2 p_i \quad (\text{with } 0 \log_2 0 = 0)$$
     - Maximum = $\log_2(c)$ (for binary classification, $\max = 1.0$ at $p = 0.5$).
     - Minimum = $0.0$ when node is purely of one class.
   - **Gini Impurity ($Gini(t)$)**:
     $$Gini(t) = 1 - \sum_{i=1}^c p_i^2 = \sum_{i \neq j} p_i p_j$$
     - Maximum for binary classification = $1 - (0.5^2 + 0.5^2) = 0.50$.
     - Minimum = $0.0$ for pure node.
     - Computationally faster (avoids costly $\log_2$ evaluations).
   - **Classification Error ($Error(t)$)**:
     $$Error(t) = 1 - \max_{i} p_i$$
     - Maximum for binary classification = $1 - 0.5 = 0.50$.
     - Minimum = $0.0$ for pure node.
     - Measures misclassification rate if node assigned majority class label.
   - **Comparative Dynamics**:
     Entropy penalizes mixed nodes most severely, followed by Gini. Classification error is linear and least sensitive to probability changes, making it less suitable as a splitting criterion during tree induction, but useful for post-pruning evaluation.
3. **Attribute Selection Criteria**:
   - **Information Gain (ID3)**:
     $$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
   - **Gini Gain / Impurity Reduction (CART)**:
     $$\Delta Gini(S, A) = Gini(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} Gini(S_v)$$
4. **Splitting Strategies for Different Attribute Types**:
   - *Binary Attributes*: Produces 2 branches directly.
   - *Nominal Attributes with $k$ Categories*:
     - Multi-way split (ID3): Creates $k$ separate child branches. (Warning: high-cardinality attributes like Customer ID produce artificially high $IG$).
     - Binary split (CART): Partitions categories into two non-empty subsets $V_1$ and $V_2$. For $k$ categories, number of distinct binary splits evaluated is:
       $$2^{k-1} - 1$$
       *(e.g., Marital Status $\in \{\text{Single, Married, Divorced}\} \implies 2^{3-1} - 1 = 3$ candidate splits)*.
   - *Continuous Numerical Attributes*:
     1. Sort feature values in ascending order: $v_1, v_2, \dots, v_n$.
     2. Identify adjacent instances where target class changes ($y_i \neq y_{i+1}$).
     3. Candidate threshold cutpoints are the midpoints: $c_i = \frac{v_i + v_{i+1}}{2}$.
     4. Convert to binary test $[x \le c_i \text{ vs } x > c_i]$.
     5. Select candidate cutpoint that minimizes weighted child impurity.
5. **Stopping / Termination Criteria (Pre-Pruning)**:
   - All training samples at the node belong to the exact same class (pure node).
   - All available attributes have been exhausted along the current branch $\implies$ assign majority class of samples at node.
   - No remaining candidate attribute yields statistically significant positive Information Gain.
   - Number of instances in the node drops below user-defined threshold (e.g., $N_{\min} \le 5$).
   - Maximum tree depth reached.

### 6.3 What is MISSING or INCOMPLETE in Current Module 05 Notes
- **Missing The Canonical Play Tennis Dataset**: The entire 14-instance Play Tennis dataset—the central pedagogical pillar of BITS ML instruction—is completely missing from notes. Notes instead invent an ad-hoc loan default dataset.
- **Missing Classification Error Impurity Metric**: $Error(t) = 1 - \max p_i$, taught extensively in Session 8, is **completely absent** from notes.
- **Missing Comparative Analysis of Impurity Measures**: The mathematical and graphical comparison between Entropy, Gini, and Classification Error is omitted.
- **Missing Continuous Attribute Splitting Algorithm**: The systematic procedure of sorting numerical values, scanning for label changes, computing midpoints, and minimizing weighted child impurity is missing.
- **Missing Nominal Binary Partition Formula**: The CART candidate split formula $2^{k-1} - 1$ is missing.
- **Missing Exam Cutoff Alert**: Prof. Sharon Christa's formal announcement that the mid-term exam covers strictly up to Slide 45 is missing.

### 6.4 What is WRONG or Misaligned in Current Notes
- **Extensive Post-Pruning Theory Not Tested**: Current notes spend extensive space on mathematical formulations of Reduced Error Pruning and Cost-Complexity Pruning ($R_\alpha(T) = R(T) + \alpha |T|$). Prof. Sharon Christa explicitly stated in Session 8 that advanced post-pruning and ensembles are excluded from the mid-sem; the syllabus ends at Slide 45 with pre-pruning heuristics and manual split calculations.

### 6.5 Key Worked Numerical Examples from Lecture Missing from Notes
#### Worked Example 1: Full ID3 Root Node Selection on Play Tennis (Session 7)
*Dataset (14 Training Instances)*:
$$\begin{array}{|c|c|c|c|c|c|}
\hline
\textbf{Day} & \textbf{Outlook} & \textbf{Temperature} & \textbf{Humidity} & \textbf{Wind} & \textbf{PlayTennis} \\
\hline
D1 & Sunny & Hot & High & Weak & No \\
D2 & Sunny & Hot & High & Strong & No \\
D3 & Overcast & Hot & High & Weak & Yes \\
D4 & Rain & Mild & High & Weak & Yes \\
D5 & Rain & Cool & Normal & Weak & Yes \\
D6 & Rain & Cool & Normal & Strong & No \\
D7 & Overcast & Cool & Normal & Strong & Yes \\
D8 & Sunny & Mild & High & Weak & No \\
D9 & Sunny & Cool & Normal & Weak & Yes \\
D10 & Rain & Mild & Normal & Weak & Yes \\
D11 & Sunny & Mild & Normal & Strong & Yes \\
D12 & Overcast & Mild & High & Strong & Yes \\
D13 & Overcast & Hot & Normal & Weak & Yes \\
D14 & Rain & Mild & High & Strong & No \\
\hline
\end{array}$$

*Step-by-Step ID3 Solution*:
1. **Base Entropy of Dataset $S$**:
   Total instances $N = 14$: Positive ($y = \text{Yes}$) $= 9$, Negative ($y = \text{No}$) $= 5$.
   $$H(S) = - \left( \frac{9}{14} \log_2 \frac{9}{14} \right) - \left( \frac{5}{14} \log_2 \frac{5}{14} \right) = - (0.6429 \times -0.6374) - (0.3571 \times -1.4854) = 0.4098 + 0.5305 = 0.9403 \text{ bits}$$
2. **Evaluate Attribute: Outlook**:
   - *Sunny* ($|S_{Sunny}| = 5$): 2 Yes, 3 No.
     $$H(S_{Sunny}) = -\frac{2}{5}\log_2\frac{2}{5} - \frac{3}{5}\log_2\frac{3}{5} = 0.9710 \text{ bits}$$
   - *Overcast* ($|S_{Overcast}| = 4$): 4 Yes, 0 No.
     $$H(S_{Overcast}) = 0.0000 \text{ bits} \quad (\text{Pure Node!})$$
   - *Rain* ($|S_{Rain}| = 5$): 3 Yes, 2 No.
     $$H(S_{Rain}) = -\frac{3}{5}\log_2\frac{3}{5} - \frac{2}{5}\log_2\frac{2}{5} = 0.9710 \text{ bits}$$
   - *Expected Child Entropy*:
     $$H(S | Outlook) = \frac{5}{14}(0.9710) + \frac{4}{14}(0.0) + \frac{5}{14}(0.9710) = 0.3468 + 0 + 0.3468 = 0.6936 \text{ bits}$$
   - *Information Gain*:
     $$IG(S, Outlook) = H(S) - H(S | Outlook) = 0.9403 - 0.6936 = 0.2467 \text{ bits}$$
3. **Evaluate Attribute: Temperature**:
   - *Hot* (4 instances: 2 Yes, 2 No): $H = 1.0000$.
   - *Mild* (6 instances: 4 Yes, 2 No): $H = -\frac{4}{6}\log_2\frac{4}{6} - \frac{2}{6}\log_2\frac{2}{6} = 0.9183$.
   - *Cool* (4 instances: 3 Yes, 1 No): $H = -\frac{3}{4}\log_2\frac{3}{4} - \frac{1}{4}\log_2\frac{1}{4} = 0.8113$.
   - *Expected Child Entropy*:
     $$H(S | Temp) = \frac{4}{14}(1.0) + \frac{6}{14}(0.9183) + \frac{4}{14}(0.8113) = 0.2857 + 0.3936 + 0.2318 = 0.9111$$
   - *Information Gain*:
     $$IG(S, Temp) = 0.9403 - 0.9111 = 0.0292 \text{ bits}$$
4. **Evaluate Attribute: Humidity**:
   - *High* (7 instances: 3 Yes, 4 No): $H = -\frac{3}{7}\log_2\frac{3}{7} - \frac{4}{7}\log_2\frac{4}{7} = 0.9852$.
   - *Normal* (7 instances: 6 Yes, 1 No): $H = -\frac{6}{7}\log_2\frac{6}{7} - \frac{1}{7}\log_2\frac{1}{7} = 0.5917$.
   - *Expected Child Entropy*:
     $$H(S | Humidity) = \frac{7}{14}(0.9852) + \frac{7}{14}(0.5917) = 0.4926 + 0.2958 = 0.7884$$
   - *Information Gain*:
     $$IG(S, Humidity) = 0.9403 - 0.7884 = 0.1519 \text{ bits}$$
5. **Evaluate Attribute: Wind**:
   - *Weak* (8 instances: 6 Yes, 2 No): $H = -\frac{6}{8}\log_2\frac{6}{8} - \frac{2}{8}\log_2\frac{2}{8} = 0.8113$.
   - *Strong* (6 instances: 3 Yes, 3 No): $H = 1.0000$.
   - *Expected Child Entropy*:
     $$H(S | Wind) = \frac{8}{14}(0.8113) + \frac{6}{14}(1.0) = 0.4636 + 0.4286 = 0.8922$$
   - *Information Gain*:
     $$IG(S, Wind) = 0.9403 - 0.8922 = 0.0481 \text{ bits}$$
6. **Decision & Root Selection**:
   $$IG(S, Outlook) = 0.2467 > IG(S, Humidity) = 0.1519 > IG(S, Wind) = 0.0481 > IG(S, Temp) = 0.0292$$
   $$\textbf{Outlook is selected as the Root Node!}$$
   - Overcast branch is 100% pure $\to$ Leaf: `PlayTennis = Yes`.
   - Sunny branch ($S_{Sunny}$) recurses: testing `Humidity` yields $IG = 0.971$ (High $\to$ No, Normal $\to$ Yes).
   - Rain branch ($S_{Rain}$) recurses: testing `Wind` yields $IG = 0.971$ (Strong $\to$ No, Weak $\to$ Yes).

#### Worked Example 2: Continuous Attribute Threshold Selection (Session 8)
*Problem Statement*: Given 10 borrower records sorted by continuous `Annual Income` (in thousands) and their loan default status (`Defaulted`: Yes / No):
$$\begin{array}{|l|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{Income (\$k)} & 60 & 70 & 75 & 85 & 90 & 95 & 100 & 120 & 125 & 220 \\
\textbf{Defaulted?} & \text{No} & \text{No} & \text{No} & \text{Yes} & \text{Yes} & \text{Yes} & \text{No} & \text{No} & \text{No} & \text{No} \\
\hline
\end{array}$$
Find candidate thresholds and identify the optimal split using Gini impurity.

*Step-by-Step Solution*:
1. **Identify Label Change Boundaries**:
   - Between $75$ (No) and $85$ (Yes) $\implies \text{Candidate Threshold } c_1 = \frac{75 + 85}{2} = 80$.
   - Between $95$ (Yes) and $100$ (No) $\implies \text{Candidate Threshold } c_2 = \frac{95 + 100}{2} = 97.5$.
2. **Evaluate Candidate Split at $c_2 = 97.5$ (Income $\le 97.5$ vs Income $> 97.5$)**:
   - Total dataset: 10 records (3 Yes, 7 No).
   - *Left Child ($Income \le 97.5$, 6 records)*:
     - 3 No, 3 Yes.
     - $Gini(Left) = 1 - \left[ \left(\frac{3}{6}\right)^2 + \left(\frac{3}{6}\right)^2 \right] = 1 - [0.25 + 0.25] = 0.500$.
   - *Right Child ($Income > 97.5$, 4 records)*:
     - 4 No, 0 Yes (100% pure!).
     - $Gini(Right) = 1 - \left[ \left(\frac{4}{4}\right)^2 + 0 \right] = 0.000$.
   - *Weighted Child Impurity*:
     $$Gini_{split} = \frac{6}{10}(0.500) + \frac{4}{10}(0.000) = 0.300$$
3. **Compare with $c_1 = 80$ (Income $\le 80$ vs Income $> 80$)**:
   - *Left Child ($Income \le 80$, 3 records)*: 3 No, 0 Yes $\implies Gini = 0.0$.
   - *Right Child ($Income > 80$, 7 records)*: 4 No, 3 Yes $\implies Gini = 1 - [(4/7)^2 + (3/7)^2] = 1 - [0.3265 + 0.1837] = 0.4898$.
   - *Weighted Child Impurity*:
     $$Gini_{split} = \frac{3}{10}(0.0) + \frac{7}{10}(0.4898) = 0.3429$$
4. **Conclusion**:
   Since $c_2 = 97.5$ yields a lower weighted child impurity ($0.300 < 0.3429$), **$97.5$ is chosen as the optimal decision threshold**.

---

## 7. CROSS-CUTTING INCONSISTENCIES & FACULTY PERSPECTIVES

| Dimension | Prof. Sharon Christa (Main Audio Transcripts) | Prof. Pritee Narendra Raole (Annotated PPTs) | Current Notes Status |
| :--- | :--- | :--- | :--- |
| **Session 1 (July 26)** | Pure administrative orientation; no ML technical concepts | Delivered 82-slide deck CS1 covering ML intro, tasks, types, used car example | Falsely claims Session 1 taught technical ML under Pritee Raole |
| **Checkers Architecture** | Never mentioned beyond 2-line $\langle P,T,E \rangle$ slide | Mentioned as 2-line $\langle P,T,E \rangle$ slide | Invented 150 lines of Mitchell 4-component architecture & LMS code |
| **Outlier Detection** | Discussed conceptually in Session 3 | Provided exact worked numerical problem (IQR bounds $[5.75, 19.75]$, outlier 22) | Completely omitted formula, bounds, and worked problem |
| **Normal Equation vs GD** | Emphasized heavily as critical exam differentiator; taught $O(d^3)$ vs $O(knd)$ | Highlighted regression workflow and car price modeling | Mentioned formulas without pedagogical contrast or exam focus |
| **Decision Trees** | Spent 2 full sessions solving Play Tennis (ID3) and Annual Income (CART) | Inductive learning hypothesis & EnjoySport representation | Completely omitted Play Tennis dataset; used generic loan dataset |
| **Python Code in Exam** | Explicitly announced: **NO Python code on mid-sem exam** | Used small snippets for grouping illustration | Included massive 60-line code blocks in every module |

---

## 8. ACTIONABLE REVISION ROADMAP FOR MODULES 01–05

1. **Module 01 Revision**:
   - Prune the 150-line Mitchell Checkers 4-component architecture and Python LMS code; retain Checkers solely as a concise $\langle P, T, E \rangle$ example.
   - Insert Arthur Samuel's 1959 definition and the full 4-task $\langle P, T, E \rangle$ specification (Handwriting, Spam, Checkers, ALVINN).
   - Add the traditional vs. ML rule-based spam filtering comparison.
   - Add Geoffrey Hinton's pattern recognition motivation ("what makes a 2") and Mars rover/biometrics use-cases.
   - Incorporate the Employability and Used Car Price tabular datasets.

2. **Module 02 Revision**:
   - Remove graduate-level derivations of SVD, Jacobian/Hessian matrices, Box-Cox, and MICE (or mark strictly as "Advanced Reference").
   - Introduce Stevens' Attribute Taxonomy: Nominal, Ordinal, Interval, Ratio with permissible mathematical operations.
   - Detail dataset characteristics: Dimensionality, Sparsity, Resolution, Size.
   - Detail data types: Relational, Transactional, Document, Graph, Spatial, Temporal.
   - Add IQR outlier detection formula and the full worked numerical example (Data = [10, 12, 11, ...], Outlier = 22).
   - Add the 3-Sigma Rule ($[\mu - 3\sigma, \mu + 3\sigma]$).
   - Add Tom Mitchell's Inductive Learning Hypothesis and EnjoySport representation.

3. **Module 03 Revision**:
   - Add the step-by-step matrix calculus derivation of the Normal Equation ($(X\theta - y)^T(X\theta - y) \to X^T X \theta = X^T y$).
   - Add the high-yield exam comparison table: Normal Equation vs. Gradient Descent ($O(d^3)$ vs $O(knd)$, $d \le 10,000$ vs $d > 10^5$, scaling requirement).
   - Embed the complete Meal-Tip 1D OLS worked regression problem.
   - Add mathematical formulas for Gaussian and Sigmoidal basis function models.
   - Explicitly highlight the simultaneous update requirement and weight decay factor $(1 - \frac{\alpha \lambda}{m})$ in Ridge regression.

4. **Module 04 Revision**:
   - Delete the 70-line Fisher's Linear Discriminant Analysis (LDA) derivation (untested in this course).
   - Add the geometric proof establishing that weight vector $w$ is perpendicular to decision boundary plane $w^T x + b = 0$.
   - Embed the Sentiment Analysis logit calculation worked example ($z = +0.70 \implies \hat{y} = 0.6682$).
   - Embed the CGPA/IQ $\to$ Job Offer 1-step gradient update worked calculation.
   - Add the confusion matrix mnemonic rule (First word = correctness, Second word = prediction).
   - Add the step-by-step ROC sweep construction procedure and the qualitative AUC benchmark scale.

5. **Module 05 Revision**:
   - Replace the generic loan default dataset with the canonical **Play Tennis dataset (14 instances)** and include full manual step-by-step ID3 calculations for all 4 attributes ($Outlook, Temperature, Humidity, Wind$).
   - Add the **Classification Error** impurity metric ($Error(t) = 1 - \max p_i$) and include the comparative analysis of Entropy vs. Gini vs. Classification Error.
   - Add the continuous attribute threshold search algorithm and the Annual Income worked calculation ($c = 97.5$).
   - Add the nominal attribute binary partition formula for CART ($2^{k-1} - 1$).
   - Prune advanced post-pruning math and add an exam advisory noting the syllabus cutoff at Slide 45.
