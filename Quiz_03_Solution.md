# Data Mining Quiz 03 - Logistic Regression
## Complete Solution Guide

---

## **Given Dataset**
```
Applicant | Hours Studied (X₁) | Attendance (%) (X₂) | Admitted (Y)
----------|-------------------|---------------------|-------------
A         | 2                 | 60                  | 0
B         | 3                 | 70                  | 0  
C         | 4                 | 80                  | 1
D         | 5                 | 90                  | 1
E         | 6                 | 95                  | 1
```

**Where:**
- X₁ = Hours Studied (Independent Variable 1)
- X₂ = Attendance Percentage (Independent Variable 2)  
- Y = Admission Status (Dependent Variable: 1 = Admitted, 0 = Not Admitted)

---

## **Q1. Understanding the Model (3 marks)**

### **a. Write the logistic regression equation with two independent variables.**

**Logistic Regression Equation:**

The logistic regression model with two independent variables is:

```
P(Y = 1) = 1 / (1 + e^-(β₀ + β₁X₁ + β₂X₂))
```

**Components:**
- **P(Y = 1)**: Probability that Y equals 1 (probability of admission)
- **β₀**: Intercept coefficient (bias term)
- **β₁**: Coefficient for X₁ (Hours Studied)
- **β₂**: Coefficient for X₂ (Attendance Percentage)
- **e**: Mathematical constant (≈ 2.718)

**Linear Combination (Log-Odds):**
```
z = β₀ + β₁X₁ + β₂X₂
```

**Alternative Form:**
```
P(Y = 1) = 1 / (1 + e^-z)
```

### **b. Explain the significance of the sigmoid function in logistic regression.**

**Sigmoid Function:**

The sigmoid function is: **σ(z) = 1 / (1 + e^-z)**

**Significance:**

1. **Maps to Probability Range**: 
   - Output is always between 0 and 1
   - Perfect for representing probabilities

2. **S-shaped Curve**:
   - Smooth transition from 0 to 1
   - Gradual change near boundaries
   - Steep change in the middle

3. **Asymptotic Behavior**:
   - As z → +∞, P(Y=1) → 1
   - As z → -∞, P(Y=1) → 0
   - At z = 0, P(Y=1) = 0.5

4. **Differentiable**:
   - Allows use of gradient-based optimization
   - Enables gradient descent for parameter estimation

5. **Log-Odds Interpretation**:
   - Linear relationship between predictors and log-odds
   - Simplifies coefficient interpretation

---

## **Q2. Manual Computation (6 marks)**

### **a. Explain the process of estimating the coefficients using gradient descent or maximum likelihood.**

## **Maximum Likelihood Estimation (MLE)**

**Concept:**
MLE finds coefficients that maximize the likelihood of observing the given data.

**Likelihood Function:**
For each observation i:
```
L(yi | xi) = P(Y=1)^yi × P(Y=0)^(1-yi)
```

**Overall Likelihood:**
```
L = ∏[P(Y=1)^yi × P(Y=0)^(1-yi)] for all i
```

**Log-Likelihood:**
```
ℓ = Σ[yi × log(P(Y=1)) + (1-yi) × log(P(Y=0))]
```

**Steps:**
1. **Initialize** coefficients β₀, β₁, β₂ randomly
2. **Calculate** P(Y=1) for each observation using current coefficients
3. **Compute** log-likelihood
4. **Update** coefficients using partial derivatives
5. **Repeat** until convergence

## **Gradient Descent Method**

**Cost Function (Cross-Entropy):**
```
J = -1/n × Σ[yi × log(h(xi)) + (1-yi) × log(1-h(xi))]
```

**Gradient Calculation:**
```
∂J/∂β₀ = 1/n × Σ(h(xi) - yi)
∂J/∂β₁ = 1/n × Σ((h(xi) - yi) × xi1)  
∂J/∂β₂ = 1/n × Σ((h(xi) - yi) × xi2)
```

**Update Rules:**
```
β₀ := β₀ - α × ∂J/∂β₀
β₁ := β₁ - α × ∂J/∂β₁  
β₂ := β₂ - α × ∂J/∂β₂
```

Where α is the learning rate.

### **b. Compute the probability of admission for given coefficients.**

**Given:**
- **β₀ = -6**
- **β₁ = 1** 
- **β₂ = 0.05**
- **Applicant:** X₁ = 4 hours studied, X₂ = 80% attendance

**Step-by-Step Calculation:**

**Step 1: Calculate the linear combination (z)**
```
z = β₀ + β₁X₁ + β₂X₂
z = -6 + (1)(4) + (0.05)(80)
z = -6 + 4 + 4
z = 2
```

**Step 2: Calculate e^-z**
```
e^-z = e^-2 = 1/e² ≈ 1/7.389 ≈ 0.1353
```

**Step 3: Apply the sigmoid function**
```
P(Y = 1) = 1 / (1 + e^-z)
P(Y = 1) = 1 / (1 + 0.1353)
P(Y = 1) = 1 / 1.1353
P(Y = 1) ≈ 0.881
```

**Result:** The probability of admission is **88.1%**

**Verification:**
```
P(Y = 1) = 1 / (1 + e^-(2)) = 1 / (1 + 0.1353) = 0.881 ✓
```

---

## **Q3. Interpretation (4 marks)**

### **a. Interpret the meaning of the coefficients in the logistic regression context.**

**Given Coefficients:**
- **β₀ = -6** (Intercept)
- **β₁ = 1** (Hours Studied coefficient)
- **β₂ = 0.05** (Attendance coefficient)

## **Coefficient Interpretations:**

### **β₀ = -6 (Intercept)**
- **Meaning:** Log-odds of admission when both X₁ = 0 and X₂ = 0
- **Interpretation:** When a student studies 0 hours and has 0% attendance, the log-odds of admission is -6
- **Probability:** P(Y=1) = 1/(1+e^6) ≈ 0.0025 (0.25% chance)

### **β₁ = 1 (Hours Studied)**
- **Meaning:** For each additional hour studied, log-odds increase by 1
- **Odds Ratio:** e^1 = 2.718
- **Interpretation:** Each additional hour of study increases the odds of admission by 172% (odds multiply by 2.718)
- **Example:** Going from 3 to 4 hours increases odds by factor of 2.718

### **β₂ = 0.05 (Attendance)**
- **Meaning:** For each 1% increase in attendance, log-odds increase by 0.05
- **Odds Ratio:** e^0.05 = 1.051
- **Interpretation:** Each 1% increase in attendance increases odds by 5.1%
- **Example:** Going from 80% to 90% attendance (10% increase) multiplies odds by e^(0.05×10) = e^0.5 ≈ 1.65

### **b. What does the model predict when the probability P(Y=1) is greater than 0.5?**

**Decision Rule:**

When **P(Y=1) > 0.5**, the model predicts:
- **Class:** Y = 1 (Admitted)
- **Decision:** The applicant will be admitted

**Why 0.5 Threshold?**

1. **Equal Probability:** 0.5 represents equal likelihood of both outcomes
2. **Decision Boundary:** Natural cutoff point for binary classification
3. **Optimal for Balanced Classes:** Minimizes classification error when classes are equally likely

**Mathematical Condition:**
```
P(Y=1) > 0.5
1/(1+e^-z) > 0.5
1 > 0.5(1+e^-z)  
1 > 0.5 + 0.5e^-z
0.5 > 0.5e^-z
1 > e^-z
z > 0
```

**Therefore:** Model predicts "Admitted" when **z = β₀ + β₁X₁ + β₂X₂ > 0**

**For our model:** -6 + X₁ + 0.05X₂ > 0, or **X₁ + 0.05X₂ > 6**

---

## **Q4. Model Evaluation (4 marks)**

### **a. Define and explain the purpose of the confusion matrix.**

## **Confusion Matrix Definition**

A confusion matrix is a table used to evaluate the performance of a binary classification model.

**Structure:**
```
                    Predicted
                 |  0   |  1   |
Actual       0   | TN   | FP   |
             1   | FN   | TP   |
```

**Components:**
- **TP (True Positive):** Correctly predicted admission (Y=1, Ŷ=1)
- **TN (True Negative):** Correctly predicted rejection (Y=0, Ŷ=0)  
- **FP (False Positive):** Incorrectly predicted admission (Y=0, Ŷ=1)
- **FN (False Negative):** Incorrectly predicted rejection (Y=1, Ŷ=0)

**Example for Our Dataset:**

Using our model with coefficients β₀=-6, β₁=1, β₂=0.05:

**Predictions:**
```
Applicant A: z = -6 + 1(2) + 0.05(60) = -1, P(Y=1) = 0.27 → Predicted: 0 (Actual: 0) ✓
Applicant B: z = -6 + 1(3) + 0.05(70) = 0.5, P(Y=1) = 0.62 → Predicted: 1 (Actual: 0) ✗
Applicant C: z = -6 + 1(4) + 0.05(80) = 2, P(Y=1) = 0.88 → Predicted: 1 (Actual: 1) ✓
Applicant D: z = -6 + 1(5) + 0.05(90) = 3.5, P(Y=1) = 0.97 → Predicted: 1 (Actual: 1) ✓
Applicant E: z = -6 + 1(6) + 0.05(95) = 4.75, P(Y=1) = 0.99 → Predicted: 1 (Actual: 1) ✓
```

**Confusion Matrix:**
```
                Predicted
             |  0  |  1  |
Actual   0   |  1  |  1  |  TN=1, FP=1
         1   |  0  |  3  |  FN=0, TP=3
```

**Purpose:**
1. **Performance Assessment:** Shows where model makes errors
2. **Error Analysis:** Identifies types of mistakes (FP vs FN)
3. **Class-specific Performance:** Reveals bias toward certain classes
4. **Metric Calculation:** Base for computing accuracy, precision, recall

### **b. Describe two evaluation metrics for logistic regression models.**

## **1. Accuracy**

**Definition:** Proportion of correct predictions out of total predictions.

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Calculation for Our Example:**
```
Accuracy = (3 + 1) / (3 + 1 + 1 + 0) = 4/5 = 0.8 = 80%
```

**Interpretation:**
- Our model correctly classifies 80% of applicants
- Good overall performance measure
- **Limitation:** Can be misleading with imbalanced datasets

**When to Use:**
- Balanced datasets
- Equal importance of both classes
- General performance overview

## **2. Precision**

**Definition:** Proportion of positive predictions that are actually correct.

**Formula:**
```
Precision = TP / (TP + FP)
```

**Calculation for Our Example:**
```
Precision = 3 / (3 + 1) = 3/4 = 0.75 = 75%
```

**Interpretation:**
- Of all applicants predicted to be admitted, 75% are actually admitted
- Measures the quality of positive predictions
- **Important when:** False positives are costly (e.g., admitting unqualified students)

**Alternative Metric: Recall (Sensitivity)**

**Definition:** Proportion of actual positives that are correctly identified.

**Formula:**
```
Recall = TP / (TP + FN)
```

**Calculation for Our Example:**
```
Recall = 3 / (3 + 0) = 3/3 = 1.0 = 100%
```

**Interpretation:**
- Our model identifies 100% of students who should be admitted
- **Important when:** False negatives are costly (e.g., rejecting qualified students)

**Trade-off:**
- High precision may lead to low recall (conservative model)
- High recall may lead to low precision (liberal model)
- **F1-Score** balances both: F1 = 2 × (Precision × Recall) / (Precision + Recall)

---

## **Q5. Reflection (3 marks)**

### **a. Discuss a real-world application of logistic regression.**

## **Real-World Application: Medical Diagnosis System**

**Scenario:** Hospital Emergency Department Triage System

**Problem:** Quickly determine if a patient with chest pain is having a heart attack (Myocardial Infarction).

**Dataset Features:**
- **Age** (years)
- **Blood Pressure** (systolic)
- **Cholesterol Level** (mg/dL)
- **Chest Pain Type** (1-4 scale)
- **ECG Results** (0=normal, 1=abnormal)
- **Previous Heart Disease** (0=no, 1=yes)

**Target Variable:**
- **Heart Attack** (1 = Yes, 0 = No)

**Logistic Regression Model:**
```
P(Heart Attack = 1) = 1 / (1 + e^-(β₀ + β₁×Age + β₂×BP + β₃×Cholesterol + ...))
```

**Business Value:**

1. **Quick Decision Making:**
   - Instant probability calculation
   - Faster than waiting for lab results
   - Immediate triage decisions

2. **Resource Allocation:**
   - High-risk patients get immediate attention
   - Low-risk patients can wait
   - Efficient use of emergency resources

3. **Cost Effectiveness:**
   - Reduces unnecessary expensive tests
   - Prevents missed diagnoses
   - Optimizes hospital workflow

4. **Risk Stratification:**
   - Probability scores help prioritize patients
   - Consistent decision-making across doctors
   - Reduces human error and bias

**Example Prediction:**
- **Patient:** 65-year-old, BP=180, High cholesterol, Abnormal ECG
- **Model Output:** P(Heart Attack) = 0.85 (85% probability)
- **Action:** Immediate cardiac intervention

### **b. Suggest how this model can be improved or extended for multiclass classification.**

## **Extensions for Multiclass Classification**

**Current Problem:** Binary classification (Admitted/Not Admitted)

**Extended Problem:** Multiclass admission decision

**Classes:**
1. **Full Admission** (Unconditional acceptance)
2. **Conditional Admission** (Acceptance with requirements)
3. **Waitlist** (Potential acceptance later)
4. **Rejection** (Not admitted)

## **1. One-vs-Rest (OvR) Approach**

**Method:** Train separate binary logistic models for each class.

**Models:**
```
Model 1: P(Full Admission vs All Others)
Model 2: P(Conditional vs All Others)  
Model 3: P(Waitlist vs All Others)
Model 4: P(Rejection vs All Others)
```

**Prediction:** Choose class with highest probability.

**Advantages:**
- Simple to implement
- Uses existing logistic regression
- Interpretable coefficients

**Disadvantages:**
- Probabilities may not sum to 1
- Class imbalance issues

## **2. Multinomial Logistic Regression**

**Method:** Extend logistic regression directly for multiple classes.

**Softmax Function:**
```
P(Y = k) = e^(βₖX) / Σ(e^(βⱼX)) for all j
```

**Reference Class:** Choose one class (e.g., Rejection) as baseline.

**Model Equations:**
```
P(Full) = e^(β₁X) / (1 + e^(β₁X) + e^(β₂X) + e^(β₃X))
P(Conditional) = e^(β₂X) / (1 + e^(β₁X) + e^(β₂X) + e^(β₃X))
P(Waitlist) = e^(β₃X) / (1 + e^(β₁X) + e^(β₂X) + e^(β₃X))
P(Rejection) = 1 / (1 + e^(β₁X) + e^(β₂X) + e^(β₃X))
```

**Advantages:**
- Probabilities sum to 1
- Theoretically sound
- Single unified model

## **3. Model Improvements**

### **Feature Engineering:**
- **Interaction Terms:** Hours × Attendance
- **Polynomial Features:** Hours²
- **Categorical Encoding:** School type, demographics
- **Feature Scaling:** Normalize numerical variables

### **Advanced Techniques:**
- **Regularization:** L1/L2 penalties to prevent overfitting
- **Cross-Validation:** Better model selection
- **Feature Selection:** Remove irrelevant variables
- **Ensemble Methods:** Combine multiple models

### **Data Improvements:**
- **Larger Dataset:** More training examples
- **Balanced Classes:** Equal representation
- **Additional Features:** SAT scores, essays, recommendations
- **Temporal Data:** Multi-year application trends

### **Threshold Optimization:**
- **ROC Curve Analysis:** Find optimal probability thresholds
- **Cost-Sensitive Learning:** Different costs for different errors
- **Class-Specific Thresholds:** Different cutoffs per class

---

## **Summary**

This quiz demonstrates key concepts of logistic regression:

1. **Model Understanding:** Sigmoid function and probability interpretation
2. **Parameter Estimation:** MLE and gradient descent methods
3. **Manual Computation:** Step-by-step probability calculation
4. **Coefficient Interpretation:** Odds ratios and log-odds meaning
5. **Model Evaluation:** Confusion matrix and performance metrics
6. **Real-World Applications:** Medical diagnosis and decision systems
7. **Extensions:** Multiclass classification approaches

**Key Takeaways:**
- Logistic regression maps linear combinations to probabilities
- Coefficients represent log-odds changes
- Sigmoid function ensures output between 0 and 1
- Evaluation requires multiple metrics beyond accuracy
- Extensions enable complex real-world applications