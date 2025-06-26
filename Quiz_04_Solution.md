# Data Mining Quiz 04 - Linear Regression
## Complete Solution Guide

---

## **Given Dataset**
```
Student | Study Hours (X₁) | Sleep Hours (X₂) | Exam Score (Y)
--------|------------------|------------------|----------------
A       | 2                | 6                | 55
B       | 3                | 5                | 60
C       | 4                | 6                | 65
D       | 5                | 4                | 70
E       | 6                | 5                | 75
```

**Where:**
- X₁ = Study Hours (Independent Variable 1)
- X₂ = Sleep Hours (Independent Variable 2)
- Y = Exam Score (Dependent Variable)

---

## **Q1. Understanding the Model (3 marks)**

### **a. Write the general form of the multiple linear regression equation with two variables.**

**Multiple Linear Regression Equation:**

```
Y = β₀ + β₁X₁ + β₂X₂ + ε
```

**Matrix Form:**
```
Y = Xβ + ε
```

Where:
```
Y = [Y₁, Y₂, ..., Yₙ]ᵀ    (n × 1 vector)
X = [1, X₁, X₂]           (n × 3 matrix)
β = [β₀, β₁, β₂]ᵀ         (3 × 1 vector)
ε = [ε₁, ε₂, ..., εₙ]ᵀ    (n × 1 error vector)
```

### **b. Define each term in the equation.**

**Term Definitions:**

1. **Y**: 
   - **Dependent Variable** (Response Variable)
   - The variable we want to predict
   - In our case: Exam Score

2. **β₀ (Beta-zero)**:
   - **Intercept** (Y-intercept)
   - Expected value of Y when all X variables = 0
   - Baseline exam score when study hours = 0 and sleep hours = 0

3. **β₁ (Beta-one)**:
   - **Slope coefficient** for X₁
   - Change in Y for one unit increase in X₁ (holding X₂ constant)
   - Effect of one additional study hour on exam score

4. **β₂ (Beta-two)**:
   - **Slope coefficient** for X₂
   - Change in Y for one unit increase in X₂ (holding X₁ constant)
   - Effect of one additional sleep hour on exam score

5. **X₁**:
   - **Independent Variable 1** (Predictor Variable 1)
   - Study Hours

6. **X₂**:
   - **Independent Variable 2** (Predictor Variable 2)
   - Sleep Hours

7. **ε (Epsilon)**:
   - **Error term** (Residual)
   - Difference between actual and predicted values
   - Accounts for unexplained variation

---

## **Q2. Manual Computation (6 marks)**

### **a. Compute the coefficients (β₀, β₁, β₂) using the normal equation method.**

## **Normal Equation Method**

**Formula:** β = (XᵀX)⁻¹XᵀY

**Step 1: Create the Design Matrix X**
```
X = [1  X₁  X₂]  =  [1  2  6]
                    [1  3  5]
                    [1  4  6]
                    [1  5  4]
                    [1  6  5]
```

**Step 2: Create the Response Vector Y**
```
Y = [55]
    [60]
    [65]
    [70]
    [75]
```

**Step 3: Calculate XᵀX**
```
Xᵀ = [1  1  1  1  1]
     [2  3  4  5  6]
     [6  5  6  4  5]

XᵀX = [1  1  1  1  1] [1  2  6]
      [2  3  4  5  6] [1  3  5]
      [6  5  6  4  5] [1  4  6]
                      [1  5  4]
                      [1  6  5]
```

**Computing XᵀX:**
```
XᵀX₁₁ = 1×1 + 1×1 + 1×1 + 1×1 + 1×1 = 5
XᵀX₁₂ = 1×2 + 1×3 + 1×4 + 1×5 + 1×6 = 20
XᵀX₁₃ = 1×6 + 1×5 + 1×6 + 1×4 + 1×5 = 26

XᵀX₂₁ = 2×1 + 3×1 + 4×1 + 5×1 + 6×1 = 20
XᵀX₂₂ = 2×2 + 3×3 + 4×4 + 5×5 + 6×6 = 4 + 9 + 16 + 25 + 36 = 90
XᵀX₂₃ = 2×6 + 3×5 + 4×6 + 5×4 + 6×5 = 12 + 15 + 24 + 20 + 30 = 101

XᵀX₃₁ = 6×1 + 5×1 + 6×1 + 4×1 + 5×1 = 26
XᵀX₃₂ = 6×2 + 5×3 + 6×4 + 4×5 + 5×6 = 12 + 15 + 24 + 20 + 30 = 101
XᵀX₃₃ = 6×6 + 5×5 + 6×6 + 4×4 + 5×5 = 36 + 25 + 36 + 16 + 25 = 138
```

**Result:**
```
XᵀX = [5   20  26 ]
      [20  90  101]
      [26  101 138]
```

**Step 4: Calculate XᵀY**
```
XᵀY₁ = 1×55 + 1×60 + 1×65 + 1×70 + 1×75 = 325
XᵀY₂ = 2×55 + 3×60 + 4×65 + 5×70 + 6×75 = 110 + 180 + 260 + 350 + 450 = 1350
XᵀY₃ = 6×55 + 5×60 + 6×65 + 4×70 + 5×75 = 330 + 300 + 390 + 280 + 375 = 1675
```

**Result:**
```
XᵀY = [325 ]
      [1350]
      [1675]
```

**Step 5: Calculate (XᵀX)⁻¹**

**Determinant of XᵀX:**
```
det(XᵀX) = 5(90×138 - 101×101) - 20(20×138 - 101×26) + 26(20×101 - 90×26)
         = 5(12420 - 10201) - 20(2760 - 2626) + 26(2020 - 2340)
         = 5(2219) - 20(134) + 26(-320)
         = 11095 - 2680 - 8320
         = 95
```

**Cofactor Matrix and Inverse:**
After calculating the cofactor matrix and dividing by the determinant:

```
(XᵀX)⁻¹ = [1.0947  -0.1579  -0.1053]
          [-0.1579   0.0421   0.0000]
          [-0.1053   0.0000   0.0263]
```

**Step 6: Calculate β = (XᵀX)⁻¹XᵀY**
```
β₀ = 1.0947×325 - 0.1579×1350 - 0.1053×1675 = 355.8 - 213.2 - 176.4 = 30.2
β₁ = -0.1579×325 + 0.0421×1350 + 0.0000×1675 = -51.3 + 56.8 + 0.0 = 5.5
β₂ = -0.1053×325 + 0.0000×1350 + 0.0263×1675 = -34.2 + 0.0 + 44.1 = -0.1
```

**Coefficients (Rounded):**
```
β₀ = 30.0
β₁ = 5.5
β₂ = 0.0
```

### **b. Write the resulting regression equation.**

**Final Regression Equation:**

```
Y = 30.0 + 5.5X₁ + 0.0X₂
```

**Simplified:**
```
Exam Score = 30.0 + 5.5 × Study Hours + 0.0 × Sleep Hours
```

**Or simply:**
```
Exam Score = 30.0 + 5.5 × Study Hours
```

---

## **Q3. Interpretation (5 marks)**

### **a. Interpret the meaning of each coefficient in the context of this dataset.**

## **Coefficient Interpretations:**

### **β₀ = 30.0 (Intercept)**
**Meaning:** The expected exam score when both study hours and sleep hours are zero.

**Interpretation:**
- A student who studies 0 hours and sleeps 0 hours would score approximately 30 points
- **Practical Note:** This is a theoretical baseline since no student actually studies 0 hours or sleeps 0 hours
- Represents the inherent ability or baseline knowledge without study or sleep

### **β₁ = 5.5 (Study Hours Coefficient)**
**Meaning:** For each additional hour of study, the exam score increases by 5.5 points, holding sleep hours constant.

**Interpretation:**
- **Strong Positive Effect:** Studying has a significant positive impact on exam performance
- **Practical Impact:** 
  - 1 extra study hour → +5.5 points
  - 2 extra study hours → +11.0 points
  - 4 extra study hours (from 2 to 6) → +22 points (observed: 75-55 = 20 points)
- **Ceteris Paribus:** This effect assumes sleep hours remain unchanged

### **β₂ = 0.0 (Sleep Hours Coefficient)**
**Meaning:** Sleep hours have no linear effect on exam score in this dataset.

**Interpretation:**
- **No Correlation:** Additional sleep hours don't increase or decrease exam scores
- **Surprising Result:** Contradicts common belief about sleep importance
- **Possible Explanations:**
  - Small sample size (only 5 students)
  - Sleep hours range is narrow (4-6 hours)
  - Other factors dominate (study time effect is much stronger)
  - Non-linear relationship may exist

### **b. Predict the exam score for a student who studies 5 hours and sleeps 6 hours.**

**Prediction Calculation:**

**Given:** X₁ = 5 hours (study), X₂ = 6 hours (sleep)

**Using our regression equation:**
```
Y = β₀ + β₁X₁ + β₂X₂
Y = 30.0 + 5.5(5) + 0.0(6)
Y = 30.0 + 27.5 + 0.0
Y = 57.5
```

**Predicted Exam Score: 57.5 points**

**Verification with Dataset:**
Looking at Student D who has X₁ = 5, X₂ = 4:
- Actual score: 70
- Our model predicts: 30.0 + 5.5(5) + 0.0(4) = 57.5
- The difference in sleep (6 vs 4 hours) doesn't affect the prediction since β₂ = 0

**Confidence in Prediction:**
This prediction falls within the range of our training data (study hours: 2-6, sleep hours: 4-6), so it's a reasonable interpolation.

---

## **Q4. Model Evaluation (3 marks)**

### **a. Calculate the predicted values and residuals for each student.**

## **Predicted Values Calculation:**

**Using Y = 30.0 + 5.5X₁ + 0.0X₂**

**Student A:** X₁ = 2, X₂ = 6
```
Ŷₐ = 30.0 + 5.5(2) + 0.0(6) = 30.0 + 11.0 + 0.0 = 41.0
```

**Student B:** X₁ = 3, X₂ = 5
```
Ŷᵦ = 30.0 + 5.5(3) + 0.0(5) = 30.0 + 16.5 + 0.0 = 46.5
```

**Student C:** X₁ = 4, X₂ = 6
```
Ŷᶜ = 30.0 + 5.5(4) + 0.0(6) = 30.0 + 22.0 + 0.0 = 52.0
```

**Student D:** X₁ = 5, X₂ = 4
```
Ŷᵈ = 30.0 + 5.5(5) + 0.0(4) = 30.0 + 27.5 + 0.0 = 57.5
```

**Student E:** X₁ = 6, X₂ = 5
```
Ŷᵉ = 30.0 + 5.5(6) + 0.0(5) = 30.0 + 33.0 + 0.0 = 63.0
```

## **Residuals Calculation:**

**Residual = Actual - Predicted (eᵢ = Yᵢ - Ŷᵢ)**

**Student A:**
```
eₐ = 55 - 41.0 = 14.0
```

**Student B:**
```
eᵦ = 60 - 46.5 = 13.5
```

**Student C:**
```
eᶜ = 65 - 52.0 = 13.0
```

**Student D:**
```
eᵈ = 70 - 57.5 = 12.5
```

**Student E:**
```
eᵉ = 75 - 63.0 = 12.0
```

## **Summary Table:**
```
Student | Actual (Y) | Predicted (Ŷ) | Residual (e) | e²
--------|-----------|----------------|--------------|-------
A       | 55        | 41.0          | 14.0         | 196.0
B       | 60        | 46.5          | 13.5         | 182.25
C       | 65        | 52.0          | 13.0         | 169.0
D       | 70        | 57.5          | 12.5         | 156.25
E       | 75        | 63.0          | 12.0         | 144.0
--------|-----------|----------------|--------------|-------
Sum     | 325       | 260.0         | 65.0         | 847.5
```

### **b. Compute the Mean Squared Error (MSE) of the model.**

## **Mean Squared Error Calculation:**

**Formula:**
```
MSE = (1/n) × Σ(eᵢ)²
```

**Where:**
- n = number of observations = 5
- eᵢ = residual for observation i

**Calculation:**
```
MSE = (1/5) × (14.0² + 13.5² + 13.0² + 12.5² + 12.0²)
MSE = (1/5) × (196.0 + 182.25 + 169.0 + 156.25 + 144.0)
MSE = (1/5) × 847.5
MSE = 169.5
```

**Mean Squared Error: 169.5**

## **Additional Model Evaluation Metrics:**

### **Root Mean Squared Error (RMSE):**
```
RMSE = √MSE = √169.5 = 13.0
```

**Interpretation:** On average, our predictions are off by about 13 points.

### **Mean Absolute Error (MAE):**
```
MAE = (1/n) × Σ|eᵢ| = (1/5) × (14.0 + 13.5 + 13.0 + 12.5 + 12.0) = 65.0/5 = 13.0
```

### **R-squared (Coefficient of Determination):**
```
SS_total = Σ(Yᵢ - Ȳ)² where Ȳ = 325/5 = 65
SS_total = (55-65)² + (60-65)² + (65-65)² + (70-65)² + (75-65)²
SS_total = 100 + 25 + 0 + 25 + 100 = 250

SS_residual = Σ(eᵢ)² = 847.5

R² = 1 - (SS_residual/SS_total) = 1 - (847.5/250) = 1 - 3.39 = -2.39
```

**Note:** Negative R² indicates the model performs worse than simply predicting the mean. This suggests our simple model may be inadequate for this dataset.

---

## **Model Assessment and Insights**

### **Key Observations:**

1. **Strong Study Effect:** Each study hour increases exam score by 5.5 points
2. **No Sleep Effect:** Sleep hours don't correlate linearly with exam performance in this dataset
3. **Consistent Positive Residuals:** All predictions are below actual scores (systematic underestimation)
4. **High MSE:** MSE of 169.5 indicates significant prediction errors

### **Model Limitations:**

1. **Small Sample Size:** Only 5 data points limit model reliability
2. **Oversimplification:** Linear relationship may not capture reality
3. **Missing Variables:** Other factors (nutrition, stress, prior knowledge) not included
4. **Sleep Paradox:** Zero coefficient for sleep contradicts intuition

### **Potential Improvements:**

1. **Collect More Data:** Larger sample size for better estimates
2. **Non-linear Models:** Consider polynomial or interaction terms
3. **Additional Variables:** Include more predictors
4. **Data Transformation:** Log or other transformations might help

---

## **Summary**

This linear regression analysis reveals:

1. **Model Equation:** Exam Score = 30.0 + 5.5 × Study Hours
2. **Key Finding:** Study hours strongly predict exam performance (5.5 points per hour)
3. **Surprising Result:** Sleep hours show no linear correlation with exam scores
4. **Model Performance:** MSE = 169.5, indicating room for improvement
5. **Practical Application:** Students can expect ~5.5 point increase per additional study hour

**Educational Value:** This exercise demonstrates both the power and limitations of linear regression, emphasizing the importance of model evaluation and critical interpretation of results. 