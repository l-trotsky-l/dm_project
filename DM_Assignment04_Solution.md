# Data Mining Assignment 04 - K-means Clustering
## Complete Solution Guide

---

## **Given Dataset**
```
Point | X   | Y
------|-----|-----
A     | 1.0 | 2.0
B     | 1.5 | 1.8
C     | 5.0 | 8.0
D     | 8.0 | 8.0
E     | 1.0 | 0.6
F     | 9.0 | 11.0
```

---

## **Q1. Understanding the Algorithm (3 marks)**

### **a. What is the purpose of K-means clustering?**

**Purpose of K-means:**
- **Data Segmentation:** Groups similar data points together
- **Pattern Recognition:** Identifies hidden patterns in data
- **Data Compression:** Reduces data complexity by representing groups with centroids
- **Market Segmentation:** Groups customers with similar behaviors
- **Image Processing:** Reduces colors in images for compression

### **b. Explain how the K-means algorithm works, including initialization, assignment, and update steps.**

**K-means Algorithm Steps:**

**Step 1: Initialization**
- Choose the number of clusters (K)
- Randomly initialize K centroids in the data space

**Step 2: Assignment**
- Calculate distance from each data point to all centroids
- Assign each point to the nearest centroid (forming clusters)

**Step 3: Update**
- Calculate new centroid position as the mean of all points in each cluster
- Move centroids to new positions

**Step 4: Repeat**
- Repeat Steps 2-3 until centroids stop moving (convergence)

**Distance Formula (Euclidean):**
```
Distance = √[(x₁ - x₂)² + (y₁ - y₂)²]
```

---

## **Q2. Manual Clustering (6 marks)**

### **a. Using K=2, randomly initialize two centroids and assign each point to the nearest cluster.**

**Step 1: Initialize Centroids (Random)**
```
Centroid C1: (2.0, 2.0)
Centroid C2: (7.0, 9.0)
```

**Step 2: Calculate Distances for Each Point**

**Point A (1.0, 2.0):**
```
Distance to C1 = √[(1.0-2.0)² + (2.0-2.0)²] = √[1 + 0] = 1.0
Distance to C2 = √[(1.0-7.0)² + (2.0-9.0)²] = √[36 + 49] = √85 = 9.22
```
**Assigned to C1** (smaller distance)

**Point B (1.5, 1.8):**
```
Distance to C1 = √[(1.5-2.0)² + (1.8-2.0)²] = √[0.25 + 0.04] = √0.29 = 0.54
Distance to C2 = √[(1.5-7.0)² + (1.8-9.0)²] = √[30.25 + 51.84] = √82.09 = 9.06
```
**Assigned to C1**

**Point C (5.0, 8.0):**
```
Distance to C1 = √[(5.0-2.0)² + (8.0-2.0)²] = √[9 + 36] = √45 = 6.71
Distance to C2 = √[(5.0-7.0)² + (8.0-9.0)²] = √[4 + 1] = √5 = 2.24
```
**Assigned to C2**

**Point D (8.0, 8.0):**
```
Distance to C1 = √[(8.0-2.0)² + (8.0-2.0)²] = √[36 + 36] = √72 = 8.49
Distance to C2 = √[(8.0-7.0)² + (8.0-9.0)²] = √[1 + 1] = √2 = 1.41
```
**Assigned to C2**

**Point E (1.0, 0.6):**
```
Distance to C1 = √[(1.0-2.0)² + (0.6-2.0)²] = √[1 + 1.96] = √2.96 = 1.72
Distance to C2 = √[(1.0-7.0)² + (0.6-9.0)²] = √[36 + 70.56] = √106.56 = 10.32
```
**Assigned to C1**

**Point F (9.0, 11.0):**
```
Distance to C1 = √[(9.0-2.0)² + (11.0-2.0)²] = √[49 + 81] = √130 = 11.40
Distance to C2 = √[(9.0-7.0)² + (11.0-9.0)²] = √[4 + 4] = √8 = 2.83
```
**Assigned to C2**

**Initial Cluster Assignment:**
```
Cluster 1 (C1): A, B, E
Cluster 2 (C2): C, D, F
```

### **b. Update the centroids and repeat the assignment. Show the process for two iterations.**

## **ITERATION 1**

**Step 3: Update Centroids**

**New C1 (mean of A, B, E):**
```
X-coordinate: (1.0 + 1.5 + 1.0) / 3 = 3.5 / 3 = 1.17
Y-coordinate: (2.0 + 1.8 + 0.6) / 3 = 4.4 / 3 = 1.47
New C1: (1.17, 1.47)
```

**New C2 (mean of C, D, F):**
```
X-coordinate: (5.0 + 8.0 + 9.0) / 3 = 22.0 / 3 = 7.33
Y-coordinate: (8.0 + 8.0 + 11.0) / 3 = 27.0 / 3 = 9.00
New C2: (7.33, 9.00)
```

**Step 4: Re-assign Points to Updated Centroids**

**Point A (1.0, 2.0):**
```
Distance to C1 = √[(1.0-1.17)² + (2.0-1.47)²] = √[0.029 + 0.281] = 0.56
Distance to C2 = √[(1.0-7.33)² + (2.0-9.00)²] = √[40.07 + 49] = 9.44
```
**Assigned to C1**

**Point B (1.5, 1.8):**
```
Distance to C1 = √[(1.5-1.17)² + (1.8-1.47)²] = √[0.109 + 0.109] = 0.47
Distance to C2 = √[(1.5-7.33)² + (1.8-9.00)²] = √[33.99 + 51.84] = 9.26
```
**Assigned to C1**

**Point C (5.0, 8.0):**
```
Distance to C1 = √[(5.0-1.17)² + (8.0-1.47)²] = √[14.67 + 42.65] = 7.57
Distance to C2 = √[(5.0-7.33)² + (8.0-9.00)²] = √[5.43 + 1] = 2.54
```
**Assigned to C2**

**Point D (8.0, 8.0):**
```
Distance to C1 = √[(8.0-1.17)² + (8.0-1.47)²] = √[46.67 + 42.65] = 9.46
Distance to C2 = √[(8.0-7.33)² + (8.0-9.00)²] = √[0.45 + 1] = 1.20
```
**Assigned to C2**

**Point E (1.0, 0.6):**
```
Distance to C1 = √[(1.0-1.17)² + (0.6-1.47)²] = √[0.029 + 0.757] = 0.89
Distance to C2 = √[(1.0-7.33)² + (0.6-9.00)²] = √[40.07 + 70.56] = 10.52
```
**Assigned to C1**

**Point F (9.0, 11.0):**
```
Distance to C1 = √[(9.0-1.17)² + (11.0-1.47)²] = √[61.31 + 90.98] = 12.34
Distance to C2 = √[(9.0-7.33)² + (11.0-9.00)²] = √[2.79 + 4] = 2.61
```
**Assigned to C2**

**Cluster Assignment after Iteration 1:**
```
Cluster 1 (C1): A, B, E  (No change)
Cluster 2 (C2): C, D, F  (No change)
```

## **ITERATION 2**

**Step 5: Update Centroids Again**

Since cluster assignments didn't change, centroids remain:
```
C1: (1.17, 1.47)
C2: (7.33, 9.00)
```

**Step 6: Check for Convergence**

Since centroids didn't move, the algorithm has **converged**.

**Final Result:**
```
Cluster 1: Points A, B, E
Cluster 2: Points C, D, F
Final Centroid 1: (1.17, 1.47)
Final Centroid 2: (7.33, 9.00)
```

---

## **Q3. Visualization (3 marks)**

### **a. Plot the dataset points and show the clusters after the final iteration.**

**ASCII Visualization:**
```
    Y
11  |           F*
10  |            
 9  |      C2●   
 8  |    C     D
 7  |            
 6  |            
 5  |            
 4  |            
 3  |            
 2  |  A         
 1  | C1● B      
 0  |  E         
    +─────────────→ X
    0 1 2 3 4 5 6 7 8 9

Legend:
• = Data Points
● = Centroids
C1 = Cluster 1 (Points A, B, E)
C2 = Cluster 2 (Points C, D, F)
```

### **b. Label the centroids on your plot.**

**Centroid Coordinates:**
```
C1 (Cluster 1 Centroid): (1.17, 1.47)
C2 (Cluster 2 Centroid): (7.33, 9.00)
```

**Cluster Characteristics:**
- **Cluster 1:** Lower-left region (close to origin)
- **Cluster 2:** Upper-right region (higher coordinates)

---

## **Q4. Evaluation and Discussion (5 marks)**

### **a. What is the within-cluster sum of squares (WCSS), and how is it used to evaluate clustering?**

**Within-Cluster Sum of Squares (WCSS):**

**Definition:** Sum of squared distances from each point to its cluster centroid.

**Formula:**
```
WCSS = Σ(distance from point to its centroid)²
```

**Calculation for our result:**

**Cluster 1 WCSS:**
```
Point A: (0.56)² = 0.31
Point B: (0.47)² = 0.22
Point E: (0.89)² = 0.79
Cluster 1 WCSS = 0.31 + 0.22 + 0.79 = 1.32
```

**Cluster 2 WCSS:**
```
Point C: (2.54)² = 6.45
Point D: (1.20)² = 1.44
Point F: (2.61)² = 6.81
Cluster 2 WCSS = 6.45 + 1.44 + 6.81 = 14.70
```

**Total WCSS = 1.32 + 14.70 = 16.02**

**Use in Evaluation:**
- **Lower WCSS = Better clustering** (points closer to centroids)
- **Compare different K values** to find optimal clustering
- **Elbow Method:** Plot WCSS vs K to find the "elbow point"

### **b. Discuss the effect of different initial centroids on the final clustering result.**

**Effects of Different Initialization:**

1. **Local Minima:** Different starting points may lead to different final clusters
2. **Convergence Speed:** Better initial positions converge faster
3. **Final WCSS:** May vary depending on initialization
4. **Cluster Quality:** Poor initialization can result in suboptimal clustering

**Example:** If we started with centroids at (1,1) and (9,9), we might get the same result, but starting with (3,3) and (4,4) could lead to different clusters.

**Solutions:**
- **K-means++:** Smart initialization method
- **Multiple Runs:** Run algorithm multiple times with different initializations
- **Choose Best Result:** Select clustering with lowest WCSS

### **c. What strategies can be used to choose the optimal number of clusters (K)?**

**1. Elbow Method:**
- Plot WCSS vs number of clusters (K)
- Look for the "elbow" where improvement slows down
- Choose K at the elbow point

**2. Silhouette Analysis:**
- Measures how similar points are within clusters vs other clusters
- Higher silhouette score = better clustering
- Choose K with highest average silhouette score

**3. Gap Statistic:**
- Compares clustering performance with random data
- Choose K where gap statistic is maximized

**4. Domain Knowledge:**
- Use business understanding or problem requirements
- Example: Customer segmentation might need 3-5 segments

**5. Cross-Validation:**
- Test clustering stability across different data samples
- Choose K that gives consistent results

---

## **Q5. Reflection (3 marks)**

### **a. Mention a real-world scenario where K-means clustering can be applied.**

**Real-World Scenario: Customer Segmentation for E-commerce**

**Application:**
- **Data:** Customer purchase history, demographics, website behavior
- **Features:** Total spend, frequency of purchases, average order value, time on site
- **Clustering:** Group customers into segments like:
  - **High Value:** Frequent buyers with high spending
  - **Bargain Hunters:** Price-sensitive customers
  - **Occasional Buyers:** Infrequent but moderate spenders

**Business Benefits:**
- **Targeted Marketing:** Customize campaigns for each segment
- **Product Recommendations:** Suggest relevant products
- **Pricing Strategies:** Different pricing for different segments
- **Inventory Management:** Stock products based on segment preferences

### **b. Describe a limitation of the K-means algorithm and suggest a possible solution.**

**Limitation: Sensitivity to Outliers**

**Problem:**
- Outliers can significantly shift centroid positions
- Single extreme point can distort entire cluster
- Results in poor clustering for main data groups

**Example:**
If we had a point at (50, 50) in our dataset, it would severely affect centroid calculations and create unnatural clusters.

**Solution: Use K-medoids (PAM - Partitioning Around Medoids)**

**How it works:**
- Uses actual data points as cluster centers (medoids) instead of calculated means
- Less sensitive to outliers since medoids are real data points
- More robust to extreme values

**Benefits:**
- **Outlier Resistance:** Medoids aren't affected by extreme values
- **Interpretability:** Cluster centers are actual data points
- **Robustness:** More stable results with noisy data

**Alternative Solutions:**
1. **Outlier Detection:** Remove outliers before clustering
2. **Robust Scaling:** Use median instead of mean for scaling
3. **DBSCAN:** Density-based clustering that handles outliers naturally

---

## **Summary**

This assignment demonstrates the key concepts of K-means clustering:

1. **Algorithm Understanding:** Initialization, assignment, and update steps
2. **Manual Implementation:** Step-by-step calculations for 2 iterations
3. **Evaluation Metrics:** WCSS for measuring cluster quality
4. **Practical Considerations:** Initialization effects and K selection
5. **Real-World Applications:** Customer segmentation and business value

**Key Takeaways:**
- K-means is simple but powerful for spherical clusters
- Initialization matters for final results
- WCSS helps evaluate clustering quality
- Multiple runs and smart initialization improve results
- Real-world applications require domain knowledge for optimal K selection