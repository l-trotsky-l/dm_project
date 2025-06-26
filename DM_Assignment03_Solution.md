# Data Mining Assignment 03 - Naive Bayes Sentiment Classification
## Complete Solution Guide

---

## **Given Dataset**
```
Review                                    | Sentiment
I loved the movie, it was fantastic!     | Positive
The film was boring and too long.        | Negative
An excellent and gripping story.         | Positive
Poor acting and terrible script.         | Negative
Amazing performance by the lead actor.   | Positive
Not worth the time, completely dull.     | Negative
```

---

## **Q1. Understanding Naive Bayes (3 marks)**

### **a. Briefly explain Bayes' theorem and how it applies to text classification.**

**Bayes' Theorem:**
```
P(A|B) = P(B|A) × P(A) / P(B)
```

**For Text Classification:**
- A = Sentiment class (Positive/Negative)
- B = Document (review text)

**Formula:**
```
P(Sentiment|Review) = P(Review|Sentiment) × P(Sentiment) / P(Review)
```

**Application:** We calculate the probability of each sentiment class given the words in a review, then choose the class with highest probability.

### **b. Why is the Naive Bayes classifier considered 'naive'?**

It assumes **conditional independence** between features (words). This means:
- Each word contributes independently to the classification
- The presence of one word doesn't affect the probability of another word
- This assumption is "naive" because in reality, words in text are often dependent on each other

---

## **Q2. Preprocessing (4 marks)**

### **a. Tokenize the reviews and remove stop words.**

**Step 1: Tokenization**
```
Review 1: [I, loved, the, movie, it, was, fantastic]
Review 2: [The, film, was, boring, and, too, long]
Review 3: [An, excellent, and, gripping, story]
Review 4: [Poor, acting, and, terrible, script]
Review 5: [Amazing, performance, by, the, lead, actor]
Review 6: [Not, worth, the, time, completely, dull]
```

**Step 2: Remove Stop Words**
*Stop words: [I, the, it, was, and, too, an, by, not]*

```
Review 1 (Positive): [loved, movie, fantastic]
Review 2 (Negative): [film, boring, long]
Review 3 (Positive): [excellent, gripping, story]
Review 4 (Negative): [poor, acting, terrible, script]
Review 5 (Positive): [amazing, performance, lead, actor]
Review 6 (Negative): [worth, time, completely, dull]
```

### **b. Create a vocabulary of words from the processed corpus.**

**Complete Vocabulary:**
```
{loved, movie, fantastic, film, boring, long, excellent, gripping, story, 
 poor, acting, terrible, script, amazing, performance, lead, actor, 
 worth, time, completely, dull}
```

**Total vocabulary size: 21 unique words**

---

## **Q3. Training (6 marks)**

### **a. Calculate the prior probabilities for each class (Positive, Negative).**

**Step-by-step calculation:**

**Count documents by class:**
- Positive reviews: 3 (Reviews 1, 3, 5)
- Negative reviews: 3 (Reviews 2, 4, 6)
- Total reviews: 6

**Prior Probabilities:**
```
P(Positive) = Number of Positive Reviews / Total Reviews = 3/6 = 0.5
P(Negative) = Number of Negative Reviews / Total Reviews = 3/6 = 0.5
```

### **b. Compute the conditional probabilities (likelihood) of selected words using Laplace smoothing.**

**Selected words for demonstration: "fantastic", "boring", "excellent", "terrible"**

**Formula with Laplace Smoothing:**
```
P(word|class) = (Count of word in class + 1) / (Total words in class + Vocabulary size)
```

**Step 1: Count words in each class**

**Positive class words:**
- Review 1: loved, movie, fantastic
- Review 3: excellent, gripping, story  
- Review 5: amazing, performance, lead, actor
- **Total positive words: 10**

**Negative class words:**
- Review 2: film, boring, long
- Review 4: poor, acting, terrible, script
- Review 6: worth, time, completely, dull
- **Total negative words: 11**

**Step 2: Calculate conditional probabilities**

**For "fantastic":**
```
P(fantastic|Positive) = (1 + 1) / (10 + 21) = 2/31 = 0.0645
P(fantastic|Negative) = (0 + 1) / (11 + 21) = 1/32 = 0.0313
```

**For "boring":**
```
P(boring|Positive) = (0 + 1) / (10 + 21) = 1/31 = 0.0323
P(boring|Negative) = (1 + 1) / (11 + 21) = 2/32 = 0.0625
```

**For "excellent":**
```
P(excellent|Positive) = (1 + 1) / (10 + 21) = 2/31 = 0.0645
P(excellent|Negative) = (0 + 1) / (11 + 21) = 1/32 = 0.0313
```

**For "terrible":**
```
P(terrible|Positive) = (0 + 1) / (10 + 21) = 1/31 = 0.0323
P(terrible|Negative) = (1 + 1) / (11 + 21) = 2/32 = 0.0625
```

---

## **Q4. Prediction (4 marks)**

### **a. & b. Classify the review: "The movie was not interesting and poorly acted."**

**Step 1: Tokenization and Preprocessing**
```
Original: "The movie was not interesting and poorly acted"
Tokens: [The, movie, was, not, interesting, and, poorly, acted]
After removing stop words: [movie, interesting, poorly, acted]
```

**Step 2: Calculate Conditional Probabilities for each word**

**For each word, we need P(word|Positive) and P(word|Negative):**

**"movie":**
```
P(movie|Positive) = (1 + 1) / (10 + 21) = 2/31 = 0.0645
P(movie|Negative) = (0 + 1) / (11 + 21) = 1/32 = 0.0313
```

**"interesting" (not in training set):**
```
P(interesting|Positive) = (0 + 1) / (10 + 21) = 1/31 = 0.0323
P(interesting|Negative) = (0 + 1) / (11 + 21) = 1/32 = 0.0313
```

**"poorly" (not in training set):**
```
P(poorly|Positive) = (0 + 1) / (10 + 21) = 1/31 = 0.0323
P(poorly|Negative) = (0 + 1) / (11 + 21) = 1/32 = 0.0313
```

**"acted" (not in training set):**
```
P(acted|Positive) = (0 + 1) / (10 + 21) = 1/31 = 0.0323
P(acted|Negative) = (0 + 1) / (11 + 21) = 1/32 = 0.0313
```

**Step 3: Calculate Posterior Probabilities**

**For Positive Class:**
```
P(Positive|Review) ∝ P(Positive) × P(movie|Positive) × P(interesting|Positive) × P(poorly|Positive) × P(acted|Positive)
= 0.5 × 0.0645 × 0.0323 × 0.0323 × 0.0323
= 0.5 × 2.17 × 10^-6
= 1.09 × 10^-6
```

**For Negative Class:**
```
P(Negative|Review) ∝ P(Negative) × P(movie|Negative) × P(interesting|Negative) × P(poorly|Negative) × P(acted|Negative)
= 0.5 × 0.0313 × 0.0313 × 0.0313 × 0.0313
= 0.5 × 9.61 × 10^-7
= 4.81 × 10^-7
```

**Step 4: Classification Decision**
```
P(Positive|Review) = 1.09 × 10^-6
P(Negative|Review) = 4.81 × 10^-7

Since P(Positive|Review) > P(Negative|Review)
```

**Classification Result: POSITIVE**

*Note: This result seems counterintuitive given the negative words "not interesting" and "poorly acted". This highlights the limitations of our small training set.*

---

## **Q5. Reflection (3 marks)**

### **a. Advantages and limitations of using Naive Bayes for sentiment analysis**

**Advantages:**
1. **Simple and Fast:** Easy to implement and computationally efficient
2. **Small Training Data:** Works well with limited training examples
3. **Handles Missing Features:** Gracefully handles words not seen in training
4. **Probabilistic Output:** Provides probability estimates, not just classifications
5. **Baseline Performance:** Good starting point for text classification tasks

**Limitations:**
1. **Independence Assumption:** Ignores word dependencies and context
2. **Feature Correlation:** Cannot capture relationships between words
3. **Data Sparsity:** Performance degrades with very small training sets
4. **Negation Handling:** Struggles with negations (e.g., "not good")
5. **Context Ignorance:** Cannot understand sarcasm or complex linguistic patterns

### **b. Suggest one improvement to increase classifier accuracy**

**Improvement: N-gram Features**

**Explanation:**
Instead of using only individual words (unigrams), use combinations of consecutive words (bigrams, trigrams):

**Example:**
- Unigrams: [not, interesting]
- Bigrams: [not_interesting]

**Benefits:**
- Captures some word dependencies
- Better handles negations ("not interesting" vs. "interesting")
- Improves context understanding
- Maintains computational efficiency

**Implementation:**
```
"not interesting" → features: [not, interesting, not_interesting]
```

This helps the classifier learn that "not_interesting" is different from just "interesting", leading to better sentiment classification.

---

## **Summary**

This assignment demonstrates the fundamental concepts of Naive Bayes classification:
1. **Preprocessing:** Tokenization and stop word removal
2. **Training:** Calculating prior and conditional probabilities
3. **Prediction:** Applying Bayes' theorem for classification
4. **Evaluation:** Understanding strengths and limitations

The small dataset size affects performance, but the methodology remains valid for larger, real-world applications. 