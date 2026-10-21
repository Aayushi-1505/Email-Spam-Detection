# Customer Segmentation using K-Means Clustering

A machine learning project that groups mall customers into segments based on their annual income and spending score, using the K-Means unsupervised algorithm.

## Problem Statement
A mall wants to understand its customers better so it can target marketing. Since there are no labels, we use clustering to discover natural groups.

## Dataset
[Mall Customers Dataset (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) with 200 customers: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100).

## Algorithm: K-Means
K-Means splits data into k groups. It places k centroids, assigns each point to the nearest one, moves the centroids to the group average, and repeats until stable.

## Steps
1. Load and inspect the data (Pandas)
2. Exploratory data analysis (Seaborn, Matplotlib)
3. Scale features (StandardScaler)
4. Choose k using the Elbow method and Silhouette score
5. Train the final K-Means model
6. Visualize and interpret the clusters

## How to Run
```bash
git clone https://github.com/aayushi-1505/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
pip install -r requirements.txt
python main.py
```
Place `Mall_Customers.csv` inside the `data/` folder first.

## Results

### Elbow Method and Silhouette Score
![Elbow and Silhouette](https://raw.githubusercontent.com/aayushi-1505/customer-segmentation-kmeans/main/images/elbow_silhouette.png)

### Customer Clusters
![Clusters](https://raw.githubusercontent.com/aayushi-1505/customer-segmentation-kmeans/main/images/clusters.png)

**Silhouette score: 0.555**

## Segments Found (k = 5)
- High income, high spending: best customers, target them
- High income, low spending: potential to convert
- Medium income, medium spending: average customers
- Low income, high spending: young, spend a lot
- Low income, low spending: careful spenders

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Author
Aayushi, B.Tech CSE, Asha M. Tarsadia Institute of Computer Science and Technology
