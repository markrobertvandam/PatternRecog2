import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from sklearn.preprocessing import StandardScaler

def getClassRatio(data):
    res = []
    for y in data:
        c = Counter(y)
        res.append(round(c[0]/c[1]))
    
    return res

def experiment(df, i):
    
    print(f"Run: {i}")
    
    RANDOM_STATE = 42
    
    X1 = df.iloc[:, 0:-1].values
    y1 = df.iloc[:, -1].values
    
    # Over and Under sampling
    over = SMOTE(sampling_strategy = 0.1)
    under = RandomUnderSampler(sampling_strategy = 0.5)
    steps = [("o", over), ("u", under)]
    pipeline = Pipeline(steps = steps)
    X2, y2 = pipeline.fit_resample(X1, y1)
    
    scaler = StandardScaler()
    
    X2 = scaler.fit_transform(X2)
    
    # Splitting the data
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X2, y2, test_size = 0.2, stratify = y2)
    X_TRAIN_UNLAB, X_TRAIN_LAB, Y_TRAIN_UNLAB, Y_TRAIN_LAB = train_test_split(X_TRAIN, Y_TRAIN, test_size = 0.3, stratify = Y_TRAIN)
    
    print(f"Class ratio: {getClassRatio([Y_TRAIN_UNLAB, Y_TRAIN_LAB])}")
    
    #Q2
    # baseline_model = SVC(kernel="rbf", C = 41)
    baseline_model = KNeighborsClassifier(3)
    baseline_model.fit(X_TRAIN_LAB, Y_TRAIN_LAB)
    pred = baseline_model.predict(X_TEST)
    score1 = f1_score(Y_TEST, pred)
    print(f"Baseline score: {score1}")

    #Q3
    semi_supervised = LabelPropagation(kernel='knn', n_neighbors = 3)
#     y_temp = Y_TRAIN_UNLAB

#     rng = np.random.RandomState(RANDOM_STATE)
#     random_unlabeled_points = rng.rand(len(y_temp)) < 0.3
#     y_temp[random_unlabeled_points] = -1
    
    Y_TRAIN_UNLAB[:] = -1
    x_temp = np.concatenate((X_TRAIN_LAB ,X_TRAIN_UNLAB))
    y_temp = np.concatenate((Y_TRAIN_LAB, Y_TRAIN_UNLAB))
    
#     semi_supervised.fit(X_TRAIN_LAB, Y_TRAIN_LAB)
#     semi_supervised.fit(X_TRAIN_UNLAB, y_temp)
    semi_supervised.fit(x_temp, y_temp)
    pred = semi_supervised.predict(X_TEST)
    score2 = f1_score(Y_TEST, pred)
    print(f"Semi-supervised score: {score2}")

    #Q4
    Y_SEMI = semi_supervised.transduction_
    print(f"Class ratio after transduction : {Counter(Y_SEMI)}")
#     y5 = semi_supervised.transduction_
#     print(np.unique(y4))
#     y4 = semi_supervised.predict(X_TRAIN)

    b2 = KNeighborsClassifier(3)
    b2.fit(x_temp, Y_SEMI)

    pred = b2.predict(X_TEST)
    score3 = f1_score(Y_TEST, pred)
    
    print(f"Baseline with labels score: {score3}")
    print("--------------------")
    
    return score1, score2, score3

if __name__ == "__main__":
    df = pd.read_csv("data/creditcard.csv")
    df = df.drop(["Time", "Amount"], axis = 1)

    q2_scores = []
    q3_scores = []
    q4_scores = []

    for i in range(100):
        s1, s2, s3 = experiment(df, i)
        q2_scores.append(s1)
        q3_scores.append(s2)
        q4_scores.append(s3)
    
    scores = [q2_scores, q3_scores, q4_scores]
    titles = ["Baseline model (SVC): Epochs vs F1 score", 
            "Semi-supervised model (LinearPropogation): Epochs vs F1 score", 
            "Baseline model (SVC) training with Semi-supervised model (LinearPropogation) labels: Epochs vs F1 score"]

    for i in range(len(scores)):
        fig, ax = plt.subplots(figsize = (15, 8))
        ax.plot(scores[i], c = "r")
        ax.set_title(titles[i])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("F1 score")

        fig.savefig(f"results/q{i}.png", bbox_inches = "tight")