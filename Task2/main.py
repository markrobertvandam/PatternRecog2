import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ss import SemiSupervised

if __name__ == "__main__":
    df = pd.read_csv("data/creditcard.csv")
    df = df.drop(["Time", "Amount"], axis = 1)

    ss = SemiSupervised(df, neighbors = 3)

    s1, s2, s3 = ss.run()

    print("================\n")
    
    print(f"Average q2 score: {np.mean(s1)}")
    print(f"Average q3 score: {np.mean(s2)}")
    print(f"Average q4 score: {np.mean(s3)}")

    scores = [s1, s2, s3]
    titles = ["Baseline model (KNN): Epochs vs F1 score", 
              "Semi-supervised model (LinearPropogation): Epochs vs F1 score", 
              "Baseline model (KNN) training with Semi-supervised model (LinearPropogation) labels: Epochs vs F1 score"]

    for i in range(len(scores)):
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(scores[i], c="r")
        ax.set_title(titles[i])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("F1 score")

        fig.savefig(os.path.join("results", f"q{i}.png"), bbox_inches="tight")
