import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

def train_and_evaluate(csv_path: str, figures_dir: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=test_size, random_state=random_state, stratify=df["label"])
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    results = {}

    
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    y_pred_nb = nb.predict(X_test_vec)
    report_nb = classification_report(y_test, y_pred_nb, output_dict=True)
    results["MultinomialNB"] = report_nb

    
    svc = LinearSVC()
    svc.fit(X_train_vec, y_train)
    y_pred_svc = svc.predict(X_test_vec)
    report_svc = classification_report(y_test, y_pred_svc, output_dict=True)
    results["LinearSVC"] = report_svc

    
    os.makedirs(figures_dir, exist_ok=True)

    
    for name, y_pred in [("MultinomialNB", y_pred_nb), ("LinearSVC", y_pred_svc)]:
        cm = confusion_matrix(y_test, y_pred, labels=["ham","spam"])
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"Confusion Matrix — {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks([0,1], ["ham","spam"])
        plt.yticks([0,1], ["ham","spam"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center', va='center')
        fig.savefig(os.path.join(figures_dir, f"cm_{name}.png"), bbox_inches='tight')
        plt.close(fig)

    
    if hasattr(nb, "predict_proba"):
        y_proba_nb = nb.predict_proba(X_test_vec)[:, list(nb.classes_).index("spam")]
        fpr, tpr, _ = roc_curve((y_test=="spam").astype(int), y_proba_nb)
        roc_auc = auc(fpr, tpr)
        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.title("ROC — MultinomialNB")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        fig.savefig(os.path.join(figures_dir, "roc_nb.png"), bbox_inches='tight')
        plt.close(fig)

        precision, recall, _ = precision_recall_curve((y_test=="spam").astype(int), y_proba_nb)
        fig = plt.figure()
        plt.plot(recall, precision)
        plt.title("Precision-Recall — MultinomialNB")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        fig.savefig(os.path.join(figures_dir, "pr_nb.png"), bbox_inches='tight')
        plt.close(fig)

    
    with open(os.path.join(figures_dir, "classification_reports.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, indent=2)

    return results
