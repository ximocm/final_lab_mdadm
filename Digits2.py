import numpy as np
import pandas as pd
import scipy.io.matlab as matlab
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def plot_roc_curve(y_true, y_scores, name=None):
    print(f"Generando curva ROC para {name}...")
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"curva_roc_{name}.png")
    plt.close()

# =========================
# Cargar dataset .mat
# =========================
print("Cargando archivo BigDigits.mat...")
mat_file = "BigDigits.mat"
mat = matlab.loadmat(mat_file, squeeze_me=True)

# Leer datos e etiquetas
data = mat["data"]
labs = mat["labs"] - 1 

# =========================
# Crear conjunto binario
# =========================
print("Preparando conjunto binario...")
myDigit = 2
others = np.setdiff1d(np.unique(labs), myDigit)

aux = labs.copy()
classone = np.isin(labs, myDigit)
classtwo = np.isin(labs, others)

aux[classone] = 0
aux[classtwo] = 1

X = data[np.logical_or(classone, classtwo)]
y = aux[np.logical_or(classone, classtwo)]

# =========================
# Dividir conjunto de datos
# =========================
print("Dividiendo conjunto en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# Clasificadores a probar
# =========================
models = {
    "SVM (RBF)": SVC(kernel='rbf', probability=True),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier()
}

# =========================
# Entrenar y guardar resultados
# =========================
results = []
cmresults = []

for name, clf in models.items():
    print(f"\nEntrenando modelo: {name}")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Obtener puntuaciones para curva ROC
    if hasattr(clf, "predict_proba"):
        y_scores = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_scores = clf.decision_function(X_test)
    else:
        y_scores = y_pred  # fallback no ideal

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    row = {
        "Model": name,
        "Precision (0)": report["0"]["precision"],
        "Recall (0)": report["0"]["recall"],
        "F1-score (0)": report["0"]["f1-score"],
        "Support (0)": report["0"]["support"],
        "Precision (1)": report["1"]["precision"],
        "Recall (1)": report["1"]["recall"],
        "F1-score (1)": report["1"]["f1-score"],
        "Support (1)": report["1"]["support"],
        "Accuracy": report["accuracy"],
        "Macro Avg F1": report["macro avg"]["f1-score"],
        "Weighted Avg F1": report["weighted avg"]["f1-score"],
    }
    results.append(row)
    cmresults.append((name, cm))

    # Graficar curva ROC
    plot_roc_curve(y_test, y_scores, name=name)

# =========================
# Guardar resultados en CSV
# =========================
print("\nGuardando métricas en results.csv...")
df_results = pd.DataFrame(results)
df_results.to_csv("results.csv", index=False)

# =========================
# Imprimir y guardar matrices de confusión
# =========================
for name, cm in cmresults:
    print(f"\nConfusion Matrix for {name}:")
    print(cm)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {name}')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [str(myDigit), 'Not ' + str(myDigit)])
    plt.yticks(tick_marks, [str(myDigit), 'Not ' + str(myDigit)])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name}.png')
    plt.close()
