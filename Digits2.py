import numpy as np
import matplotlib.pyplot as plt
import scipy.io.matlab as matlab
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# Cargar y preparar datos
# =========================

# Cargar archivo .mat
mat_file = "BigDigits.mat"
mat = matlab.loadmat(mat_file, squeeze_me=True)  # Carga y aplana estructuras innecesarias

# Leer datos e etiquetas
data = mat["data"]      # Matriz de imágenes (vectorizadas)
labs = mat["labs"] - 1  # Etiquetas ajustadas de 1–10 a 0–9

# Obtener todas las etiquetas únicas (0 al 9)
allNlabs = np.unique(labs)

# Calcular cantidad de muestras por clase
classsiz = ()
for c in allNlabs:
    classsiz += (np.size(np.nonzero(labs == c)),)  # Calcula la cantidad de muestras por clase

# Imprimir etiquetas y frecuencias
print('\n%% Etiquetas de clase: %s' % allNlabs)
print('%% Frecuencia por clase:', classsiz)  # Cambiado para evitar el error


# =========================
# Crear conjunto binario
# =========================

taska = True
myDigit = 2  # Mi dígito de interés

otherDigits = np.setdiff1d(allNlabs, myDigit)         # Otros dígitos distintos al mío
other3Digits = np.random.permutation(otherDigits)[:3]  # Selección aleatoria de 3 de ellos

# Selección de clases dependiendo del valor de 'taska'
others = other3Digits if taska else otherDigits

print("\nClase 1 (mi dígito) = %s" % myDigit)
print("Clase 2 (otros) = %s" % others)

# Crear etiquetas binarias: 0 para mi dígito, 1 para los otros
aux = labs.copy()
classone = np.in1d(labs, myDigit)
classtwo = np.in1d(labs, others)

aux[classone] = 0
aux[classtwo] = 1

# Filtrar datos y etiquetas
X = data[np.logical_or(classone, classtwo)]
y = aux[np.logical_or(classone, classtwo)]

# =========================
# Mostrar algunas imágenes
# =========================

hwmny = 20
some1 = np.random.permutation(np.where(y == 0)[0])[:hwmny]
some2 = np.random.permutation(np.where(y == 1)[0])[:hwmny]

img1 = np.reshape(X[some1, :], (28 * hwmny, 28)).T
plt.figure(figsize=(10, 3))
plt.imshow(img1, cmap=plt.cm.gray_r)
plt.xticks([])
plt.yticks([])
plt.title('Dígitos clase 0 (mi dígito) = ' + str(myDigit))
plt.show()

img2 = np.reshape(X[some2, :], (28 * hwmny, 28)).T
plt.figure(figsize=(10, 3))
plt.imshow(img2, cmap=plt.cm.gray_r)
plt.xticks([])
plt.yticks([])
plt.title('Dígitos clase 1 (otros) = ' + str(others))
plt.show()


# =========================
# Entrenar clasificador SVM
# =========================

# Dividir en entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar clasificador SVM
clf = SVC(kernel='rbf', C=1)
clf.fit(X_train, y_train)

# Predecir sobre test
y_pred = clf.predict(X_test)

# Mostrar resultados
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
