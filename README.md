# 🎬 Analyse de sentiments des avis de films

Projet de **Natural Language Processing (NLP)** visant à analyser automatiquement le sentiment exprimé dans des avis de films et à classer chaque avis comme **positif** ou **négatif**.

Le projet couvre l'ensemble du workflow Machine Learning appliqué au texte : **nettoyage des données, exploration, représentation TF-IDF, entraînement de plusieurs modèles, évaluation et prédiction sur de nouveaux avis**.

---

## 🎯 Objectif du projet

L'objectif est de construire un modèle capable de déterminer automatiquement si un avis de film exprime un sentiment :

* 🟢 **Positif**
* 🔴 **Négatif**

Le projet permet également de comparer plusieurs algorithmes de classification afin d'identifier leurs performances sur la tâche de sentiment analysis.

---

## 🔄 Workflow du projet

```text
Dataset d'avis de films
        ↓
Nettoyage du texte
        ↓
Suppression des stopwords
        ↓
Exploration des données
        ↓
TF-IDF
        ↓
Séparation Train / Test
        ↓
Entraînement des modèles
        ↓
Évaluation
        ↓
Comparaison des performances
        ↓
Prédiction sur de nouveaux avis
```

---

## 🧹 1. Prétraitement du texte

Les avis sont nettoyés avant leur utilisation par les modèles.

Les principales étapes sont :

* suppression des balises HTML ;
* suppression des caractères spéciaux ;
* conversion en minuscules ;
* normalisation des espaces ;
* suppression des mots vides (*stopwords*).

Exemple :

```python
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

Les stopwords anglais sont ensuite supprimés avec **NLTK**.

---

## 📊 2. Exploration des données

Une exploration textuelle a été réalisée afin d'identifier les mots les plus fréquents dans les avis positifs et négatifs.

Des **WordClouds** ont été générés pour visualiser les termes fréquemment utilisés dans chaque catégorie.

---

## 🔢 3. Représentation TF-IDF

Les textes sont transformés en représentation numérique grâce à **TF-IDF (Term Frequency–Inverse Document Frequency)**.

```python
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data['review_clean'])
```

Cette représentation permet aux algorithmes de Machine Learning de travailler avec les informations textuelles sous forme de vecteurs numériques.

---

## 🤖 4. Modèles testés

Trois algorithmes de classification ont été entraînés et comparés :

### Naive Bayes

```python
model_nb = MultinomialNB()

model_nb.fit(X_train, y_train)

y_pred_nb = model_nb.predict(X_test)
```

### Régression Logistique

```python
model_lr = LogisticRegression(max_iter=1000)

model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
```

### SVM linéaire

```python
model_svm = LinearSVC()

model_svm.fit(X_train, y_train)

y_pred_svm = model_svm.predict(X_test)
```

---

## 📏 5. Évaluation

Les modèles sont évalués à l'aide de plusieurs métriques :

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Matrice de confusion**

Exemple :

```python
print("Accuracy :", accuracy_score(y_test, y_pred_nb))

print(
    classification_report(
        y_test,
        y_pred_nb
    )
)
```

Une matrice de confusion est également utilisée pour analyser les classifications correctes et incorrectes.

---

## 🔮 6. Prédiction sur de nouveaux avis

Une fonction permet d'utiliser le modèle entraîné sur de nouveaux textes.

```python
def predire_sentiment(texte, model, vectorizer):

    texte_nettoye = clean_text(texte)
    texte_nettoye = remove_stopwords(texte_nettoye)

    texte_vectorise = vectorizer.transform([texte_nettoye])

    prediction = model.predict(texte_vectorise)[0]
    proba = model.predict_proba(texte_vectorise)[0]

    resultat = "POSITIF" if prediction == 1 else "NÉGATIF"
    confiance = max(proba) * 100

    return f"{resultat} (confiance : {confiance:.1f}%)"
```

Exemples :

```python
predire_sentiment(
    "This movie was absolutely amazing, I loved every minute!",
    model_lr,
    vectorizer
)
```

et :

```python
predire_sentiment(
    "Terrible film, a complete waste of time.",
    model_lr,
    vectorizer
)
```

Le modèle retourne la classe prédite ainsi qu'un score de confiance basé sur les probabilités du modèle.

---

## 🖥️ 7. Interface de démonstration

Une interface **Streamlit** a également été développée afin de permettre à un utilisateur de saisir directement un avis de film et d'obtenir une prédiction.

```text
Avis utilisateur
       ↓
Nettoyage
       ↓
TF-IDF
       ↓
Modèle entraîné
       ↓
POSITIF / NÉGATIF
       ↓
Score de confiance
```

---

## 🛠️ Technologies utilisées

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **NLTK**
* **Matplotlib**
* **Seaborn**
* **WordCloud**
* **Streamlit**
* **Joblib**
* **Jupyter Notebook / Google Colab**

---

## 📁 Structure du projet

```text
Analyse-sentiments-NLP/
│
├── notebook/
│   └── analyse_sentiments.ipynb
│
├── images/
│   └── resultat_streamlit.png
│
├── modele_sentiment.pkl
├── vectorizer.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## 💡 Compétences démontrées

Ce projet met en pratique plusieurs compétences en **Data Science et NLP** :

* Prétraitement et nettoyage de données textuelles
* Analyse exploratoire
* Feature Engineering
* TF-IDF
* Classification supervisée
* Comparaison de modèles
* Évaluation avec plusieurs métriques
* Matrice de confusion
* Analyse de sentiments
* Sauvegarde d'un modèle avec Joblib
* Développement d'une interface de démonstration avec Streamlit

---

## 🚀 Perspectives d'amélioration

Plusieurs améliorations pourraient être envisagées :

* tester d'autres techniques de représentation textuelle ;
* optimiser les hyperparamètres des modèles ;
* comparer davantage d'algorithmes ;
* utiliser des modèles de Deep Learning ;
* expérimenter avec **BERT / Transformers** ;
* améliorer l'interface Streamlit ;
* déployer l'application pour permettre son utilisation en ligne.

---

## 👩‍💻 Auteur

**Thècle Nathalie Ramanampamonjy**

Projet réalisé dans le cadre de mon parcours en **Data Science, Machine Learning et Intelligence Artificielle**.
