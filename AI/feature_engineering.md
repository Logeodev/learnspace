# Feature Engineering

Ou comment prétraiter un _dataset_ et en extraire les données importantes, des *__insights__*.
Notamment, lorsqu'on veut réaliser du _machine learning_ sur un ensemble de données, il est important de préparer les données, ignorer des paramètres trop corrélées, rendre plus intelligible mathématiquement un paramètre, ...

## Points sensibles

On considère certains points auxquels il faut faire particulièrement attention :
- Traitement des extrêmes, les valeurs maximales ou minimales qui pourraient être aberrantes ou juste faire dévier sans valeur ajoutée nos résultats.
- Valeurs manquantes, nulles ou des zeros.
- Réduction de dimensionnalité, pour éviter la redondance et la sur-complexification des _features_.
- Le lissage des valeurs, interpoler une meilleure continuité des valeurs qui sont censées se suivre.
- La mise à l'échelle.

### Créer un modèle de comparaison

Un modèle _baseline_ est un modèle de prédiction simple qui consiste à prédire toujours la moyenne des valeurs à prédire. Ainsi, avec l'erreur moyenne réalisée avec ce modèle simple, on aura un point de comparaison : vérifier qu'on ne fait pas pire que cette _baseline_.

### Mise à l'échelle

La transformation, standardisation ou normalisation, du _dataset_ permet de mettre chaque colonne dans une même échelle et de leur donner une même "importance". 

Par exemple, si dans un même _dataset_ contient à la fois des nombres d'habitants et des températures, on aurait alors des valeurs chiffrées d'ordre de grandeur très différents. Alors dans les calculs réalisés, la comparaison entre les 2 n'aurait pas de sens ou serait du moins difficile à lire. En réalisant une opération de normalisation on les ramènes toutes deux sur une même échelle.

### Réduction de dimensionnalité

La __dimensionnalité__  d'un _dataset_ c'est le nombre de colonnes / paramètres qu'il contient. Réduire sa dimensionnalité c'est donc trouver des colonnes qui n'auront pas d'influences sur le résultats voir qui risquerait de le détériorer.

Souvent on passe pour cela par une analyse de la correlation entre les données en entrée. Le coefficient de corrélation linéare s'écrit, pour deux variables $X$ et $Y$ : 
$$
Corr(X, Y) = \frac{Cov(X, Y)}{\sigma_X \times \sigma_Y}
$$
Où $\sigma_X = E(X - E(X))$ est l'écart type individuel de la variable et $Cov(X, Y)$ = la covariance entre les 2 variables = $E((X - E(X))(Y - E(Y))) = E(XY)-E(X)E(Y)$
La valeur du coefficient de corrélation est comprise dans $[-1;1]$, en valeur absolue donc on peut le lire comme plus sa valeur est proche de 1 plus les 2 variables sont corrélées. Mais $Corr(X,Y) = 0$ n'indique néanmoins pas l'indépendance statistiques des variables, d'autres corrélations moins usuelles peuvent calculées.

En pratique, si on travaille avec le module `pandas` de Python, on abtient toutes les corrélations linéaires avec :
```py
import pandas as pd
df = pd.read_csv('some_dataset.csv')
df.corr() # donnera les corrélations entre TOUTES les colonnes du dataset
```

Ainsi on préfèrera ignorer les colonnes qui sont "trop" corrélées entre elles.

### Variables catégoriques

- Dummies / 1-hot
- Comptage
- Groupes / bins
- Clusters (par les K-means ?)

### Aggrégations

Une aggrégations (moyenne, médiane, somme, comptage, ...) peut représenter une nouvelle _feature_.

## Pre-processing

Le __pre-processing_ correspond aux étapes mises en places pour rendre le _dataset_ le plus pertinent possible pour son utilisation en _machine learning_. On rassemble notamment les points vus précédemment si besoin.

### Formatter

#### Train VS Test

La séparation du _dataset_ en _train set_ et _test set_ permet de contrôler nos actions et d'en évaluer la pertinence. On ne réalisera des actions que par l'exploration du _train set_, tandis que le _test set_ est utilisé pour l'évaluation des performances suites aux transformations mises en places.

Pour évaluer la pertinence d'un _dataset_ pour le ML, on peut mettre en place un modèle simple dont les seuls résultats qui nous intéresserons ici sont les performances statistiques (et pas ses réelles capacités en prédictions).

```py
from sklearn.model_selection import train_test_split
trainSet, testSet = train_test_split(df, test_size=0.3, random_state=0)
# on fixe le random_state pour la reproductibilité et la comparaison des résultats
```

#### Encodage

L'encodage des variables catégorielles consiste simplement à établir une relation de la catégorie (souvent une chaîne de charactère) vers une valeur chiffrée, qui elle pourra être interprétée par les processus de ML. 

```py
def encode(df):
  code = {'key1':1,
          'key2':0 }
  # les var categoriques sont les strings restantes dans le df
  for col in df.select_dtypes('object'):
    df[col] = df[col].map(code)
  return df
```

#### Nettoyage des valeurs manquantes

Les valeurs manquantes ne pourront pas être interprétées par le processus de ML. Il faut donc trouver une bonne solution pour ne plus en avoir dans le _dataset_ total. Il peut s'agir de simplement enlever les lignes qui en contiennent, les remplir par une moyenne, ou toute autre opérations qui permettrait de garder un maximum de lignes de données sans détériorer les informations du _dataset_.

On peut connaître le taux de valeurs manquantes par colonnes avec :
```py
missingRate = df.isna().sum() / df.shape[0] *100
```

Le nettoyage aura lieu alors dans :
```py
def cleanMissing(df):
  # opérations à définir sur le df pour ne plus avoir de valeurs manquantes au final
  # ex :
  df = df.dropna(axis=0) # supprime les lignes avec des valeurs manquantes
  return df
```

### Améliorer

Les actions prises dans la phase d'amélioration sont facultatives (si une amélioration n'est pas perçue) et correspondent à :
- sélection de _features_
- création de _features_ (à partir d'autres, en faisant des agrégations, des simplifications, ...)
- mise à l'échelle (rendre plus ou moins importante une variable ou les rendre comparables)
- suppression des valeurs extrêmes

Le _pre-processing_ va alors ressembler à : 
```py
def preprocessing(df):
  # toutes autres opérations voulues ici
  df = encode(df)
  df = cleanMissing(df)

  X = df.drop(['target columns'], axis=1)
  y = df[['target columns']]
  return X, y
```

### Performances

En choisissant différents modèles de classifications (`DecisionTreeClassifier`, `RandomForestClassifier`, ...), modifiant les étapes dans `preprocessing(df)`, on va pouvoir évaluer si d'un changement à l'autre on arrive à de meilleurs résultats.

```py
from sklearn.tree import RandomForestClassifier # un exemple de modèle simple de classification
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif

model = make_pipeline(SelectKBest(f_classif, k=10), # on garde les 10 features les plus importantes
                      RandomForestClassifier(random_state=0))

from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import learning_curve # pour comprendre si le modèle over/under fits

def evaluation(model):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))

  N, train_score, val_score = learning_curve(
                                             model, 
                                             X_train, 
                                             y_train, 
                                             cv=4, 
                                             scoring='f1', 
                                             train_size=np.linspace(0.1, 1, 10)
                                             )
  plt.figure(figsize=(12,8))
  plt.plot(N, train_score.mean(axis=1), label='train score')
  plt.plot(N, val_score.mean(axis=1), label='validation score')
  plt.show()

evaluation(model)
# visualiser l'importance des features dans le modèle :
pd.DataFrame(model.feature_importance_,
             index=X_train.columns).plot().bar(figsize=(12,8))
```

Avec `model.feature_importance_` on se donne une idée de quelles _features_ ont une influence sur le résultat. Selon cette indication on peut décider de nouvelles transformations (enlever une colonnes qui n'est pas importante, fusionner plusieurs de ces colonnes en une nouvelle, ...) : du _feature engineering_.

## Annexe : Optimisation de modèle

[Youtube : _Deep Learnia_](https://youtu.be/r58meM7ieaQ)

