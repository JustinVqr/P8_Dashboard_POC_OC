import re
import string
import nltk
from nltk.tokenize import word_tokenize, WordPunctTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment

from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Optimisation des téléchargements
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def segmentation_categorie(df, col_name='product_category_tree'):
    """
    Traite la colonne d'arborescence de catégories de produits et segmente les niveaux dans des colonnes séparées.
    
    Args:
    df (pd.DataFrame): Le DataFrame contenant la colonne à traiter.
    col_name (str): Le nom de la colonne contenant l'arborescence des catégories de produits.
    
    Returns:
    pd.DataFrame: Un DataFrame avec les colonnes de l'arborescence éclatées et jointes au DataFrame original.
    """
    # Fonction interne pour segmenter l'arborescence
    def segment_tree(tree):
        # Supprimer les crochets et diviser par le séparateur >>
        levels = tree.strip('[]').split('>>')
        # Enlever les espaces et les guillemets éventuels
        return [level.strip(' "').strip() for level in levels]

    # Appliquer la segmentation à chaque entrée de la colonne spécifiée
    segmented = df[col_name].apply(segment_tree)

    # Convertir en DataFrame avec colonnes séparées
    segmented_df = pd.DataFrame(segmented.tolist())

    # Renommer les colonnes
    segmented_df.columns = [f'CAT_{i+1}' for i in range(segmented_df.shape[1])]

    # Fusionner avec le DataFrame d'origine sur l'index
    df = pd.concat([df, segmented_df], axis=1)

    # Retourner le DataFrame avec les nouvelles colonnes ajoutées
    return df


def prepa_texte(text, lowercase=True, remove_digits=True, 
                tokenize_method='word_tokenize', remove_stopwords=True, 
                stopwords_list=None, stemming=False, lemmatization=False, 
                words_to_remove=None):
    """
    Fonction pour prétraiter un texte avant un modèle Word2Vec.

    Paramètres :
    - text (str) : Le texte à traiter.
    - lowercase (bool) : Si True, convertit le texte en minuscules.
    - remove_digits (bool) : Si True, supprime les chiffres.
    - tokenize_method (str) : Méthode de tokenisation à utiliser ('word_tokenize' par défaut).
    - remove_stopwords (bool) : Si True, supprime les stopwords du texte tokenisé.
    - stopwords_list (list) : Liste personnalisée de stopwords. Si None, utilise les stopwords anglais.
    - stemming (bool) : Si True, applique le stemming aux tokens.
    - lemmatization (bool) : Si True, applique la lemmatisation aux tokens.
    - words_to_remove (list) : Liste optionnelle de mots à supprimer du texte tokenisé.

    Retourne :
    - list : Une liste de tokens après traitement.
    """
    
    # Vérifier que le texte est valide
    if not isinstance(text, str) or len(text) == 0:
        raise ValueError("Le texte fourni doit être une chaîne non vide.")
    
    # Mise en minuscules
    if lowercase:
        text = text.lower()
    
    # Suppression des chiffres
    if remove_digits:
        text = re.sub(r'\d+', '', text)
    
    # Suppression de la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenisation
    if tokenize_method == 'word_tokenize':
        tokens = word_tokenize(text)
    elif tokenize_method == 'wordpunct':
        tokenizer = WordPunctTokenizer()
        tokens = tokenizer.tokenize(text)
    elif tokenize_method == 'regex':
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
    else:
        raise ValueError(f"Méthode de tokenisation invalide : {tokenize_method}")
    
    # Suppression des stopwords
    if remove_stopwords:
        if not stopwords_list:
            stopwords_list = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stopwords_list]
    
    # Suppression des mots spécifiques
    if words_to_remove:
        tokens = [token for token in tokens if token not in words_to_remove]
    
    # Application du stemming
    if stemming:
        stemmer = PorterStemmer()
        tokens = Parallel(n_jobs=-1)(delayed(stemmer.stem)(token) for token in tokens)
    
    # Application de la lemmatisation
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = Parallel(n_jobs=-1)(delayed(lemmatizer.lemmatize)(token) for token in tokens)
    
    # Retour sous forme de liste
    return list(tokens)


def get_average_word2vec(tokens_list, model, vector_size):
    """
    Retourne la moyenne des vecteurs Word2Vec pour une liste de tokens.
    Si aucun mot n'est dans le vocabulaire, retourne un vecteur nul.
    """
    # Vérifier que chaque token est bien dans le vocabulaire du modèle
    vectorized = [model.wv[token] for token in tokens_list if token in model.wv]
    
    if len(vectorized) == 0:
        # Retourner un vecteur nul si aucun des tokens n'est dans le vocabulaire
        return np.zeros(vector_size)
    
    # Retourner la moyenne des vecteurs
    return np.mean(vectorized, axis=0)



def kmeans(X_data, true_labels, label_names, n_clusters=7, random_state=42, visualize_tsne=False, **kwargs):
    """
    Effectue le clustering K-Means sur des données réduites et évalue les résultats avec le score de silhouette
    et l'ARI.

    Paramètres :
    - X_data : les données sur lesquelles appliquer le K-Means.
    - true_labels : les vraies étiquettes de catégories pour le calcul de l'ARI.
    - label_names : les noms réels des catégories pour l'affichage.
    - n_clusters : le nombre de clusters à former.
    - random_state : la graine aléatoire pour la reproductibilité des résultats.
    - visualize_tsne (bool) : Si True, visualise les résultats avec t-SNE.

    Retourne :
    - dict : Un dictionnaire contenant les scores de silhouette, ARI et d'accuracy.
    """
    print("Démarrage du KMeans clustering...")

    # Limitation du nombre de threads pour éviter les fuites de mémoire
    kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=random_state)
    kmeans.fit(X_data)
    clusters = kmeans.predict(X_data)

    def conf_mat_transform(y_true, y_pred):
        conf_mat = confusion_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-conf_mat)
        mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
        y_pred_transformed = pd.Series(y_pred).apply(lambda x: mapping.get(x, x))
        return y_pred_transformed

    clusters_aligned = conf_mat_transform(true_labels, clusters)

    # Ajout d'une vérification de la longueur des données pour éviter les erreurs de dimensions
    print(f'Longueur de clusters alignés: {len(clusters_aligned)}')
    print(f'Longueur de true_labels: {len(true_labels)}')

    try:
        silhouette_avg = silhouette_score(X_data, clusters_aligned)
        ari_score = adjusted_rand_score(true_labels, clusters_aligned)
        accuracy = accuracy_score(true_labels, clusters_aligned)

        # Affichage des métriques dans le notebook
        print(f'Silhouette Score: {silhouette_avg:.4f}')
        print(f'Adjusted Rand Score: {ari_score:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
    except Exception as e:
        print(f"Erreur lors du calcul des métriques : {e}")
        return

    # Affichage de la matrice de confusion
    conf_matrix = confusion_matrix(true_labels, clusters_aligned)
    plt.figure(figsize=(7, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Matrice de confusion')
    plt.xlabel('Prédit')
    plt.ylabel('Vrai')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

    # Option pour visualiser les clusters avec t-SNE
    if visualize_tsne:
        tsne_graph(X_data, clusters_aligned, label_names, perplexity=kwargs.get('perplexity', 30))

    return {
        'Silhouette Score': silhouette_avg,
        'Adjusted Rand Score': ari_score,
        'Accuracy': accuracy
    }


def tsne_graph(data, categories_encoded, category_names, perplexity):
    logging.info("Visualisation t-SNE...")
    plt.figure(figsize=(7, 6))
    unique_categories = np.unique(categories_encoded)
    category_colors = sns.color_palette('tab10', len(unique_categories))

    # Instanciation et transformation avec t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='random')
    tsne_results = tsne.fit_transform(data)

    # Création du scatter plot
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=categories_encoded,
                    palette=category_colors, legend='full', s=50, alpha=0.6)

    plt.title(f't-SNE avec perplexité = {perplexity}')
    plt.xlabel('Composante t-SNE 1')
    plt.ylabel('Composante t-SNE 2')

    # Création des éléments de la légende avec les noms des catégories
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=category_names[cat],
                                  markerfacecolor=category_colors[i], markersize=12)
                       for i, cat in enumerate(unique_categories)]

    # Affichage de la légende
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Vraies catégories")
    plt.show()

    return tsne_results


def classify_with_word2vec_embeddings(data, embedding_col, target_col, test_size=0.2, random_state=42, verbose=True):
    """
    data: DataFrame contenant les données avec les colonnes d'embeddings et de la target
    embedding_col: Nom de la colonne des embeddings
    target_col: Nom de la colonne cible (target)
    test_size: Taille du jeu de test (par défaut 0.2, soit 20%)
    random_state: Graine pour la répartition aléatoire des données
    verbose: Si True, affiche les étapes de modélisation
    """
    if verbose:
        print("Étape 1: Extraction des embeddings et de la target...")

    # Initialiser X à None
    X = None
    
    # Vérifier si les embeddings sont dans le DataFrame ou s'ils sont passés directement sous forme de matrice NumPy
    if embedding_col is not None:
        # Extraire les embeddings à partir des colonnes du DataFrame
        X = np.vstack(data[embedding_col].values)
    else:
        raise ValueError("Vous devez fournir soit une matrice X, soit des colonnes d'embeddings via embedding_col.")
    
    # Extraire la target
    y = data[target_col]

    # Récupérer les étiquettes des classes
    labels = y.unique()

    if verbose: 
        print(f"Nombre d'échantillons: {X.shape[0]}, Nombre de caractéristiques: {X.shape[1]}")

    # Diviser les données en jeu d'entraînement et de test
    if verbose: print("Étape 2: Séparation des données en jeu d'entraînement et de test...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if verbose: print(f"Jeu d'entraînement: {X_train.shape}, Jeu de test: {X_test.shape}")
    
    # Normaliser les données
    if verbose: print("Étape 3: Normalisation des données...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Appliquer la PCA pour expliquer 99% de la variance
    if verbose: print("Étape 4: Application de la PCA pour réduire la dimensionnalité...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.99, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    if verbose: 
        print(f"Nombre de composants principaux après PCA: {X_train_pca.shape[1]}")
    
    # Entraînement de la régression logistique
    if verbose: print("Étape 5: Entraînement du modèle de régression logistique...")
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train_pca, y_train)
    
    # Prédictions sur le jeu de test
    if verbose: print("Étape 6: Prédiction sur le jeu de test...")
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    y_pred = clf.predict(X_test_pca)
    
    # Calcul de l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Rapport de classification avec les noms de classes
    classif_report = classification_report(y_test, y_pred, target_names=labels)
    
    # Matrice de confusion avec les étiquettes de classes
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Affichage des résultats
    if verbose:
        print("\nÉtape 7: Résultats finaux")
        print(f"Accuracy: {accuracy}")
        print("\nClassification Report:\n", classif_report)
    
    # Affichage de la matrice de confusion avec un jeu de couleurs et les étiquettes de classes
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title('Matrice de confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités terrain')
    plt.show()

    return accuracy, classif_report, conf_matrix



def classify_with_minilm_embeddings(data, X=None, embedding_col=None, target_col='CAT_1', test_size=0.2, random_state=42, verbose=True):
    """
    Fonction pour effectuer une classification supervisée avec des embeddings générés par MiniLM.
    
    data: DataFrame contenant les données avec les colonnes d'embeddings et de la target
    X: Matrice NumPy contenant les embeddings (facultatif, utilisé si déjà fourni)
    embedding_col: Liste des colonnes des embeddings dans le DataFrame (si X n'est pas fourni)
    target_col: Nom de la colonne cible (target)
    test_size: Taille du jeu de test (par défaut 0.2, soit 20%)
    random_state: Graine pour la répartition aléatoire des données
    verbose: Si True, affiche les étapes de modélisation
    """
    if verbose:
        print("Étape 1: Extraction des embeddings et de la target...")

    # Vérifier si une matrice X est fournie ou si les embeddings sont dans le DataFrame
    if X is None:
        if embedding_col is not None:
            # Extraire les embeddings à partir des colonnes du DataFrame
            X = np.vstack(data[embedding_col].values)
        else:
            raise ValueError("Vous devez fournir soit une matrice X, soit des colonnes d'embeddings via embedding_col.")
    
    # Extraire la target (variable cible)
    y = data[target_col]
    
    # Récupérer les étiquettes des classes
    labels = y.unique()

    if verbose: 
        print(f"Nombre d'échantillons: {X.shape[0]}, Nombre de caractéristiques: {X.shape[1]}")

    # Diviser les données en jeu d'entraînement et de test
    if verbose: print("Étape 2: Séparation des données en jeu d'entraînement et de test...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if verbose: print(f"Jeu d'entraînement: {X_train.shape}, Jeu de test: {X_test.shape}")
    
    # Normaliser les données
    if verbose: print("Étape 3: Normalisation des données...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Appliquer la PCA pour expliquer 99% de la variance
    if verbose: print("Étape 4: Application de la PCA pour réduire la dimensionnalité...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.99, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    if verbose: 
        print(f"Nombre de composants principaux après PCA: {X_train_pca.shape[1]}")
    
    # Entraînement de la régression logistique
    if verbose: print("Étape 5: Entraînement du modèle de régression logistique...")
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train_pca, y_train)
    
    # Prédictions sur le jeu de test
    if verbose: print("Étape 6: Prédiction sur le jeu de test...")
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    y_pred = clf.predict(X_test_pca)
    
    # Calcul de l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Rapport de classification
    classif_report = classification_report(y_test, y_pred, target_names=labels)
    
    # Matrice de confusion avec les labels
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Affichage des résultats
    if verbose:
        print("\nÉtape 7: Résultats finaux")
        print(f"Accuracy: {accuracy}")
        print("\nClassification Report:\n", classif_report)
    
    # Affichage de la matrice de confusion avec un jeu de couleurs et les étiquettes de classes
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title('Matrice de confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités terrain')
    plt.show()

    return accuracy, classif_report, conf_matrix
