import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
# Import pour K-Medoids
from pyclustering.cluster.kmedoids import kmedoids
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Supprimer les warnings de convergence KMeans pour la démo
warnings.filterwarnings("ignore", module="sklearn")

# Configuration de la page
st.set_page_config(page_title="Comparaison d'Algorithmes de Clustering", layout="wide")

# Titre de l'application
st.title("Comparaison d'Algorithmes de Clustering")

# Initialiser l'état si nécessaire (pour conserver les données et résultats entre les interactions)
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'clustered_data' not in st.session_state:
    st.session_state.clustered_data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'elbow_data' not in st.session_state:
    st.session_state.elbow_data = None
if 'cluster_centers' not in st.session_state:
    st.session_state.cluster_centers = None
if 'medoids' not in st.session_state:
    st.session_state.medoids = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'encoded_columns' not in st.session_state:
    st.session_state.encoded_columns = []
if 'row_indices' not in st.session_state:
    st.session_state.row_indices = None

# Implémentation améliorée de DIANA (Divisive Analysis)
class ImprovedDIANA:
    def __init__(self, n_clusters=2, metric='euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.labels_ = None
        self.clusters = None
        self.division_history = []  # Pour stocker l'historique des divisions
    
    def _find_most_dissimilar_point(self, points, dist_matrix):
        """
        Trouve le point le plus dissimilaire dans un cluster
        (celui qui a la somme des distances aux autres points la plus élevée)
        """
        max_dissimilarity = -1
        most_dissimilar_idx = -1
        
        for i, point_idx in enumerate(points):
            # Calculer la somme des distances à tous les autres points
            dissimilarity = sum(dist_matrix[point_idx, points[j]] for j in range(len(points)) if j != i)
            
            if dissimilarity > max_dissimilarity:
                max_dissimilarity = dissimilarity
                most_dissimilar_idx = i
        
        return most_dissimilar_idx if most_dissimilar_idx != -1 else 0
    
    def _split_cluster(self, cluster, dist_matrix):
        """
        Divise un cluster en deux en utilisant le point le plus dissimilaire
        comme point de départ pour le nouveau cluster
        """
        if len(cluster) <= 1:
            return cluster, []
        
        # Trouver le point le plus dissimilaire
        splitter_idx_in_cluster = self._find_most_dissimilar_point(cluster, dist_matrix)
        splitter_idx = cluster[splitter_idx_in_cluster]
        
        # Initialiser les deux nouveaux clusters
        cluster1 = []  # Nouveau cluster 1
        cluster2 = [splitter_idx]  # Nouveau cluster 2 avec le point le plus dissimilaire
        
        # Assigner chaque point au cluster le plus proche
        for idx in cluster:
            if idx == splitter_idx:
                continue  # Déjà assigné à cluster2
            
            # Calculer la distance moyenne aux points dans cluster1 et cluster2
            if not cluster1:
                dist_to_cluster1 = float('inf')
            else:
                dist_to_cluster1 = np.mean([dist_matrix[idx, j] for j in cluster1])
            
            dist_to_cluster2 = np.mean([dist_matrix[idx, j] for j in cluster2])
            
            if dist_to_cluster2 < dist_to_cluster1:
                cluster2.append(idx)
            else:
                cluster1.append(idx)
        
        # Si l'un des clusters est vide, diviser le cluster de manière plus simple
        if not cluster1 or not cluster2:
            # Diviser le cluster en deux parties égales
            mid = len(cluster) // 2
            cluster1 = cluster[:mid]
            cluster2 = cluster[mid:]
        
        return cluster1, cluster2
    
    def fit_predict(self, X):
        n_samples = X.shape[0]
        
        # Calculer la matrice de distance une seule fois
        dist_matrix = squareform(pdist(X, metric=self.metric))
        
        # Initialiser avec tous les points dans un seul cluster
        clusters = [list(range(n_samples))]
        
        # Stocker l'état initial pour le dendrogramme
        self.division_history = [{'clusters': [list(range(n_samples))], 'split_idx': None}]
        
        # Continuer à diviser jusqu'à obtenir le nombre de clusters souhaité
        while len(clusters) < self.n_clusters:
            # Trouver le cluster avec le plus grand diamètre (distance maximale entre deux points)
            max_diameter = -1
            cluster_to_split_idx = -1
            
            for i, cluster in enumerate(clusters):
                if len(cluster) <= 1:
                    continue  # Ne pas diviser les clusters de taille 1
                
                # Calculer le diamètre du cluster
                diameter = 0
                for idx1 in range(len(cluster)):
                    for idx2 in range(idx1 + 1, len(cluster)):
                        diameter = max(diameter, dist_matrix[cluster[idx1], cluster[idx2]])
                
                if diameter > max_diameter:
                    max_diameter = diameter
                    cluster_to_split_idx = i
            
            # Si aucun cluster ne peut être divisé, sortir de la boucle
            if cluster_to_split_idx == -1:
                break
            
            # Diviser le cluster sélectionné
            cluster_to_split = clusters[cluster_to_split_idx]
            cluster1, cluster2 = self._split_cluster(cluster_to_split, dist_matrix)
            
            # Remplacer le cluster original par les deux nouveaux clusters
            if len(cluster1) > 0 and len(cluster2) > 0:
                clusters[cluster_to_split_idx] = cluster1
                clusters.append(cluster2)
            else:
                # Si la division a échoué, forcer une division simple
                mid = len(cluster_to_split) // 2
                clusters[cluster_to_split_idx] = cluster_to_split[:mid]
                clusters.append(cluster_to_split[mid:])
            
            # Enregistrer cette étape pour le dendrogramme
            self.division_history.append({
                'clusters': clusters.copy(),
                'split_idx': cluster_to_split_idx
            })
        
        # Convertir les clusters en labels
        self.clusters = clusters
        labels = np.zeros(n_samples, dtype=int)
        
        for i, cluster in enumerate(clusters):
            for idx in cluster:
                labels[idx] = i
        
        self.labels_ = labels
        return labels

# Fonctions pour calculer l'inertie intraclasse (WCSS) et interclasse (BCSS)
def calculate_wcss_bcss(X, labels, centroids=None):
    """
    Calcule l'inertie intraclasse (WCSS) et interclasse (BCSS)
    
    Args:
        X: Données normalisées
        labels: Étiquettes de cluster pour chaque point
        centroids: Centres des clusters (si None, ils seront calculés)
        
    Returns:
        wcss: Inertie intraclasse (Within-Cluster Sum of Squares)
        bcss: Inertie interclasse (Between-Cluster Sum of Squares)
    """
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Exclure les points de bruit (label -1) s'il y en a
    if -1 in unique_labels:
        mask = labels != -1
        X = X[mask]
        labels = labels[mask]
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
    
    # Si aucun centroïde n'est fourni, les calculer
    if centroids is None:
        centroids = np.zeros((n_clusters, n_features))
        for i, label in enumerate(unique_labels):
            centroids[i] = X[labels == label].mean(axis=0)
    
    # Calculer le centre global des données
    global_centroid = X.mean(axis=0)
    
    # Initialiser WCSS et BCSS
    wcss = 0.0
    bcss = 0.0
    
    # Calculer WCSS et BCSS
    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        n_points = len(cluster_points)
        
        # WCSS: somme des distances au carré entre chaque point et le centre de son cluster
        cluster_wcss = np.sum(np.linalg.norm(cluster_points - centroids[i], axis=1) ** 2)
        wcss += cluster_wcss
        
        # BCSS: distance au carré entre le centre du cluster et le centre global, pondérée par la taille du cluster
        cluster_bcss = n_points * np.linalg.norm(centroids[i] - global_centroid) ** 2
        bcss += cluster_bcss
    
    return wcss, bcss

# Fonction pour prétraiter les données avec encodage ordinal pour les colonnes non numériques
def preprocess_data(df, selected_features, max_rows=None, sampling_method="random"):
    """
    Prétraite les données en appliquant un encodage ordinal aux colonnes non numériques
    et une normalisation aux colonnes numériques.
    
    Args:
        df: DataFrame contenant les données
        selected_features: Liste des colonnes à utiliser
        max_rows: Nombre maximum de lignes à traiter (pour les grands datasets)
        sampling_method: Méthode d'échantillonnage ("random" ou "first")
        
    Returns:
        X_processed: Données prétraitées
        feature_names: Noms des colonnes après prétraitement
        encoders: Dictionnaire des encodeurs utilisés
        row_indices: Indices des lignes utilisées (pour les grands datasets)
    """
    # Initialiser les variables
    X_processed = []
    feature_names = []
    encoders = {}
    encoded_columns = []
    
    # Sélectionner les colonnes demandées
    df_selected = df[selected_features].copy()
    
    # Pour les grands datasets, prendre un échantillon selon la méthode choisie
    if max_rows is not None and len(df_selected) > max_rows:
        if sampling_method == "aléatoire":
            st.info(f"Utilisation d'un échantillon aléatoire de {max_rows} lignes sur {len(df_selected)} pour le clustering.")
            # Échantillonnage aléatoire avec seed fixe pour reproductibilité
            np.random.seed(42)
            row_indices = np.random.choice(len(df_selected), max_rows, replace=False)
            df_selected = df_selected.iloc[row_indices].copy()
        else:  # "premières lignes"
            st.info(f"Utilisation des {max_rows} premières lignes sur {len(df_selected)} pour le clustering.")
            row_indices = np.arange(max_rows)
            df_selected = df_selected.iloc[:max_rows].copy()
    else:
        row_indices = np.arange(len(df_selected))
    
    # Traiter chaque colonne
    for column in df_selected.columns:
        # Vérifier si la colonne est numérique
        if pd.api.types.is_numeric_dtype(df_selected[column]):
            # Colonne numérique: gérer les valeurs manquantes
            if df_selected[column].isna().any():
                df_selected[column] = df_selected[column].fillna(df_selected[column].mean())
            
            # Ajouter la colonne numérique aux données à traiter
            X_processed.append(df_selected[column].values.reshape(-1, 1))
            feature_names.append(column)
        else:
            # Colonne non numérique: appliquer l'encodage ordinal
            # Gérer les valeurs manquantes
            if df_selected[column].isna().any():
                df_selected[column] = df_selected[column].fillna(df_selected[column].mode()[0])
            
            # Créer et appliquer l'encodeur
            le = LabelEncoder()
            encoded_values = le.fit_transform(df_selected[column])
            
            # Stocker l'encodeur pour référence future
            encoders[column] = le
            encoded_columns.append(column)
            
            # Ajouter la colonne encodée aux données à traiter
            X_processed.append(encoded_values.reshape(-1, 1))
            feature_names.append(f"{column} (encodé)")
            
            # Afficher les mappings d'encodage
            st.write(f"**Encodage pour {column}:**")
            mapping_df = pd.DataFrame({
                'Valeur originale': le.classes_,
                'Encodage': range(len(le.classes_))
            })
            st.dataframe(mapping_df)
    
    # Concaténer toutes les colonnes traitées
    if X_processed:
        X_combined = np.hstack(X_processed)
    else:
        return None, [], {}, [], None
    
    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Stocker le DataFrame prétraité pour référence
    df_processed = df_selected.copy()
    
    return X_scaled, feature_names, encoders, encoded_columns, row_indices, df_processed

def plot_agnes_dendrogram(X, linkage_method="ward"):
    """
    Crée un dendrogramme pour l'algorithme AGNES (clustering hiérarchique agglomératif)
    
    Args:
        X: Données normalisées
        linkage_method: Méthode de linkage ('ward', 'complete', 'average', 'single')
        
    Returns:
        fig: Figure Plotly contenant le dendrogramme
    """
    # Calculer la matrice de linkage
    Z = scipy_linkage(X, method=linkage_method)
    
    # Créer le dendrogramme avec plotly
    fig = ff.create_dendrogram(
        X, 
        linkagefun=lambda x: Z,
        labels=None,
        color_threshold=0.7*max(Z[:,2])  # Coloration des clusters
    )
    
    # Mise en forme du graphique
    fig.update_layout(
        title=f"Dendrogramme AGNES (méthode: {linkage_method})",
        xaxis_title="Échantillons",
        yaxis_title="Distance",
        width=800,
        height=500,
        margin=dict(l=50, r=50, b=50, t=50)
    )
    
    return fig

def plot_diana_dendrogram(diana_model, X):
    """
    Crée une visualisation du processus de division pour l'algorithme DIANA
    
    Args:
        diana_model: Instance de ImprovedDIANA après fit_predict
        X: Données normalisées
        
    Returns:
        fig: Figure Plotly contenant la visualisation
    """
    if not hasattr(diana_model, 'division_history') or not diana_model.division_history:
        return None
    
    # Créer une figure avec des sous-graphiques pour chaque étape
    n_steps = len(diana_model.division_history)
    fig = make_subplots(
        rows=1, cols=n_steps,
        subplot_titles=[f"Étape {i}" for i in range(n_steps)],
        shared_yaxes=True
    )
    
    # Réduire la dimension pour la visualisation si > 2 caractéristiques
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_viz = pca.fit_transform(X)
    else:
        X_viz = X
    
    # Palette de couleurs
    colors = plt.cm.tab10.colors
    
    # Tracer chaque étape du processus de division
    for step, history in enumerate(diana_model.division_history):
        clusters = history['clusters']
        
        for i, cluster in enumerate(clusters):
            # Sélectionner une couleur pour ce cluster
            color = colors[i % len(colors)]
            
            # Tracer les points de ce cluster
            if cluster:  # Vérifier que le cluster n'est pas vide
                cluster_points = X_viz[cluster]
                fig.add_trace(
                    go.Scatter(
                        x=cluster_points[:, 0],
                        y=cluster_points[:, 1],
                        mode='markers',
                        marker=dict(
                            color=f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})',
                            size=8
                        ),
                        name=f"Cluster {i} (Étape {step})",
                        showlegend=(step == n_steps-1)  # Afficher la légende uniquement pour la dernière étape
                    ),
                    row=1, col=step+1
                )
    
    # Mise en forme du graphique
    fig.update_layout(
        title="Processus de division DIANA",
        height=400,
        width=200 * n_steps,
        margin=dict(l=50, r=50, b=50, t=80)
    )
    
    return fig

# Sidebar pour les contrôles
with st.sidebar:
    st.header("Paramètres")

    # 1. Importation des données
    st.subheader("1. Importation des données")
    uploaded_file = st.file_uploader("Choisir un fichier CSV ou Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # Tenter de charger les données avec différentes encodages si nécessaire
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else: # assumed .xlsx
                    df = pd.read_excel(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0) # Revenir au début du fichier
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                else: # assumed .xlsx
                    df = pd.read_excel(uploaded_file)


            st.success(f"Fichier chargé avec succès: {uploaded_file.name}")
            st.session_state.df = df # Stocker dans l'état de session
            # Réinitialiser les encodeurs lors du chargement d'un nouveau fichier
            st.session_state.encoders = {}
            st.session_state.encoded_columns = []
            st.session_state.df_processed = None
            st.session_state.row_indices = None

            # Afficher un aperçu des données
            if st.checkbox("Afficher un aperçu des données", key="show_data_preview"):
                st.dataframe(df.head())

        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {str(e)}")
            st.session_state.df = None
    else:
        # Données d'exemple si aucun fichier n'est téléchargé et pas déjà chargé
        if st.session_state.df is None:
            st.info("Aucun fichier chargé. Utilisation de données d'exemple (5 caractéristiques).")
            np.random.seed(42)
            # Créer des données avec plusieurs blobs pour un clustering visuellement clair
            from sklearn.datasets import make_blobs
            X_example, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
            # Ajouter quelques caractéristiques de bruit
            noise_features = np.random.randn(300, 3)
            X_example = np.hstack((X_example, noise_features))
            
            # Ajouter une colonne catégorielle pour démontrer l'encodage
            categories = np.random.choice(['A', 'B', 'C', 'D'], size=300)
            
            df_example = pd.DataFrame(X_example, columns=[f'Feature {i+1}' for i in range(X_example.shape[1])])
            df_example['Catégorie'] = categories
            
            st.session_state.df = df_example
            # Afficher un aperçu des données d'exemple par défaut
            if st.checkbox("Afficher un aperçu des données d'exemple", value=True, key="show_example_data_preview"):
                 st.dataframe(st.session_state.df.head())


    # Assurez-vous que df est défini avant de continuer
    df = st.session_state.df
    all_cols = []
    if df is not None:
        all_cols = df.columns.tolist()

    # Option pour contrôler le nombre d'instances à utiliser
    st.subheader("Contrôle des instances")
    use_sample = st.checkbox("Limiter le nombre d'instances", value=False, 
                            help="Permet de sélectionner un sous-ensemble des données pour accélérer l'analyse")
    
    max_rows = None
    sampling_method = "random"
    
    if use_sample and df is not None:
        col1_sample, col2_sample = st.columns(2)
        
        with col1_sample:
            max_rows = st.number_input(
                "Nombre d'instances à utiliser", 
                min_value=2,  # Minimum de 2 instances nécessaires pour le clustering
                max_value=len(df), 
                value=min(5000, len(df)),
                step=10,  # Pas plus petit pour plus de flexibilité
                help="Nombre de lignes du dataset à utiliser pour l'analyse"
            )

            # Vérifier que le nombre d'instances est suffisant pour l'algorithme choisi
            if max_rows < 10:
                st.warning(f"Attention: Utiliser seulement {max_rows} instances peut produire des résultats de clustering peu fiables. Un minimum de 10-20 instances est généralement recommandé.")
        
        with col2_sample:
            sampling_method = st.selectbox(
                "Méthode d'échantillonnage",
                ["aléatoire", "premières lignes"],
                help="'aléatoire' sélectionne des instances au hasard, 'premières lignes' prend les n premières lignes du dataset"
            )
        
        st.info(f"L'analyse sera effectuée sur {max_rows} instances sur un total de {len(df)} ({(max_rows/len(df)*100):.1f}% des données)")

    # 2. Choix de l'algorithme
    st.subheader("2. Choix de l'algorithme")
    algorithms_list = ["K-means", "K-Medoids", "AGNES", "DIANA", "DBSCAN"] # Ajout de DIANA

    algorithm = st.selectbox(
        "Sélectionnez un algorithme de clustering",
        algorithms_list
    )

    # 3. Paramètres supplémentaires
    st.subheader("3. Paramètres supplémentaires")

    # Sélection des caractéristiques
    st.write("Caractéristiques à utiliser pour le clustering:")
    if not all_cols:
        st.warning("Le fichier chargé ne contient aucune colonne.")
        features = []
    else:
        # Pré-sélectionner les 2 premières colonnes ou toutes si moins de 2
        default_features = all_cols[:min(len(all_cols), 2)]
        features = st.multiselect(
            "Sélectionnez les caractéristiques",
            all_cols,
            default=default_features,
            help="Sélectionnez au moins 2 caractéristiques (numériques ou catégorielles)."
        )

    # Paramètres spécifiques à l'algorithme
    if features: # Afficher les paramètres si des caractéristiques sont sélectionnées
        st.subheader("Paramètres de l'algorithme")
        if algorithm in ["K-means", "K-Medoids", "AGNES", "DIANA"]:
            n_clusters = st.slider(f"Nombre de clusters (k) pour {algorithm}", 2, 10, 3)
            
            # Paramètres spécifiques pour K-Medoids
            if algorithm == "K-Medoids":
                distance_metric = st.selectbox(
                    "Métrique de distance",
                    ["euclidean", "manhattan", "chebyshev"],
                    help="Métrique utilisée pour calculer les distances entre points"
                )
                
                tolerance = st.slider(
                    "Tolérance", 
                    min_value=0.001, 
                    max_value=0.1, 
                    value=0.01, 
                    step=0.001,
                    help="Critère de convergence (plus petit = plus précis mais plus lent)"
                )
                
                max_iter_kmedoids = st.slider(
                    "Nombre max d'itérations", 
                    min_value=10, 
                    max_value=300, 
                    value=100, 
                    step=10,
                    help="Nombre maximum d'itérations pour K-Medoids"
                )
            
            # Paramètres spécifiques pour AGNES
            if algorithm == "AGNES":
                linkage = st.selectbox("Méthode de linkage", ["ward", "complete", "average", "single"])
            
            # Paramètres spécifiques pour DIANA
            if algorithm == "DIANA":
                diana_metric = st.selectbox(
                    "Métrique de distance pour DIANA",
                    ["euclidean", "manhattan", "chebyshev"],
                    help="Métrique utilisée pour calculer les distances entre points"
                )
                
        elif algorithm == "DBSCAN":
            eps = st.slider("DBSCAN - Epsilon (distance max)", 0.1, 5.0, 0.5, 0.1)
            min_samples = st.slider("DBSCAN - Min Samples (points min dans le voisinage)", 1, 10, 5)
            st.info("Le nombre de clusters de DBSCAN est déterminé par l'algorithme et non un paramètre d'entrée fixe.")

        # Bouton pour lancer le clustering
        run_button = st.button("Lancer le Clustering", type="primary")

        # Bouton pour lancer la méthode du coude (principalement pour K-means)
        if algorithm in ["K-means", "K-Medoids"]:
            run_elbow_button = st.button(f"Exécuter la Méthode du Coude (pour {algorithm})")
        else:
            run_elbow_button = False
            st.info("La méthode du coude est généralement utilisée pour K-means et K-Medoids.")


# --- Logique de Clustering et Affichage ---
if run_button and df is not None and features:
    if len(features) < 2:
        st.warning("Veuillez sélectionner au moins 2 caractéristiques pour le clustering.")
        st.session_state.clustered_data = None
        st.session_state.metrics = None
        st.session_state.elbow_data = None # Reset elbow data
    else:
        # Prétraitement des données avec encodage ordinal pour les colonnes non numériques
        st.subheader("Prétraitement des données")
        X_scaled, feature_names, encoders, encoded_columns, row_indices, df_processed = preprocess_data(df, features, max_rows, sampling_method)
        
        # Stocker les encodeurs et les colonnes encodées dans l'état de session
        st.session_state.encoders = encoders
        st.session_state.encoded_columns = encoded_columns
        st.session_state.row_indices = row_indices
        st.session_state.df_processed = df_processed
        
        if X_scaled is None:
            st.error("Erreur lors du prétraitement des données.")
            st.session_state.clustered_data = None
            st.session_state.metrics = None
            st.session_state.elbow_data = None
        else:
            st.info(f"Exécution de l'algorithme : **{algorithm}**...")

            try:
                # Exécution de l'algorithme sélectionné
                labels = None
                metrics_dict = {"Algorithme": algorithm}

                if algorithm == "K-means":
                    metrics_dict["N clusters (input)"] = n_clusters
                    
                    # Correction: Utiliser un nombre suffisant d'initialisations et un random_state fixe
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
                    
                    # Ajuster le modèle et prédire les clusters
                    labels = model.fit_predict(X_scaled)
                    
                    # Calculer l'inertie (somme des carrés des distances)
                    metrics_dict["Inertie"] = model.inertia_
                    
                    # Calculer le score de silhouette si plus d'un cluster
                    if n_clusters > 1:
                        metrics_dict["Silhouette Score"] = silhouette_score(X_scaled, labels)
                        
                elif algorithm == "K-Medoids":
                    metrics_dict["N clusters (input)"] = n_clusters
                    metrics_dict["Distance metric"] = distance_metric
                    metrics_dict["Tolerance"] = tolerance
                    metrics_dict["Max iterations"] = max_iter_kmedoids
                    
                    # Initialisation aléatoire des médoïdes (choix de k indices aléatoires)
                    np.random.seed(42)  # Pour reproductibilité
                    initial_medoids = np.random.choice(len(X_scaled), n_clusters, replace=False).tolist()
                    
                    # Création de la matrice de distance en fonction de la métrique choisie
                    distance_matrix = squareform(pdist(X_scaled, metric=distance_metric))
                    
                    # Création et exécution du modèle K-Medoids
                    kmedoids_instance = kmedoids(distance_matrix, initial_medoids, tolerance=tolerance, max_iterations=max_iter_kmedoids)
                    kmedoids_instance.process()
                    
                    # Récupération des clusters et médoïdes
                    clusters = kmedoids_instance.get_clusters()
                    medoids_indices = kmedoids_instance.get_medoids()
                    
                    # Conversion du format de clusters en labels (format compatible avec sklearn)
                    labels = np.zeros(len(X_scaled), dtype=int)
                    for i, cluster in enumerate(clusters):
                        for point_idx in cluster:
                            labels[point_idx] = i
                    
                    # Récupération des médoïdes (centres de clusters)
                    medoids = X_scaled[medoids_indices]
                    
                    # Calcul du score de silhouette si plus d'un cluster
                    if n_clusters > 1 and len(set(labels)) > 1:
                        metrics_dict["Silhouette Score"] = silhouette_score(X_scaled, labels)

                elif algorithm == "AGNES":
                    metrics_dict["N clusters (input)"] = n_clusters
                    metrics_dict["Linkage"] = linkage
                    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                    labels = model.fit_predict(X_scaled)
                    if n_clusters > 1:
                        metrics_dict["Silhouette Score"] = silhouette_score(X_scaled, labels)
                    
                    # Stocker le modèle pour pouvoir accéder à ses attributs plus tard
                    st.session_state.agnes_model = model

                elif algorithm == "DIANA":
                    metrics_dict["N clusters (input)"] = n_clusters
                    metrics_dict["Distance metric"] = diana_metric
                    
                    # Utiliser notre implémentation améliorée de DIANA
                    model = ImprovedDIANA(n_clusters=n_clusters, metric=diana_metric)
                    labels = model.fit_predict(X_scaled)
                    
                    # Vérifier le nombre de clusters obtenus
                    n_clusters_found = len(set(labels))
                    metrics_dict["N clusters trouvés"] = n_clusters_found
                    
                    # Afficher les tailles des clusters
                    cluster_sizes = {}
                    for i in range(n_clusters_found):
                        cluster_sizes[f"Taille cluster {i}"] = np.sum(labels == i)
                    
                    st.write("**Tailles des clusters DIANA:**")
                    st.write(cluster_sizes)
                    
                    # Calcul du score de silhouette si plus d'un cluster
                    if n_clusters_found > 1:
                        metrics_dict["Silhouette Score"] = silhouette_score(X_scaled, labels)

                elif algorithm == "DBSCAN":
                    metrics_dict["Epsilon"] = eps
                    metrics_dict["Min Samples"] = min_samples
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = model.fit_predict(X_scaled)
                    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                    metrics_dict["N clusters trouvés"] = n_clusters_found
                    if -1 in labels:
                        st.info(f"DBSCAN a identifié **{np.sum(labels == -1)}** points comme bruit (label -1).")

                    # Calculate Silhouette Score for DBSCAN, excluding noise if any and if more than 1 cluster found
                    if n_clusters_found > 1:
                        core_samples_mask = np.zeros_like(labels, dtype=bool)
                        core_samples_mask[model.core_sample_indices_] = True
                        # Filter out noise points for metric calculation
                        non_noise_indices = labels != -1
                        if np.sum(non_noise_indices) > 1: # Ensure there's more than 1 point to calculate score
                            metrics_dict["Silhouette Score"] = silhouette_score(X_scaled[non_noise_indices], labels[non_noise_indices])
                        else:
                            metrics_dict["Silhouette Score"] = "N/A (Moins de 2 points non-bruit)"
                    elif n_clusters_found <= 1:
                        metrics_dict["Silhouette Score"] = "N/A (Moins de 2 clusters ou seulement du bruit)"


                # Stocker les résultats dans l'état de session
                if algorithm == "K-means":
                    st.session_state.clustered_data = {'X_scaled': X_scaled, 'labels': labels, 'algorithm': algorithm, 'cluster_centers': model.cluster_centers_, 'feature_names': feature_names}
                    st.session_state.medoids = None
                elif algorithm == "K-Medoids":
                    st.session_state.clustered_data = {'X_scaled': X_scaled, 'labels': labels, 'algorithm': algorithm, 'feature_names': feature_names}
                    st.session_state.medoids = {'indices': medoids_indices, 'coordinates': medoids}
                else:
                    st.session_state.clustered_data = {'X_scaled': X_scaled, 'labels': labels, 'algorithm': algorithm, 'feature_names': feature_names}
                    st.session_state.medoids = None
                
                st.session_state.metrics = metrics_dict
                st.session_state.elbow_data = None # Reset elbow data on regular run

            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'exécution du clustering : {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                st.session_state.clustered_data = None
                st.session_state.metrics = None
                st.session_state.elbow_data = None
                st.session_state.medoids = None


# --- Logique Méthode du Coude ---
if run_elbow_button and df is not None and features:
    if len(features) < 2:
        st.warning("Veuillez sélectionner au moins 2 caractéristiques pour la méthode du coude.")
        st.session_state.elbow_data = None
    else:
        st.info(f"Exécution de la Méthode du Coude pour **{algorithm}** sur {len(features)} caractéristiques...")

        # Prétraitement des données avec encodage ordinal pour les colonnes non numériques
        X_scaled, feature_names, encoders, encoded_columns, row_indices, df_processed = preprocess_data(df, features, max_rows, sampling_method)
        
        # Stocker les encodeurs  df_processed = preprocess_data(df, features, max_rows)
        
        # Stocker les encodeurs et les colonnes encodées dans l'état de session
        st.session_state.encoders = encoders
        st.session_state.encoded_columns = encoded_columns
        st.session_state.row_indices = row_indices
        st.session_state.df_processed = df_processed
        
        if X_scaled is None:
            st.error("Erreur lors du prétraitement des données.")
            st.session_state.elbow_data = None
        else:
            inertia_values = []
            k_range = range(1, 11) # Tester k de 1 à 10

            try:
                if algorithm == "K-means":
                    # Calculer l'inertie pour chaque valeur de k
                    for k_val in k_range:
                        if k_val == 1:  # K-means avec k=1 est trivial
                            model = KMeans(n_clusters=k_val, random_state=42, n_init=1)
                        else:
                            model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                        model.fit(X_scaled)
                        inertia_values.append(model.inertia_)
                
                elif algorithm == "K-Medoids":
                    # Créer la matrice de distance
                    distance_matrix = squareform(pdist(X_scaled, metric=distance_metric))
                    
                    # Calculer "l'inertie" pour K-Medoids (somme des distances aux médoïdes)
                    for k_val in k_range:
                        if k_val == 1:  # Skip k=0
                            continue
                        
                        # Initialisation aléatoire des médoïdes
                        np.random.seed(42)
                        initial_medoids = np.random.choice(len(X_scaled), k_val, replace=False).tolist()
                        
                        # Exécuter K-Medoids
                        kmedoids_instance = kmedoids(distance_matrix, initial_medoids, tolerance=tolerance, max_iterations=max_iter_kmedoids)
                        kmedoids_instance.process()
                        
                        # Calcul de "l'inertie" (somme des distances au médoïde le plus proche)
                        clusters = kmedoids_instance.get_clusters()
                        medoids_indices = kmedoids_instance.get_medoids()
                        
                        # Initialiser l'inertie
                        inertia = 0
                        
                        # Pour chaque cluster, calculer la somme des distances des points au médoïde
                        for cluster_idx, cluster in enumerate(clusters):
                            medoid_idx = medoids_indices[cluster_idx]
                            for point_idx in cluster:
                                inertia += distance_matrix[point_idx][medoid_idx]
                        
                        inertia_values.append(inertia)

                st.session_state.elbow_data = {'k_range': list(k_range), 'inertia': inertia_values, 'algorithm': algorithm} # Store as list for session state compatibility
                st.session_state.clustered_data = None # Reset clustering data on elbow run
                st.session_state.metrics = None # Reset metrics on elbow run
                st.session_state.medoids = None # Reset medoids on elbow run

            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'exécution de la méthode du coude : {str(e)}")
                st.session_state.elbow_data = None


# --- Affichage des résultats ---
col1, col2 = st.columns(2)

with col1:
    st.header("Visualisation du Clustering")

    if st.session_state.clustered_data:
        # Afficher des informations sur l'échantillonnage
        if st.session_state.row_indices is not None and len(st.session_state.row_indices) < len(df):
            st.info(f"Visualisation basée sur {len(st.session_state.row_indices)} instances ({(len(st.session_state.row_indices)/len(df)*100):.1f}% des données)")
        X_scaled = st.session_state.clustered_data['X_scaled']
        labels = st.session_state.clustered_data['labels']
        algorithm_name = st.session_state.clustered_data['algorithm']
        feature_names = st.session_state.clustered_data.get('feature_names', [])

        # Réduire la dimension pour la visualisation si > 2 caractéristiques
        if X_scaled.shape[1] > 2:
            st.write(f"Visualisation en 2D via PCA ({X_scaled.shape[1]}D -> 2D)")
            pca = PCA(n_components=2)
            X_viz = pca.fit_transform(X_scaled)
            xlabel = f'Composante Principale 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'
            ylabel = f'Composante Principale 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
            
            # Si K-means, transformer aussi les centres des clusters
            if algorithm_name == "K-means" and 'cluster_centers' in st.session_state.clustered_data:
                centers_viz = pca.transform(st.session_state.clustered_data['cluster_centers'])
            
            # Si K-Medoids, transformer aussi les médoïdes
            if algorithm_name == "K-Medoids" and st.session_state.medoids is not None:
                medoids_viz = pca.transform(st.session_state.medoids['coordinates'])
        else:
            X_viz = X_scaled
            xlabel = feature_names[0] if len(feature_names) > 0 else 'Feature 1'
            ylabel = feature_names[1] if len(feature_names) > 1 else 'Feature 2'
            
            # Utiliser directement les centres pour la visualisation 2D
            if algorithm_name == "K-means" and 'cluster_centers' in st.session_state.clustered_data:
                centers_viz = st.session_state.clustered_data['cluster_centers']
            
            # Utiliser directement les médoïdes pour la visualisation 2D
            if algorithm_name == "K-Medoids" and st.session_state.medoids is not None:
                medoids_viz = st.session_state.medoids['coordinates']

        fig, ax = plt.subplots(figsize=(10, 6))

        # Gérer les labels de bruit (-1) pour DBSCAN et DIANA spécifiquement
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels) if -1 not in unique_labels else len(unique_labels)-1))
        
        # Index de couleur pour itérer à travers les couleurs
        color_idx = 0
        
        for k in unique_labels:
            if k == -1:
                # Bruit représenté en noir
                col = [0, 0, 0, 1]
                marker = '.'
                label_text = 'Bruit/Non assigné'
            else:
                col = colors[color_idx]
                color_idx += 1
                marker = 'o'
                label_text = f'Cluster {k}' # Les labels sklearn commencent à 0

            class_member_mask = (labels == k)

            # Plot points
            ax.scatter(X_viz[class_member_mask, 0], X_viz[class_member_mask, 1],
                      marker=marker, c=[col], edgecolors='k', s=50,
                      label=label_text, alpha=0.7)
        
        # Afficher les centres des clusters pour K-means
        if algorithm_name == "K-means" and 'centers_viz' in locals():
            ax.scatter(centers_viz[:, 0], centers_viz[:, 1],
                      marker='X', c='red', s=200, alpha=1, 
                      label='Centres de clusters', edgecolors='k')
        
        # Afficher les médoïdes pour K-Medoids
        if algorithm_name == "K-Medoids" and 'medoids_viz' in locals():
            ax.scatter(medoids_viz[:, 0], medoids_viz[:, 1],
                      marker='*', c='red', s=300, alpha=1, 
                      label='Médoïdes', edgecolors='k')

        ax.set_title(f'Visualisation du Clustering ({algorithm_name})')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        st.pyplot(fig)

    else:
        st.info("Cliquez sur 'Lancer le Clustering' dans la barre latérale pour voir les résultats du clustering.")


with col2:
    st.header("Métriques d'Évaluation")

    if st.session_state.metrics:
        st.subheader("Métriques du dernier Clustering")
        metrics = st.session_state.metrics
        # Afficher les métriques pertinentes
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                st.metric(metric_name, f"{value:.3f}")
            else:
                st.metric(metric_name, value)

        # Ajouter d'autres métriques si possibles (WCSS et BCSS)
        if st.session_state.clustered_data and len(set(st.session_state.clustered_data['labels'])) > 1:
            try:
                # Exclure le bruit pour WCSS et BCSS si DBSCAN ou DIANA
                labels_filtered = st.session_state.clustered_data['labels']
                X_scaled_filtered = st.session_state.clustered_data['X_scaled']
                if st.session_state.clustered_data['algorithm'] in ["DBSCAN", "DIANA"]:
                    non_noise_indices = labels_filtered != -1
                    labels_filtered = labels_filtered[non_noise_indices]
                    X_scaled_filtered = X_scaled_filtered[non_noise_indices]

                if len(set(labels_filtered)) > 1 and X_scaled_filtered.shape[0] > 1:
                    # Calculer WCSS et BCSS
                    centroids = None
                    if st.session_state.clustered_data['algorithm'] == "K-means" and 'cluster_centers' in st.session_state.clustered_data:
                        centroids = st.session_state.clustered_data['cluster_centers']
                    
                    wcss, bcss = calculate_wcss_bcss(X_scaled_filtered, labels_filtered, centroids)
                    
                    st.metric("Inertie intraclasse (WCSS)", f"{wcss:.3f}", 
                             help="Plus bas est mieux. Mesure la compacité des clusters (somme des distances au carré entre chaque point et le centre de son cluster).")
                    st.metric("Inertie interclasse (BCSS)", f"{bcss:.3f}", 
                             help="Plus haut est mieux. Mesure la séparation entre les clusters (somme pondérée des distances au carré entre les centres des clusters et le centre global).")
                    
                    # Calculer le ratio BCSS/WCSS (similaire à F-statistic)
                    if wcss > 0:
                        st.metric("Ratio BCSS/WCSS", f"{bcss/wcss:.3f}", 
                                 help="Plus haut est mieux. Ratio entre la séparation des clusters et leur compacité.")

            except Exception as e:
                st.warning(f"Impossible de calculer les métriques WCSS et BCSS : {str(e)}")

        # Afficher les résultats avec les clusters
        if st.checkbox("Ajouter les clusters au jeu de données"):
            # Vérifier si nous avons utilisé un échantillon ou le dataset complet
            if st.session_state.row_indices is not None and len(st.session_state.row_indices) < len(df):
                st.warning("Les clusters sont disponibles uniquement pour l'échantillon utilisé pour le clustering.")
                df_with_clusters = st.session_state.df_processed.copy()
                df_with_clusters['Cluster'] = st.session_state.clustered_data['labels']
            else:
                # Créer une copie du DataFrame original
                df_with_clusters = df.copy()
                
                # Vérifier que les dimensions correspondent
                if len(df_with_clusters) == len(st.session_state.clustered_data['labels']):
                    df_with_clusters['Cluster'] = st.session_state.clustered_data['labels']
                else:
                    # Si les dimensions ne correspondent pas, utiliser les indices stockés
                    if st.session_state.row_indices is not None:
                        # Initialiser tous les clusters à -1 (non assigné)
                        df_with_clusters['Cluster'] = -1
                        # Assigner les clusters aux lignes qui ont été utilisées pour le clustering
                        df_with_clusters.loc[st.session_state.row_indices, 'Cluster'] = st.session_state.clustered_data['labels']
                        st.warning(f"Seules {len(st.session_state.row_indices)} lignes sur {len(df)} ont été utilisées pour le clustering.")
                    else:
                        st.error("Impossible d'ajouter les clusters au jeu de données : les dimensions ne correspondent pas.")
                        df_with_clusters = None
            
            if df_with_clusters is not None:
                st.dataframe(df_with_clusters)
                
                # Option pour télécharger le DataFrame avec les clusters
                csv = df_with_clusters.to_csv(index=False)
                st.download_button(
                    label="Télécharger les données avec clusters",
                    data=csv,
                    file_name="data_with_clusters.csv",
                    mime="text/csv"
                )


    elif st.session_state.elbow_data:
        st.subheader("Méthode du Coude")
        elbow_data = st.session_state.elbow_data
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(elbow_data['k_range'], elbow_data['inertia'], 'bo-')
        ax.set_title(f'Méthode du Coude pour {elbow_data["algorithm"]}')
        ax.set_xlabel('Nombre de Clusters (k)')
        ax.set_ylabel('Inertie')
        ax.set_xticks(elbow_data['k_range']) # Assure que tous les k testés sont affichés
        ax.grid(True, linestyle='--', alpha=0.7)

        # Optionnel : Marquer le "coude" manuellement ou automatiquement si vous implémentez une détection
        # ax.plot(5, inertia[4], 'ro', markersize=10)
        # ax.annotate('Coude (k=5)', xy=(5, inertia[4]), xytext=(5.5, inertia[4]+10),
        #              arrowprops=dict(facecolor='red', shrink=0.05))

        st.pyplot(fig)


    else:
        st.info("Cliquez sur 'Lancer le Clustering' ou 'Exécuter la Méthode du Coude' pour voir les résultats.")

# --- Affichage des dendrogrammes pour AGNES et DIANA ---
if st.session_state.clustered_data and st.session_state.clustered_data['algorithm'] in ["AGNES", "DIANA"]:
    st.header("Dendrogramme")
    
    if st.session_state.clustered_data['algorithm'] == "AGNES":
        # Récupérer la méthode de linkage utilisée
        linkage_method = st.session_state.metrics.get("Linkage", "ward")
        
        # Créer et afficher le dendrogramme AGNES
        fig_dendrogram = plot_agnes_dendrogram(
            st.session_state.clustered_data['X_scaled'], 
            linkage_method=linkage_method
        )
        st.plotly_chart(fig_dendrogram, use_container_width=True)
        
        st.info("""
        **Interprétation du dendrogramme AGNES:**
        - L'axe vertical représente la distance entre les clusters fusionnés
        - L'axe horizontal représente les échantillons
        - Chaque jonction représente une fusion de clusters
        - La hauteur de la jonction indique la dissimilarité entre les clusters fusionnés
        """)
        
    elif st.session_state.clustered_data['algorithm'] == "DIANA":
        # Vérifier si nous avons un modèle DIANA avec l'historique des divisions
        if hasattr(model, 'division_history') and model.division_history:
            # Créer et afficher la visualisation du processus de division DIANA
            fig_diana = plot_diana_dendrogram(model, st.session_state.clustered_data['X_scaled'])
            if fig_diana:
                st.plotly_chart(fig_diana, use_container_width=True)
                
                st.info("""
                **Interprétation du processus de division DIANA:**
                - Chaque étape montre l'état des clusters après une division
                - Le processus commence avec tous les points dans un seul cluster (à gauche)
                - À chaque étape, le cluster avec le plus grand diamètre est divisé
                - Les couleurs représentent les différents clusters
                """)
            else:
                st.warning("Impossible de générer la visualisation du processus de division DIANA.")
        else:
            st.warning("L'historique des divisions n'est pas disponible pour ce modèle DIANA.")

# Pied de page
st.markdown("---")
st.caption("Application de comparaison d'algorithmes de clustering (K-means, K-Medoids, AGNES, DIANA, DBSCAN)")

# Note pour l'utilisateur
st.markdown("""
**Notes :**
* Les données sont automatiquement prétraitées:
  * Les colonnes numériques sont normalisées avec StandardScaler
  * Les colonnes non numériques sont encodées avec LabelEncoder (encodage ordinal)
* Si vous sélectionnez plus de 2 caractéristiques, la visualisation est réalisée sur les 2 premières composantes principales (PCA) des données mises à l'échelle pour permettre un affichage en 2D.
* La méthode du coude est implémentée pour K-means et K-Medoids, elle aide à choisir un bon nombre de clusters ($k$). Elle calcule l'inertie (somme des distances des points au centre de leur cluster) pour différents $k$. Le "coude" sur le graphique suggère une valeur appropriée pour $k$.
* Les métriques d'évaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz) donnent une indication de la qualité du clustering (compacité et séparation des clusters).
* **K-Medoids** est similaire à K-means mais utilise des points réels du jeu de données (médoïdes) comme centres de clusters, ce qui le rend plus robuste aux valeurs aberrantes.
* **DIANA** (Divisive Analysis) est un algorithme hiérarchique descendant qui commence avec tous les points dans un seul cluster et divise récursivement les clusters jusqu'à atteindre le nombre de clusters souhaité.
* Pour les grands datasets, un échantillonnage aléatoire est utilisé pour accélérer le clustering.
""")
