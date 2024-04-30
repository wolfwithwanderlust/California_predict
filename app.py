import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import math
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn import set_config
import joblib
from sklearn.preprocessing import RobustScaler


logement = pd.read_csv("housing-train-data.csv")
logement = logement.drop(columns=['Unnamed: 0'])

st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)
if page == pages[0] :

    st.write("### Contexte du Projet : L'IA au service des agents immobiliers - Silicon Valley")

    st.write("#### Introduction")
    st.write("Bienvenue à cette présentation sur notre projet d'automatisation de l'estimation des prix immobiliers. Au sein de notre startup de la Silicon Valley, nous avons été confrontés à un défi crucial : répondre à la demande croissante de nos clients en matière d'estimations immobilières personnalisées.")

    st.write("#### Objectifs")
    st.write("Notre objectif principal est de développer un modèle prédictif précis pour estimer les prix médians des logements dans les districts de Californie, en utilisant les données du recensement de 1990. De plus, nous allons créer une application web permettant à nos clients d'accéder facilement à nos prédictions. Enfin, nous évaluerons rigoureusement notre modèle avec des données de validation pour garantir sa fiabilité.")

    st.write("#### Données Disponibles")
    st.write("Nous disposons d'une base de données comprenant des informations cruciales sur les districts californiens, notamment :")
    st.write("- Longitude et Latitude")
    st.write("- Âge médian des maisons dans un pâté de maisons")
    st.write("- Nombre total de chambres et de chambres à coucher dans un bloc")
    st.write("- Population et nombre de ménages dans un bloc")
    st.write("- Revenu médian des ménages dans un bloc")
    st.write("- Valeur médiane des maisons pour les ménages d'un bloc")
    st.write("- Situation par rapport à la mer")

    st.write("#### Approche")
    st.write("Pour atteindre nos objectifs, nous suivrons une méthodologie rigoureuse, comprenant :")
    st.write("- Exploration approfondie des données pour comprendre leur nature et leur qualité")
    st.write("- Nettoyage des données pour traiter les doublons, les valeurs manquantes et les valeurs aberrantes")
    st.write("- Ingénierie des caractéristiques pour extraire des informations pertinentes")
    st.write("- Création de pipelines automatisés pour la préparation des données")
    st.write("- Développement d'un modèle KNN pour la prédiction des prix immobiliers")
    st.write("- Création d'une application web conviviale pour les prédictions en temps réel")

    st.write("#### Conclusion")
    st.write("Ce projet représente une étape cruciale dans notre mission d'offrir des solutions innovantes dans le domaine de l'investissement immobilier. Nous sommes impatients de partager nos résultats et de contribuer à simplifier le processus d'estimation des prix immobiliers pour nos clients.")

elif page == pages[1]:
    st.write("### Exploration des données")

    st.dataframe(logement.head())

    st.write("Dimensions du dataframe :")

    st.write(logement.shape)

    if st.checkbox("Afficher les valeurs manquantes") :
        st.dataframe(logement.isna().sum())

    if st.checkbox("Afficher les doublons") :
        st.write(logement.duplicated().sum())

elif page == pages[2]:
    st.write("### Analyse de données")

    plt.figure(figsize=(12, 6))
    sc = plt.scatter(logement["longitude"],
                 logement["latitude"],
                 alpha=0.4,
                 cmap="Reds",
                 c=logement["median_house_value"],
                 s=logement["population"]/50,
                 label='population')
    plt.colorbar(sc)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.title("Aperçu de la population en Californie")
    plt.legend()

    st.pyplot(plt)

    plt.figure(figsize=(15, 8))
    sns.scatterplot(x="latitude", y="longitude", data=logement, hue="median_house_value", palette="coolwarm")
    plt.title("Localisation de la valeur médiane des maisons")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.legend(title="Valeur médiane des maisons")
    plt.grid(True)

    st.pyplot(plt)

    fig = px.density_mapbox(logement,
                        lat='latitude',
                        lon='longitude',
                        z='median_house_value',
                        radius=10,
                        center=dict(lat=36.7783, lon=-119.4179),
                        zoom=4.5,
                        mapbox_style="open-street-map",
                        color_continuous_scale="Oranges",
                        title="Carte de densité des valeurs médianes des maisons en Californie")

    st.plotly_chart(fig)
    fig = px.density_mapbox(logement, lat='latitude', lon='longitude', z='population', radius=10,
                        center=dict(lat=36.7783, lon=-119.4179), zoom=4.5,
                        mapbox_style="open-street-map", color_continuous_scale="Burg", title="Carte de la densité de la population")
    st.plotly_chart(fig)

    fig = px.density_mapbox(logement, lat='latitude', lon='longitude', z='median_income', radius=10,
                        center=dict(lat=36.7783, lon=-119.4179), zoom=4.5,
                        mapbox_style="open-street-map", color_continuous_scale="Magenta", title="Carte du revenus median")
    st.plotly_chart(fig)
    fig = px.density_mapbox(logement, lat='latitude', lon='longitude', z='housing_median_age', radius=10,
                        center=dict(lat=36.7783, lon=-119.4179), zoom=4.5,
                        mapbox_style="open-street-map", color_continuous_scale="Brwnyl")
    st.plotly_chart(fig)
    fig = px.density_mapbox(logement, lat='latitude', lon='longitude', z='households', radius=10,
                        center=dict(lat=36.7783, lon=-119.4179), zoom=4.5,
                        mapbox_style="open-street-map", color_continuous_scale="Sunsetdark")
    st.plotly_chart(fig)
    fig = px.scatter_mapbox(logement, lat='latitude', lon='longitude', color='ocean_proximity',
                        center=dict(lat=36.7783, lon=-119.4179), zoom=4.5,
                        mapbox_style="open-street-map", color_continuous_scale="Purp")
    st.plotly_chart(fig)
    plt.figure(figsize=(20, 10))
    logement.hist(bins=25)
    plt.title("Histogramme des données du logement")
    plt.grid(False)
    st.pyplot(plt)

    income_bins = pd.cut(logement["median_income"],
                     bins=[0, 1.5, 3, 4.5, 6, float('inf')],
                     labels=["0 - 1.5", "1.5 - 3", "3 - 4.5", "4.5 - 6", " > 6 "])
    plt.figure(figsize=(10, 6))
    sns.countplot(x=income_bins)
    plt.title("Répartition des revenus médians")
    plt.xlabel("Revenu médian (en dizaines de milliers de dollars)")
    plt.ylabel("Nombre de districts")
    plt.xticks(rotation=45)

    st.pyplot(plt)
    house_value = pd.cut(x=logement["median_house_value"],
                     bins=(-np.inf, 100000, 200000, 300000, 400000, 500000, np.inf),
                     labels=('-inf to 100k', '100k to 200k', '200k to 300k', '300k to 400k', '400k to 500k', '500k to inf'))

    plt.figure(figsize=(15, 6))
    sns.countplot(x=house_value)
    plt.title('Catégories de valeur de la maison', fontsize=14)
    plt.xlabel('House Value Bins', fontsize=14)
    plt.ylabel('counts', fontsize=14)
    plt.xticks(rotation=45)

    st.pyplot(plt)

    plt.figure(figsize=(15, 10))
    sns.relplot(x="median_income", y='median_house_value', data=logement, col="ocean_proximity", col_wrap=3, color="#F27F55")
    plt.show()

    st.pyplot(plt)

    counts = logement['ocean_proximity'].value_counts()

    plt.figure(figsize=(10, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    plt.title('Répartition des logements par proximité de l\'océan')

    st.pyplot(plt)

elif page == pages[3]:
    st.write("### Modélisation")
