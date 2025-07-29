import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2
import os
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
# from datetime import datetime, timedelta

# ---- CONFIGURATION ----
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

config = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": [
        "zoom",
        "pan",
        "select",
        "lasso2d",
        "zoomIn",
        "zoomOut",
        "autoScale",
        "resetScale",
    ],
}
# st.plotly_chart(fig, config=config)


# ---- CONNEXION DATABASE ----
# @st.cache_resource
# def get_connection():
#     conn = psycopg2.connect(
#         dbname=os.getenv("POSTGRES_DB", "abq_db"),
#         user=os.getenv("POSTGRES_USER", "abq_user"),
#         password=os.getenv("POSTGRES_PASSWORD", "abq_pass"),
#         host=os.getenv("POSTGRES_HOST", "localhost"),
#         port=5432,  # utilisé dans le container
#     )
#     return conn


# conn = get_connection()


# ---- RÉCUPÉRATION DES SYMBOLS DISPONIBLES ----
# @st.cache_data
# def get_available_symbols():
#     with conn.cursor() as cur:
#         cur.execute("SELECT DISTINCT symbol FROM main_database ORDER BY symbol;")
#         rows = cur.fetchall()
#     return [row[0] for row in rows]


# symbols = get_available_symbols()
symbols = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA", "BTC"]

# def get_available_sources():
#     with conn.cursor() as cur:
#         cur.execute("SELECT DISTINCT source FROM main_database ORDER BY source;")
#         rows = cur.fetchall()
#     return [row[0] for row in rows]


# sources = get_available_sources()
sources = ["stocktwits", "yahoo", "all"]

# ---- UI ----
st.title("💬 Analyse de Sentiment des Marchés Financiers")
st.markdown(
    "**Démonstration d'une plateforme de data analytics appliquée aux marchés boursiers – scraping, NLP, Spark et Airflow**"
)


st.markdown("<div style='margin-top: 40px'></div>", unsafe_allow_html=True)
st.header("Introduction", divider="gray")

with st.container(border=False):
    left, middle, right = st.columns([4, 4, 4])
    with left:
        bulle = left.container(border=False)
        bulle_expander = bulle.expander(
            "Description du projet :", expanded=False, icon="✨"
        )
        bulle.image("images/zen_man.jpg")
        bulle_expander.markdown("""
        
        Ce projet vise à **analyser en temps réel le sentiment des investisseurs** à partir des forums boursiers comme *Stocktwits*, *Yahoo Finance*, et *Investing.com*.

        🎯 Ce projet fournit un outil de visualisation panoramique, dynamique et analytique de l'état des sentiments d'utilisateurs de forums financiers à propos d'un panel de cours d'actions en bourses.

        🚨 Approche de l'analyse qualitative vs quantitative :
        Face à la saturation des données chiffrées, l'analyse qualitative devient un levier différenciant pour mieux comprendre les dynamiques boursières. Ainsi les données qualitatives des forums financiers deviennent un indicateur stratégique : ce projet capte et analyse ces signaux pour favoriser la prédiction des tendances de marché.

        💡 Ce projet est un socle technique pouvant être décliné pour d'autres objectifs comme de l'analyse de satisfaction clients, de la veille concurentielle automatisée ou encore des analyses RH.

        **Résultat** : un tableau de bord interactif vous permet d'explorer le sentiment sur différentes actions boursières.
        """)

    with middle:
        bulle = middle.container(border=False)
        bulle_expander = bulle.expander("Stack technique :", expanded=False, icon="🛠️")
        bulle.image("images/stack.png")
        bulle_expander.markdown("""
        
        - ⚙️ **:grey-background[Scraping multi-sources]** : récupération automatisée de messages financiers publiés sur des forums et blogs spécialisé en ligne via **API et web scraping (Python requests & urllib)**.
        - 🧠 **:violet-background[Traitement NLP]** : analyse de sentiment en langage financier avec le modèle FinBERT avec **Hugging Face** et Python.
        - 🗓️ **:orange-background[Orchestration]** : gestion des workflows avec **Airflow** pour planification, exécution et monitoring des tâches (scraping, traitement, stockage).
        - 🐘 **:grey-background[Stockage et persistance]** : bases **PostgreSQL** pour stockage des données brutes et enrichies.
        - 🐳 **:blue-background[Infrastructure]** : conteneurisation avec **Docker**, gestion fine des ressources allouées et optimisées pour les performances et la montée en charge.
        - 🔀 **:violet-background[Big Data]** : calcul distribué avec **Spark (pyspark)**, traitement de large volumes de données.
        - 📊 **:green-background[Interface]** : affichage frontend dynamique avec **Streamlit**, visualisation interactive des données analysées via graphes.

        WIP : 

        - 🔁 **CI/CD** : intégration et déploiement continu avec Github Action
        - ☁️ **Cloud** : déploiement de l'application sur VPS cloud (Contabo)""")

    with right:
        bulle = right.container(border=False)
        bulle_expander = bulle.expander("Quel objectif ?", expanded=False, icon="🎯")
        bulle.image("images/target.jpg")
        bulle_expander.markdown("""
        
        Mon objectif est de présenter une démonstration technique de mes compétences en ingénierie des données.
        
        **Data Engineering** : Mettre en oeuvre une infrastructure compléte de traitement de données.
        
        **Data Science** : Mettre en oeuvre des algorithmes à la pointe de technologie et la science des données.
        
        **Data Analytics** : Mettre en oeuvre des outils de visualisation et d'analyse de données sur un cas concret.
        
        
        🚀 Je suis professionnellement à l'écoute pour tout projet de conception et déploiement de plateformes de data analytics, alors n'hésitez pas à me contacter si cela vous intéresse !
        
        📬 Contact : Renan DENIAU :
        - LinkedIn : https://www.linkedin.com/in/renan-deniau
        - Email : d.renan@outlook.fr
        - Téléphone : +33 6 46 88 05 43
        """)

with st.sidebar:
    message = st.markdown(
        """
    Bonjour ! 👋😊  
    
    Je m'appelle **Renan Deniau** et je vous invite à découvrir ce récent projet que j'ai développé de A à Z, de bout en bout.
    
    Bonne exploration du tableau de bord interactif !
    
    Si vous avez des questions, des remarques ou des suggestions, n'hésitez pas à me contacter. \
    
    
    
    📬 Contact :
    - Email : d.renan@outlook.fr
    - Téléphone : +33 6 46 88 05 43 
    """
    )
    button = st.link_button(
        "LinkedIn",
        "https://www.linkedin.com/in/renan-deniau",
        type="secondary",
        use_container_width=True,
    )

with st.expander("Technologies utilisées", expanded=False, icon="👉"):
    container = st.container(border=False)
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = container.columns(
        10, gap="large", vertical_alignment="center"
    )
    c1.image("images/Airflow.png")
    c2.image("images/ApacheSpark.png")
    c3.image("images/Docker.png")
    c4.image("images/FastAPI.png")
    c5.image("images/Contabo.png")
    c6.image("images/PostgreSQL.png")
    c7.image("images/Python.png")
    c8.image("images/Streamlit.png")
    c9.image("images/HuggingFace.png")
    c10.image("images/Pandas.png")


def get_neutral_color():
    # Détection du thème actif
    theme = st.get_option("theme.base")  # 'light' ou 'dark'
    # Couleurs personnalisées
    if theme == "dark":
        neutral_color = "#262730"  # gris foncé pour fond sombre
    else:
        neutral_color = "#F0F2F6"  # gris clair pour fond clair

    print(neutral_color)
    return neutral_color


# ---- RÉCUPÉRATION DES DONNÉES ----
sources_test = ["stocktwits", "yahoo", "all"]


# ,source  AND (source = %s OR %s = 'all')
# @st.cache_data
# def load_data(symbol, sources_condition_str):
#     query = """
#     SELECT sentiment, sentiment_score, username, body, created_at, source
#     FROM main_database
#     WHERE symbol = %s
#     """
#     if sources_condition_str:
#         query += " " + sources_condition_str
#     df = pd.read_sql_query(
#         query, conn, params=(symbol,)
#     )  # si un seul paramètre, mettre une virgule quand même pour tuple
#     return df


@st.cache_data
def load_data():
    return pd.read_csv("export.csv", parse_dates=["created_at"])


# , source, source

# Pour extraire des données vers un csv
# def load_data(symbol):
#     query = """
#     SELECT id, id_stocktwits, symbol, sentiment, sentiment_score, username, body, created_at, scraped_at
#     FROM stocktwits_messages
#     WHERE symbol = 'AAPL' OR symbol = 'GOOGL' OR symbol = 'AMZN' OR symbol = 'MSFT' OR symbol = 'STLA' OR symbol = 'BTC'
#     """
#     df = pd.read_sql_query(query, conn, params=(symbol,))
#     return df
st.markdown("<div style='margin-top: 80px'></div>", unsafe_allow_html=True)
st.header("Dashboard", divider="gray")

left, right = st.columns([6, 2])

with left:
    main_container = st.container(border=True)
    # space, content = main_container.columns([1, 50])

    left_container = main_container.container(border=False)
    left_container.subheader("📊 Évolution quotidienne du sentiment")

    col1, col2, col3, col4, col5, col6 = left_container.columns([1, 1, 1, 1, 1, 1])

    with col1:
        selected_symbol = st.selectbox("Choisir un symbole :", symbols, index=0)

    with col2:
        period_label = st.selectbox(
            "Période à afficher :",
            ["2 derniers jours", "10 derniers jours", "30 derniers jours"],
            index=2,
        )

    # with col3:
    #     days = st.slider(
    #         "🕒 Période analysée (en jours)", min_value=1, max_value=30, value=7
    #     )

    with col6:
        # st.markdown(
        #     unsafe_allow_html=True, body="<font size='2'>Sélection de la source :</font>"
        # )
        source_expander = st.expander("Sources", expanded=False, icon="🌐")
        all_sources = source_expander.checkbox("Tous", value=True, key="all")
        if all_sources:
            stocktwits = source_expander.checkbox(
                "Stocktwits", value=False, key="stocktwits", disabled=True
            )
            yahoo = source_expander.checkbox(
                "Yahoo Finance", value=False, key="yahoo", disabled=True
            )
        else:
            stocktwits = source_expander.checkbox(
                "Stocktwits", value=True, key="stocktwits", disabled=False
            )
            yahoo = source_expander.checkbox(
                "Yahoo Finance", value=True, key="yahoo", disabled=False
            )
        if not (all_sources or stocktwits or yahoo):
            st.warning("Veuillez choisir au moins une source.")

        # Construire la condition de la requête SQL
        sources_condition = []
        if all_sources:
            sources_condition_str = ""  # condition toujours vraie
        else:
            flag = False
            if stocktwits:
                sources_condition.append("stocktwits")
                flag = True
            if yahoo:
                sources_condition.append("yahoo")
                flag = True

            if flag:
                sources_condition_str = "AND source IN " + str(
                    sources_condition
                ).replace("[", "(").replace("]", ")").replace('"', "'")
            else:
                sources_condition_str = ""

    df = load_data()
    # df.to_csv("export_symboles.csv", index=False)

    # with st.expander("📄 Afficher les données brutes "):
    #     st.dataframe(df)
    # ----------------------------------------GRAPHIQUE COURBES----------------------------------------

    # Dictionnaire pour traduire le label en nombre de jours
    period_map = {
        "10 derniers jours": 10,
        "2 derniers jours": 2,
        "30 derniers jours": 30,
    }

    period = period_map[period_label]

    # Calcul des bornes temporelles
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=period)

    # Filtrage

    df_filtered = df[df["created_at"] >= cutoff_date]

    # Agrégation
    df_filtered["date"] = df_filtered["created_at"].dt.date
    daily_counts = (
        df_filtered[df_filtered["sentiment"].isin(["positive", "negative"])]
        .groupby(["date", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    # Pivot pour avoir une colonne par sentiment
    pivot_df = daily_counts.pivot(
        index="date", columns="sentiment", values="count"
    ).fillna(0)

    # Création du graphique
    fig_timeline = px.line(
        pivot_df,
        x=pivot_df.index,
        y=["positive", "negative"],
        labels={"value": "Nombre de messages", "date": "Date"},
        title="Messages <span style='color:green'>positifs</span> vs <span style='color:red'>négatifs</span> publiés par jour",
        markers=True,
        color_discrete_map={
            "positive": "#42ff87",  # vert
            "negative": "#ff5182",  # rose/rouge
        },
    )

    # Personnalisation
    fig_timeline.update_traces(mode="lines+markers")
    fig_timeline.update_layout(
        xaxis_title="Date", yaxis_title="Nombre de messages", height=492
    )

    left_container.plotly_chart(fig_timeline, use_container_width=True)

with right:
    right_container = st.container(border=True)

    right_container.markdown("Évolution journalière moyenne des messages par catégorie")

    # -----------------------------------------------------------------------------------------------------------------------------------------
    # df_filtered["heure"] = df_filtered["created_at"].dt.hour
    # df_filtered["jour"] = df_filtered["created_at"].dt.day
    df_filtered["jour"] = df_filtered["created_at"].apply(lambda x: x.toordinal())
    # df : DataFrame avec colonnes ['date', 'sentiment', 'count']

    # df_bi = df_filtered[
    #     df_filtered["sentiment"].isin(["positive", "negative", "neutral"])
    # ]
    daily_df = (
        df_filtered.groupby(["jour", "sentiment"])["body"]
        .count()
        .rename("msg_count")
        .reset_index()
    )  # with st.expander("📄 Afficher les données brutes daily_df"):
    #     st.dataframe(daily_df)

    daily_ratio = (
        daily_df.groupby(["jour"])["msg_count"]
        .apply(
            lambda x: x[daily_df["sentiment"] == "positive"].sum()
            / (
                x[daily_df["sentiment"] == "negative"].sum()
                + x[daily_df["sentiment"] == "positive"].sum()
            )
        )
        .reset_index()
    )

    def tendance_stats_v2(df_daily, sentiment, verbose):
        df_daily = df_daily.dropna()  # Supprimer les lignes avec des valeurs NaN
        if len(df_daily) > 1:
            X = np.arange(len(df_daily)).reshape(-1, 1)
            y = df_daily["msg_count"].values.reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0][0]
            mean = np.mean(df_daily["msg_count"])
            variation = (slope / mean) * 100
            if verbose:
                if slope > 0:
                    st.success(
                        f"📊 Tendance : Hausse quotidienne de **{variation:.2f}%**, soit **+{slope:.2f}** messages {sentiment} publiés par jour."
                    )
                else:
                    st.warning(
                        f"📊 Tendance : Baisse quotidenne de **{variation:.2f}%**, soit **{slope:.2f}** messages {sentiment} publiés par jour."
                    )
        else:
            slope, mean, variation = None, None, None
            st.markdown("Pas assez de données")
        return (slope, mean, variation)

    # Tendance quotidienne
    df_pos = daily_df[daily_df["sentiment"] == "positive"]
    df_neg = daily_df[daily_df["sentiment"] == "negative"]
    df_all = daily_df.groupby("jour")["msg_count"].sum().reset_index()
    df_ratio = daily_ratio

    tendance_pos = tendance_stats_v2(df_pos, "positive", False)
    tendance_neg = tendance_stats_v2(df_neg, "negative", False)
    # tendance_neutral = tendance_stats(df_filtered, "neutral", False)
    tendance_all = tendance_stats_v2(df_all, "all", False)
    tendance_ratio = tendance_stats_v2(df_ratio, "ratio", False)

    # right.metric(
    #     "Ratio msg +/- publiés /jour",
    #     f"{tendance_ratio[1] * 100:.1f}%",
    #     f"{tendance_ratio[0] * 100:.1f}%",
    #     border=True,
    # )

    ratio_value = tendance_ratio[1] * 100

    # Créer le gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=ratio_value,
            number={
                "font_color": "#ff5182"
                if ratio_value < 20
                else "#ffa5be"
                if ratio_value < 40
                else "#31333F"
                if ratio_value < 60
                else "#adffcb"
                if ratio_value < 80
                else "#42ff87"
            },
            title={
                "text": "Indicateur du sentiment",  # Moyenne du ratio msg +/- publiés /jour
                "font_color": "#31333F",
            },
            delta={
                "reference": (tendance_ratio[1] - tendance_ratio[0]) * 100,
                "increasing": {"color": "#31333F"},
            },
            gauge={
                "axis": {"range": [0, 100], "visible": True},
                "bar": {"color": "#31333F", "thickness": 0.05},
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, 20],
                        "color": "#ff5182",
                    },
                    {"range": [20, 40], "color": "#ffa5be"},
                    {"range": [40, 60], "color": "#F0F2F6"},
                    {"range": [60, 80], "color": "#adffcb"},
                    {"range": [80, 100], "color": "#42ff87"},
                ],
                "threshold": {
                    "value": ratio_value,
                    "thickness": 0.6,
                    "line": {"color": "#31333F", "width": 4},
                },
            },
        ),
    )
    fig.update_layout(height=350)

    # Afficher le gauge chart avec Streamlit
    with right_container.container(border=True):
        st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = right_container.columns(3)
    col1.metric(
        "Messages positifs",
        f"{tendance_pos[1]:.1f}",
        f"{tendance_pos[0]:.1f}",
        border=True,
    )

    col2.metric(
        "Messages négatifs",
        f"{tendance_neg[1]:.1f}",
        f"{tendance_neg[0]:.1f}",
        delta_color="inverse",
        border=True,
    )
    col3.metric(
        "Total messages",
        f"{tendance_all[1]:.1f}",
        f"{tendance_all[0]:.1f}",
        border=True,
    )

    with right_container.expander("Quelle analyse ?", expanded=False, icon="🔎"):
        st.markdown(
            """
            Les graphiques ci-dessus permettent de visualiser les tendances du sentiment des utilisateurs. 
            Chaque indicateur s'appuie sur une moyenne quotidienne. La variation adjacente correspond à la variation moyenne quotidienne sur la période sélectionnée.
            """
        )
        st.markdown("Exemple :")
        if tendance_pos[0] > 0:
            st.success(
                f"📊 Tendance messages positifs : Moyenne de **{tendance_pos[1]:.2f}** messages positifs publiés par jour avec une hausse quotidienne de **+{tendance_pos[0]:.2f}** messages positifs publiés par jour."
            )
        else:
            st.warning(
                f"📊 Tendance messages positifs : Moyenne de **{tendance_pos[1]:.2f}** messages positifs publiés par jour avec une baisse quotidenne de **{tendance_pos[0]:.2f}** messages positifs publiés par jour."
            )


# ---- AGRÉGATION POUR GRAPHIQUE ----
sentiment_counts = (
    df_filtered["sentiment"]
    .value_counts()
    .reindex(["negative", "neutral", "positive"])
    .fillna(0)
    .astype(int)
)
bar_data = pd.DataFrame(
    {"Sentiment": sentiment_counts.index, "Nombre de messages": sentiment_counts.values}
)
# ---- GAUGE-STYLE BAR ----

# # Total messages
# total = sentiment_counts.sum()
# percentages = (sentiment_counts / total * 100).round(2)

# # Préparer les données pour une jauge horizontale
# gauge_df = pd.DataFrame(
#     {
#         "Sentiment": ["negative", "neutral", "positive"],
#         "Pourcentage": percentages.values,
#         "Couleur": [
#             "#ff5182",
#             "#262730" if st.get_option("theme.base") == "dark" else "#F0F2F6",
#             "#42ff87",
#         ],
#     }
# )

# # Trace un bar chart 100% stacké horizontal
# fig = px.bar(
#     gauge_df,
#     x="Pourcentage",
#     y=[""] * 3,  # Une seule ligne horizontale
#     color="Sentiment",
#     title="Jauge du total de messages positifs, neutres et négatifs sur la période sélectionnée",
#     orientation="h",
#     color_discrete_map={
#         "negative": "#ff5182",
#         "neutral": "#262730" if st.get_option("theme.base") == "dark" else "#F0F2F6",
#         "positive": "#42ff87",
#     },
#     text=gauge_df["Pourcentage"].astype(str) + "%",
# )

# # Personnaliser l'apparence
# fig.update_layout(
#     barmode="stack",
#     showlegend=False,
#     height=100,
#     margin=dict(l=0, r=0, t=30, b=20),
#     xaxis=dict(
#         showticklabels=False,
#         showgrid=False,
#         zeroline=False,
#         visible=False,
#         range=[0, 100],
#     ),
#     yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
# )
# fig.update_traces(textposition="inside", insidetextanchor="middle")

# st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------


with st.expander("Statistiques supplémentaires", expanded=False, icon="👉"):
    left, middle, right = st.columns([2.5, 10, 4], border=True)
    left.markdown("Nombre de messages par source")

    # Agrégation
    df_source_counts = df_filtered.groupby("source").size().reset_index(name="count")

    # Affichage du bar chart
    # couleurs = ["#00AA76", "#9D61FF", "#008FFE"]
    # left.bar_chart(
    #     df_source_counts,
    #     x="source",
    #     y="count",
    #     y_label="Nombre de messages",
    #     x_label="",
    #     color="source",
    #     width=200,
    #     height=483,
    #     use_container_width=True,
    # )

    fig = px.bar(df_source_counts, x="source", y="count", color="source", height=483)

    # Masquer les labels de l'axe des abscisses
    fig.update_layout(
        xaxis_title="Sources",
        xaxis=dict(
            showticklabels=False  # ⛔️ masque les textes des ticks (abscisse)
        ),
        yaxis_title="Nombre de messages",
        yaxis=dict(
            tickformat=".2s"  # ⬅️ format compact : 4500 devient 4.5k
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            title_text="",
            orientation="h",  # Horizontal
            yanchor="bottom",
            y=1.05,  # Position verticale (négatif = en dessous du graphe)
            xanchor="center",
            x=0.5,  # Centré horizontalement
        ),
    )

    # Affichage dans Streamlit (à gauche si dans une colonne)
    left.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)

    # lefty = left.expander("Quelle analyse ?", expanded=False, icon="🔎")
    # lefty.markdown(
    #     """
    #     Identification des plateformes populaires.
    #     """
    # )

    # # Extraire l'heure de la journée à partir de la colonne 'created_at'
    # df_filtered["heure"] = df_filtered["created_at"].dt.hour

    # # Créer un histogramme du nombre de messages envoyés selon les plages horaires de la journée
    # fig, ax = plt.subplots()
    # ax.hist(df_filtered["heure"], bins=24)
    # ax.set_xlabel("Heure de la journée")
    # ax.set_ylabel("Nombre de messages envoyés")
    # ax.set_title("Activité des utilisateurs selon l'heure de la journée")

    # # Afficher le graphique avec Streamlit
    # middle.pyplot(fig)

    middle.markdown("Heures de publications des messages")
    mid_toggle = middle.toggle(
        "Afficher la répartition négatifs/neutres/positifs", value=False
    )
    # Extraire l'heure de la journée à partir de la colonne 'created_at'
    df_filtered["heure"] = df_filtered["created_at"].dt.hour

    if not (mid_toggle):
        # with st.expander("📄 Afficher les données brutes"):
        #     st.dataframe(df_filtered)

        # Créer un histogramme du nombre de messages envoyés selon les plages horaires de la journée

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df_filtered["heure"].value_counts().index,
                    y=df_filtered["heure"].value_counts(),
                    text=df_filtered["heure"].value_counts().apply(lambda x: f"{x}"),
                    textposition="auto",
                    marker=dict(
                        color=df_filtered["heure"].value_counts().values,
                        colorscale="Magenta",  # Agsunset, Purp, Dense, Peach
                        reversescale=True,
                        showscale=False,
                    ),
                )
            ]
        )
        heure_labels = [
            "Minuit",
            "2h",
            "4h",
            "6h",
            "8h",
            "10h",
            "12h",
            "14h",
            "16h",
            "18h",
            "20h",
            "22h",
        ]
        heure_values = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

        fig.update_layout(
            xaxis_title="Heure de la journée",
            xaxis=dict(ticktext=heure_labels, tickvals=heure_values),
            yaxis_title="Nombre de messages envoyés",
        )

        # Ajouter un encadré avec un texte au dessus de la plus haute barre
        max_value = df_filtered["heure"].value_counts().max()
        max_index = df_filtered["heure"].value_counts().idxmax()
        fig.add_annotation(
            x=max_index,
            y=max_value
            + 0.1
            * max_value,  # décalage pour que le texte ne soit pas trop proche de la barre
            text=f"Pic horaire d'activité : {max_index}h",
            showarrow=False,
            font=dict(size=12),
            borderwidth=1,
            bgcolor="#F0F2F6",
        )
    else:
        # Définir les couleurs pour les messages négatifs, neutres et positifs
        colors = ["#ff5182", "#F0F2F6", "#42ff87"]

        # Créer un tableau de données pour chaque type de message
        negative_counts = df_filtered[df_filtered["sentiment"] == "negative"][
            "heure"
        ].value_counts()
        neutral_counts = df_filtered[df_filtered["sentiment"] == "neutral"][
            "heure"
        ].value_counts()
        positive_counts = df_filtered[df_filtered["sentiment"] == "positive"][
            "heure"
        ].value_counts()

        # Créer un stacked bar plot
        fig = go.Figure(
            data=[
                go.Bar(
                    x=negative_counts.index,
                    y=negative_counts.values,
                    name="Négatifs",
                    marker=dict(color=colors[0]),
                ),
                go.Bar(
                    x=neutral_counts.index,
                    y=neutral_counts.values,
                    name="Neutres",
                    marker=dict(color=colors[1]),
                ),
                go.Bar(
                    x=positive_counts.index,
                    y=positive_counts.values,
                    name="Positifs",
                    marker=dict(color=colors[2]),
                ),
            ]
        )

        heure_labels = [
            "Minuit",
            "2h",
            "4h",
            "6h",
            "8h",
            "10h",
            "12h",
            "14h",
            "16h",
            "18h",
            "20h",
            "22h",
        ]
        heure_values = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
        # Définir la propriété barmode à "stack"
        fig.update_layout(
            barmode="stack",
            xaxis_title="Heure de la journée",
            xaxis=dict(ticktext=heure_labels, tickvals=heure_values),
            yaxis_title="Nombre de messages envoyés",
            legend=dict(
                orientation="h", x=1, y=1.02, xanchor="right", yanchor="bottom"
            ),
        )

        # Afficher le graphique
        # right.plotly_chart(fig)

    # Afficher le graphique avec Streamlit
    middle.plotly_chart(fig)
    # with middle.container("Quelle analyse ?", expanded=False, icon="🔎"):
    #     st.markdown(
    #         """
    #         Ces données permettent d'identifier les créneaux horaires où les utilisateurs sont les plus actifs.
    #         Cela peut notamment permettre d'identifier la meilleur période pour solliciter les utilisateurs avec une publicité par exemple.
    #         Observation gloable : les utilisateurs sont plus actifs l'aprés-midi et le soir que le matin, avec des créneaux favoris autour de 14h et de 19h.
    #         D'ailleurs, les utilisateurs de Yahoo Finance ont tendance à favoriser le créneau de 19h plutôt que 14h contrairement aux utilisateurs de StockTwits.
    #         Cela peut démontrer la différence de profil et d'utilisation de ces plateformes, stocktwits ayant une audience plus jeune et plus active que yahoo finance.
    #         """
    #     )

    right.markdown("De nouveaux indicateurs arriveront bientôt ...")
# -----------------------------------------------------WORDCLOUD------------------------------------------------------------------

st.markdown("<div style='margin-top: 40px'></div>", unsafe_allow_html=True)

with st.expander("Nuages de mots", expanded=False, icon="💭"):
    st.markdown(
        "Représente les mots les plus fréquents dans les messages :red-background[négatifs], :gray-background[neutres] et :green-background[positifs] "
    )
    left, middle, right = st.columns(3, border=True)

    text = " ".join(df[df.sentiment == "negative"]["body"].dropna().astype(str))
    wordcloud = WordCloud(
        background_color="white", colormap="OrRd", width=800, height=400
    ).generate(text)
    left.image(wordcloud.to_array())

    text = " ".join(df[df.sentiment == "neutral"]["body"].dropna().astype(str))
    wordcloud = WordCloud(
        background_color="white", colormap="Greys_r", width=800, height=400
    ).generate(text)
    middle.image(wordcloud.to_array())

    text = " ".join(df[df.sentiment == "positive"]["body"].dropna().astype(str))
    wordcloud = WordCloud(
        background_color="white", colormap="Greens_r", width=800, height=400
    ).generate(text)
    right.image(wordcloud.to_array())

with st.expander("Quelle analyse ?", expanded=False, icon="🔎"):
    st.markdown(
        """
        Connaître les mots les plus fréquents dans les messages négatifs, neutres et positifs permet de mieux comprendre les opinions des utilisateurs et d'identifier les mots les plus influents sur leur sentiment.
        Cela peut se révéler trés important pour déterminer les mots clés et les conseils appropriés pour améliorer la satisfaction des utilisateurs ou la conception d'une publicité.
        """
    )
# ---- LISTE DES MESSAGES PAR SENTIMENT (TRIABLE) ----

message_container = st.container(border=True)
cols = message_container.columns(3)


for i, sentiment in enumerate(["negative", "neutral", "positive"]):
    with cols[i]:
        st.subheader(f"Messages {sentiment.capitalize()}")

        # Filtrer les messages par sentiment
        filtered = df[df["sentiment"] == sentiment]

        # Choix de tri local à chaque sentiment
        sort_option = st.selectbox(
            "Trier par :",
            ["Date (plus récent d'abord)", "Score (plus élevé d'abord)"],
            key=f"{sentiment}_sort_option",
        )

        if sort_option == "Date (plus récent d'abord)":
            filtered = filtered.sort_values(by="created_at", ascending=False)
        else:
            filtered = filtered.sort_values(by="sentiment_score", ascending=False)

        limite = st.slider(
            "Nombre de messages à afficher",
            min_value=0,
            max_value=len(filtered),
            step=10,
            value=15,
        )

        filtered = filtered.head(limite)
        # filtered = filtered.head(30)

        # Affichage des messages triés
        for _, row in filtered.iterrows():
            st.markdown(
                f"**{row['created_at'].strftime('%Y-%m-%d %H:%M')}** — "
                f"Score: `{row['sentiment_score']:.2f}` — "
                f"@{row['username']} : {row['body']}"
            )
