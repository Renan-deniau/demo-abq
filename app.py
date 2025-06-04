import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# ---- CONFIGURATION ----
st.set_page_config(page_title="Sentiment StockTwits", layout="wide")

symbols = ["AAPL", "GOOGL", "AMZN", "MSFT", "STLA", "BTC"]

# ---- UI ----
st.title("📈 Dashboard de sentiment StockTwits")


# Charger les données CSV
@st.cache_data
def load_data():
    return pd.read_csv("export_symboles.csv", parse_dates=["created_at"])


df = load_data()


# Interface Streamlit
symboles = df["symbol"].unique()
selected_symbole = st.selectbox("Choisissez un symbole", symboles)

# Filtrer et forcer une copie propre
df_filtered = df[df["symbol"] == selected_symbole].copy()


# ---- AGRÉGATION POUR GRAPHIQUE ----
sentiment_counts = (
    df["sentiment"]
    .value_counts()
    .reindex(["negative", "neutral", "positive"])
    .fillna(0)
    .astype(int)
)
bar_data = pd.DataFrame(
    {"Sentiment": sentiment_counts.index, "Nombre de messages": sentiment_counts.values}
)

# ----------------------------------------GRAPHIQUE COURBES----------------------------------------

st.subheader("📊 Évolution quotidienne des messages par sentiment")

# Sélecteur de période
period_label = st.selectbox(
    "Période à afficher :",
    ["10 derniers jours", "2 derniers jours", "30 derniers jours"],
)

# Dictionnaire pour traduire le label en nombre de jours
period_map = {
    "10 derniers jours": 10,
    "2 derniers jours": 2,
    "30 derniers jours": 30,
}

period = period_map[period_label]

# --- Filtrage par symbole ---
# Convertir created_at en datetime
df_filtered["created_at"] = pd.to_datetime(df_filtered["created_at"], errors="coerce")

# Date de coupure (7 jours)
cutoff_date = datetime.now() - timedelta(days=period)

# Filtrage final
df_filtered = df_filtered[df_filtered["created_at"] >= cutoff_date]

# Agrégation
df_filtered["date"] = df_filtered["created_at"].dt.date
daily_counts = (
    df_filtered[df_filtered["sentiment"].isin(["positive", "negative"])]
    .groupby(["date", "sentiment"])
    .size()
    .reset_index(name="count")
)

# Pivot pour avoir une colonne par sentiment
pivot_df = daily_counts.pivot(index="date", columns="sentiment", values="count").fillna(
    0
)

# Création du graphique
fig_timeline = px.line(
    pivot_df,
    x=pivot_df.index,
    y=["positive", "negative"],
    labels={"value": "Nombre de messages", "date": "Date"},
    title="Messages positifs vs négatifs par jour",
    markers=True,
    color_discrete_map={
        "positive": "#42ff87",  # vert
        "negative": "#ff5182",  # rose/rouge
    },
)

# Personnalisation
fig_timeline.update_traces(mode="lines+markers")
fig_timeline.update_layout(xaxis_title="Date", yaxis_title="Nombre de messages")
st.plotly_chart(fig_timeline, use_container_width=True)


# ---- GAUGE-STYLE BAR ----

# Total messages
total = sentiment_counts.sum()
percentages = (sentiment_counts / total * 100).round(2)

# Préparer les données pour une jauge horizontale
gauge_df = pd.DataFrame(
    {
        "Sentiment": ["negative", "neutral", "positive"],
        "Pourcentage": percentages.values,
        "Couleur": [
            "#ff5182",
            "#262730" if st.get_option("theme.base") == "dark" else "#F0F2F6",
            "#42ff87",
        ],
    }
)

# Trace un bar chart 100% stacké horizontal
fig = px.bar(
    gauge_df,
    x="Pourcentage",
    y=[""] * 3,  # Une seule ligne horizontale
    color="Sentiment",
    orientation="h",
    color_discrete_map={
        "negative": "#ff5182",
        "neutral": "#262730" if st.get_option("theme.base") == "dark" else "#F0F2F6",
        "positive": "#42ff87",
    },
    text=gauge_df["Pourcentage"].astype(str) + "%",
)

# Personnaliser l'apparence
fig.update_layout(
    barmode="stack",
    showlegend=False,
    height=100,
    margin=dict(l=0, r=0, t=30, b=20),
    xaxis=dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False,
        range=[0, 100],
    ),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
)
fig.update_traces(textposition="inside", insidetextanchor="middle")

st.plotly_chart(fig, use_container_width=True)

# ---- LISTE DES MESSAGES PAR SENTIMENT (TRIABLE) ----
cols = st.columns(3)

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

        # Limiter à 20 messages
        filtered = filtered.head(20)

        # Affichage des messages triés
        for _, row in filtered.iterrows():
            st.markdown(
                f"**{row['created_at'].strftime('%Y-%m-%d %H:%M')}** — "
                f"Score: `{row['sentiment_score']:.2f}` — "
                f"@{row['username']} : {row['body']}"
            )
