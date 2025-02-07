import altair as alt
import pandas as pd
import streamlit as st
from typing import List, Tuple

# Show the page title and description.
st.set_page_config(page_title="The Long Ball", page_icon="🎬")
st.title("🎬 The Long Ball")
st.write(
    """
    This app lets you examine pitcher-batter matchups to determine the likelyhood of
    a homerun.
    """
)


# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    df = pd.read_csv("data/test_data.csv")
    return df

def display_kpi_metrics(kpis: List[float], kpi_names: List[str]):
    st.header("KPI Metrics")
    for i, (col, (kpi_name, kpi_value)) in enumerate(zip(st.columns(4), zip(kpi_names, kpis))):
        col.metric(label=kpi_name, value=kpi_value)

@st.cache_data
def calculate_kpis(data: pd.DataFrame) -> List[float]:
    avg_release_speed = data['release_speed'].mean().round(2)
    avg_release_pos_x = data['release_pos_x'].mean().round(2)
    return [avg_release_speed, avg_release_pos_x]


df = load_data()



# Show a multiselect widget with the genres using `st.multiselect`.
pitchers = st.multiselect(
    "Pitcher",
    df.pitcher.unique()
)

events = st.multiselect(
    "Events",
    df.events.unique()
)


# Filter the dataframe based on the widget input and reshape it.
df_filtered = df[(df["pitcher"].isin(pitchers)) & (df["events"].isin(events))]

kpis = calculate_kpis(df_filtered)
kpi_names = ["AVG RELEASE SPEED",'AVG RELEASE POS X']
display_kpi_metrics(kpis, kpi_names)


# Display the data as a table using `st.dataframe`.
st.dataframe(
    df_filtered,
    use_container_width=True,
    column_config={"year": st.column_config.TextColumn("Year")},
)

chart = (
    alt.Chart(df_filtered)
    .mark_line()
    .properties(height=320)
)
st.altair_chart(chart, use_container_width=True)