import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Tuple
from pybaseball.plotting import plot_strike_zone

# Show the page title and description.
st.set_page_config(page_title="The Long Ball", page_icon="⚾",layout="wide",
        initial_sidebar_state="expanded")
st.sidebar.header("Filters")
st.title("⚾ The Long Ball")
st.write(
    """
    This app lets you examine pitcher-batter matchups with an emphasis on home run potential. 
    """
)


# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    df = pd.read_csv("data/test_data.csv")
    return df

def display_kpi_metrics(series: pd.Series):

    kpis = series.values
    kpi_names = series.index
    st.header("KPI Metrics")
    for i, (col, (kpi_name, kpi_value)) in enumerate(zip(st.columns(len(kpi_names)), zip(kpi_names, kpis))):
        col.metric(label=kpi_name, value=kpi_value)


@st.cache_data
def calculate_kpis(data: pd.DataFrame) -> List[float]:
    #Categorical Values
    kpis = data.agg({'pitch_name':lambda x: x.mode().iloc[0], 
              'zone':lambda x: x.mode().iloc[0],
              'release_speed':'mean'
    })


    return kpis

df = load_data()

# Show a single select widget for pitchers.
pitcher = st.sidebar.selectbox(
    "Pitcher",
    df.pitcher.unique(),
    placeholder="Select a pitcher to analyze...",
    help=None
)


# Filter the dataframe based on the widget input and reshape it.
#df_filtered = df[(df["pitcher"].isin(pitchers))]
pitcher_df = df[(df["pitcher"]==pitcher)]
#start_date = pd.Timestamp(st.sidebar.date_input("Start date", df_filtered['game_date'].min().date()))
#end_date = pd.Timestamp(st.sidebar.date_input("End date", df_filtered['game_date'].max().date()))


kpis = calculate_kpis(pitcher_df.loc[pitcher_df.events=='home_run'])

kpi_names = ["HRs","HRs/X","AVG RELEASE SPEED",'AVG RELEASE POS X']
#display_kpi_metrics(kpis, kpi_names)
display_kpi_metrics(kpis)
# fig, ax = plt.subplots()
# plot_strike_zone(df_filtered, title = "Outcome", colorby='release_speed', annotation="events",axis=ax)
# st.pyplot(fig)



#Display the data as a table using `st.dataframe`.
st.dataframe(
    pitcher_df,
    use_container_width=True,
    hide_index=True
)

chart = (
    alt.Chart(pitcher_df)
    .mark_line()
    .properties(height=320)
)

st.altair_chart(chart, use_container_width=True)