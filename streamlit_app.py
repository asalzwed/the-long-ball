import altair as alt
import pandas as pd
import streamlit as st

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