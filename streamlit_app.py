import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from pybaseball.plotting import plot_strike_zone
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder


cont_pitch_measures = [
        "release_speed", "release_pos_x", "release_pos_y", "release_pos_z","balls",
        "strikes","pfx_x","pfx_z","plate_x","plate_z","inning",
        "vx0","vy0","vz0","ax","ay","az","sz_top","sz_bot",
        "effective_speed","release_spin_rate","release_extension","pitch_number",
        "spin_axis","api_break_z_with_gravity","api_break_x_arm","api_break_x_batter_in",
        "arm_angle"
    ]

cat_pitch_measures = ["pitch_type","zone","stand","p_throws","type"]


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
    #df = pd.read_csv("data/restricted_data.csv")
    df = pd.read_parquet("data/restricted_data.parquet")
    def is_barrel(launch_speed, launch_angle):
        """Return 1 if the batted ball is classified as a barrel, else 0"""
        if launch_speed >= 98:
            if 26 <= launch_angle <= 30:
                return 1
            elif launch_speed >= 99 and 24 <= launch_angle <= 32:
                return 1
            elif launch_speed >= 100 and 22 <= launch_angle <= 34:
                return 1
        return 0

    # Create barrel column
    df['barreled'] = df.apply(lambda row: is_barrel(row['launch_speed'], row['launch_angle']), axis=1)
    
    return df

def encoder_scaler(df: pd.DataFrame):
    # One-hot encode categorical attributes
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(df[cat_pitch_measures])
    #encoded_categorical = encoder.fit_transform(df[cat_pitch_measures]).toarray()
    #encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(cat_pitch_measures))
    #encoded_df = pd.concat([df[['pitcher','batter']],encoded_df],axis=1)
    scaler = StandardScaler()
    scaler.fit(df[cont_pitch_measures])
    
    return encoder, scaler

df = load_data()
encoder, scaler = encoder_scaler(df)

# Show a single select widget for pitchers.
pitcher = st.sidebar.selectbox(
    "Pitcher",
    df.pitcher.unique(),
    placeholder="Select a pitcher to analyze...",
    help=None
)
batter = st.sidebar.multiselect(
    "Batter",
    df.batter.unique(),
    placeholder="Select a batter to analyze...",
    help=None
)


def compute_pitch_attributes(df: pd.DataFrame, encoder: OneHotEncoder, scaler: StandardScaler, ids: List, ptype='pitcher'):
    """
    In progress...
    """
    
    if ptype == 'batter':
        pindex = df[(df.batter.isin(ids)) & (df.barreled==1)].index
    else:
        #pindex = df[(df.pitcher.isin(ids)) & (df.barreled==1)].index
        pindex = df[(df.pitcher.isin(ids))].index

    # Aggregate pitch characteristics
    cont_mean = df.loc[pindex].groupby(ptype)[cont_pitch_measures].mean()
    cat_mode = df.loc[pindex].groupby(ptype)[cat_pitch_measures].apply(lambda x: x.mode().iloc[0])
    
    encoded = encoder.transform(df.loc[pindex][cat_pitch_measures]).toarray()
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(),index=pindex)
    encoded_mean_df = pd.concat([df.loc[pindex][[ptype]],encoded_df],axis=1).groupby(ptype).mean()
    
    scaled = scaler.transform(df.loc[pindex][cont_pitch_measures])
    scaled_df = pd.DataFrame(scaled, columns=scaler.feature_names_in_,index=pindex)
    scaled_mean_df = pd.concat([df.loc[pindex][[ptype]],scaled_df],axis=1).groupby(ptype).mean()

    raw_profiles = pd.concat([cat_mode,cont_mean],axis=1)
    stat_profiles = pd.concat([encoded_mean_df,scaled_mean_df],axis=1)
   
    return raw_profiles, stat_profiles

pitcher_att, pitcher_es = compute_pitch_attributes(df, encoder, scaler, [pitcher])
batter_att, batter_es = compute_pitch_attributes(df, encoder, scaler, batter,'batter')
cos_sim_score = cosine_similarity(pitcher_es, batter_es)#[0][0]
print(cos_sim_score)

#Display the data as a table using `st.dataframe`.
st.dataframe(
    pitcher_es,
    use_container_width=True,
    hide_index=True
)

chart = (
    alt.Chart(pitcher_es)
    .mark_line()
    .properties(height=320)
)

st.altair_chart(chart, use_container_width=True)

#Display the data as a table using `st.dataframe`.
st.dataframe(
    batter_es,
    use_container_width=True,
    hide_index=True
)

chart = (
    alt.Chart(batter_es)
    .mark_line()
    .properties(height=320)
)

st.altair_chart(chart, use_container_width=True)