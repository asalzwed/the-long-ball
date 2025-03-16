import pandas as pd
import streamlit as st
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import plotly.graph_objects as go

cont_pitch_measures = [
        "release_speed", "release_pos_x", "release_pos_y", "release_pos_z","balls",
        "strikes","pfx_x","pfx_z","plate_x","plate_z","inning",
        "vx0","vy0","vz0","ax","ay","az","sz_top","sz_bot",
        "effective_speed","release_spin_rate","release_extension","pitch_number",
        "spin_axis","api_break_z_with_gravity","api_break_x_arm","api_break_x_batter_in",
        "arm_angle"
    ]

cat_pitch_measures = ["pitch_type","zone","stand","p_throws","type"]

st.markdown("""
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 500px;
            font-size: 0.6rem;
        }

    </style>
    """, unsafe_allow_html=True)


st.sidebar.header("Filters")
st.title("⚾ The Long Ball")
st.write(
    """
    This app lets you examine pitcher-batter matchups with an emphasis on barreled balls. 
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

def compute_pitch_attributes(df: pd.DataFrame, encoder: OneHotEncoder, scaler: StandardScaler, ids: List, subset: List, ptype='pitcher_name_id'):
    """
    In progress...
    """
    if subset: 
        pindex = df[(df[ptype].isin(ids)) & (df[subset[0]]==subset[1])].index
    else:
        pindex = df[(df[ptype].isin(ids))].index
            

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
   
    return raw_profiles.round(2), stat_profiles.round(2)

def compute_metric(df: pd.DataFrame, field: str, event, delta: int, date_flag=0):
        total_events = len(df)
        num_events = len(df[df[field]==event])

        if date_flag==0:
            last_game = df['game_date'].max()
        else: 
            last_game = pd.Timestamp.today()   

        df_dX = df[(df['game_date']<=last_game) & (df['game_date']>=(last_game-pd.Timedelta(days=delta)))]

        dX_total_events = len(df_dX)
        dX_num_events = len(df_dX[df_dX[field]==event])

        return [[num_events,total_events],[dX_num_events,dX_total_events]], last_game

def create_gauge(player_metrics,average_metrics,factor=100,gsetting=4):
    dpv = (player_metrics[1][0]/player_metrics[1][1])*factor
    tpv = (player_metrics[0][0]/player_metrics[0][1])*factor
    apv = (average_metrics[0][0]/average_metrics[0][1])*factor
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value =  dpv,
        number_font_color="black",
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Barrel Rate", 'font': {'size': 12,'color':'black'}},
        delta = {'reference': tpv,'increasing': {'color': "green"},'decreasing': {'color':'red'}},
        gauge = {
            'axis': {'range': [None, gsetting], 'tickwidth': 1},
            'bar': {'color': "rgba(0,0,0,0)", "thickness": 1},
            'bgcolor': "white",
            'borderwidth': 2,
            'steps' : [{'range': [0, apv], 'color': "lightgray"},
                        {'range': [dpv, tpv], 'color': "red" if dpv<tpv else "green","thickness":0.5}],
            'threshold' : {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': dpv}
            }
            )
            )
    
    fig.update_layout(
        autosize=False,
        height=200,  # Adjust this value to control height
        width=200,
        margin=dict(l=15, r=15, t=0, b=0)  # Reduces extra spacing
    )
    return fig

def player_card(name, ban, m1, m2, factor=100, gsetting=4):
    """Reusable Player Card Component"""
    
    with st.container():
        
        #st.markdown(f"#### {name}")
        fig = create_gauge(m1, m2, factor, gsetting)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=name)
        
        st.metric("BSI", round(ban, 2))  # Stack name & metric

df_raw = load_data()

encoder, scaler = encoder_scaler(df_raw)
try: 
    
    pitcher = st.sidebar.multiselect(
        "Pitcher",
        df_raw.pitcher_name_id.unique(),
        placeholder="Select a pitcher to analyze...",
        default = 'Paul Skenes (694973)',
        max_selections=1,
        help=None
    )

    batter = st.sidebar.multiselect(
        "Batter",
        df_raw.batter_name_id.unique(),
        placeholder="Select a batter to analyze...",
        default=['Aaron Judge (592450)','Juan Soto (665742)'],
        help=None
    )

    date_range = st.sidebar.select_slider(
        "Select Date Range",
        options=df_raw["game_date"].sort_values().unique(),
        value=(df_raw["game_date"].min(), df_raw["game_date"].max())  # Default range
    )

    filtered_df = df_raw[(df_raw["game_date"] >= date_range[0]) & (df_raw["game_date"] <= date_range[1])]

    pitcher_att, pitcher_es = compute_pitch_attributes(filtered_df, encoder, scaler, pitcher, subset=[])
    batter_att, batter_es = compute_pitch_attributes(filtered_df, encoder, scaler, batter,['barreled',1],'batter_name_id')

    cos_sim_score = cosine_similarity(pitcher_es, batter_es)

    pmetrics, lgame  = compute_metric(filtered_df[filtered_df['pitcher_name_id']==pitcher[0]],'barreled',1,delta=30)
    ametrics, lgame = compute_metric(filtered_df,'barreled',1,delta=30)
    
    num_batters_per_row = 3  # Batters per row after the first row

    # Ensure there are at least 2 batters to complete the first row
    first_row_batters = batter[:2]  # First two batters

    # First row: 1 pitcher + 2 batters
    first_row_cols = st.columns(3)  # Ensure 3 columns

    # Pitcher goes in the first column
    with first_row_cols[0]:  
        with st.expander(pitcher[0], expanded=True):
            pometrics, lgame =  compute_metric(filtered_df[(filtered_df['pitcher_name_id']!=pitcher[0]) & (filtered_df['batter_name_id'].isin(batter))],'barreled',1,delta=30)
            player_card(pitcher[0], 0, pmetrics, pometrics)

    # Place the first two batters in the remaining spots
    for i, x in enumerate(first_row_batters):
        with first_row_cols[i + 1]:  # Start from index 1
            with st.expander(x, expanded=True):
                bmetrics, lgame = compute_metric(filtered_df[filtered_df['batter_name_id'] == x], 'barreled', 1, delta=30)
                bometrics, lgame =  compute_metric(filtered_df[(filtered_df['pitcher_name_id']==pitcher[0]) & (filtered_df['batter_name_id']!=x)],'barreled',1,delta=30)
                player_card(x, cos_sim_score[0][i], bmetrics, bometrics,factor=20,gsetting=2)

    # Remaining batters (starting from index 2)
    remaining_batters = batter[2:]  
    for i, x in enumerate(remaining_batters):
        # Create a new row every 3 batters
        if i % num_batters_per_row == 0:
            cols = st.columns(num_batters_per_row)

        with cols[i % num_batters_per_row]:  
            with st.expander(x, expanded=True):
                bmetrics, lgame = compute_metric(filtered_df[filtered_df['batter_name_id'] == x], 'barreled', 1, delta=30)
                bometrics, lgame =  compute_metric(filtered_df[(filtered_df['pitcher_name_id']==pitcher[0]) & (filtered_df['batter_name_id']!=x)],'barreled',1,delta=30)
                player_card(x, cos_sim_score[0][i + 2], bmetrics, bometrics,factor=20,gsetting=2)  # Offset index
    
    #ptype = pitcher_es.filter(like="pitch_type_")
    #ptype.columns = [col.replace("pitch_type_", "") for col in ptype.columns]
    #st.bar_chart(ptype[ptype>0].iloc[0,:])
    #st.bar_chart(ptype.iloc[0,:],horizontal=True,width=100, use_container_width=False)
   
except:
    st.write("Please select a pitcher and batter with data...")