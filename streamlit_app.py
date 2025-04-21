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

cm = pd.read_parquet("data/current_matchup.parquet")

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
    scaler = StandardScaler()
    scaler.fit(df[cont_pitch_measures])
    
    return encoder, scaler

def compute_pitch_attributes(df: pd.DataFrame, encoder: OneHotEncoder, scaler: StandardScaler, ids: List, subset: List, ptype='pitcher'):
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
        
        fig = create_gauge(m1, m2, factor, gsetting)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=name)
        
        
            st.metric("BSI", round(ban, 2) if ban != 0 else " - ")

def create_metric(player_metrics,factor=100,gsetting=4):
    dpv = (player_metrics[1][0]/player_metrics[1][1])*factor
    tpv = (player_metrics[0][0]/player_metrics[0][1])*factor

    return [dpv,tpv]
df_raw = load_data()

encoder, scaler = encoder_scaler(df_raw)
try: 
    
    date_range = st.sidebar.select_slider(
        "Select Date Range",
        options=df_raw["game_date"].sort_values().unique(),
        value=(df_raw["game_date"].min(), df_raw["game_date"].max())  # Default range
    )

    filtered_df = df_raw[(df_raw["game_date"] >= date_range[0]) & (df_raw["game_date"] <= date_range[1])]
    pitchers = cm['pitcher_id'].unique()

    pitcher_bm = []
    for i, p in enumerate(pitchers):
        pmetrics, lgame = compute_metric(filtered_df[filtered_df['pitcher'] == p], 'barreled', 1, delta=30)
        test = create_metric(pmetrics)
        pitcher_bm.append({
            "Pitcher": filtered_df[filtered_df['pitcher'] == p]['pitcher_name_id'].unique()[0],
            "Pitcher ID": p,
            "Pitcher Barrel Delta": test[0],
            "Pitcher Barrel Total": test[1]
        })

    batters = cm['batter_id'].unique()
    batter_bm = []
    for i, b in enumerate(batters):
        try:
            bmetrics, lgame = compute_metric(filtered_df[filtered_df['batter'] == b], 'barreled', 1, delta=30)
            test = create_metric(bmetrics)
            batter_bm.append({
                "Batter": filtered_df[filtered_df['batter'] == b]['batter_name_id'].unique()[0],
                "Batter ID": b,
                "Pitcher ID": cm[cm.batter_id==b]['pitcher_id'].unique()[0],
                "Batter Barrel Delta": test[0],
                "Batter Barrel Total": test[1]
            })
        except:
            batter_bm.append({
                "Batter": "No Data",
                "Batter ID": b,
                "Pitcher ID": cm[cm.batter_id==b]['pitcher_id'].unique()[0],
                "Batter Barrel Delta": 0,
                "Batter Barrel Total": 0
            })
            
    # Convert to DataFrame
    b0 = pd.DataFrame(pitcher_bm)
    b1 = pd.DataFrame(batter_bm)
    bfinal = b0.merge(b1)
    bfinal['Combined Delta'] = bfinal['Pitcher Barrel Delta']+bfinal['Batter Barrel Delta']
    bfinal['Combined Total'] = bfinal['Pitcher Barrel Total']+bfinal['Batter Barrel Total']
    
    cos_sim_score = []
    for j,c in enumerate(bfinal['Pitcher ID']):
        
        b = bfinal.iloc[j]['Batter ID']
        p = bfinal.iloc[j]['Pitcher ID']
        
        try:
            batter_att, batter_es = compute_pitch_attributes(filtered_df, encoder, scaler, [b],['barreled',1],'batter')
            pitcher_att, pitcher_es = compute_pitch_attributes(filtered_df, encoder, scaler, [p], subset=[])
            c = cosine_similarity(pitcher_es, batter_es)
            cos_sim_score.append(c[0][0])
        except:
            cos_sim_score.append(0)

    bfinal['SIM_SCORE'] = cos_sim_score
    bfinal['X-Factor'] = (bfinal['Combined Delta'] * bfinal['Combined Total']* bfinal['SIM_SCORE']).round(2)
    st.dataframe(bfinal[['Pitcher','Batter','X-Factor']],use_container_width=True,hide_index=True)
except:
    st.write("Please select a pitcher and batter with data...")