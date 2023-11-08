import pickle
import streamlit as st
import pandas as pd

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Gujarat Titans',
    'Lucknow Super Giants'
    ]

cities = ['Mumbai', 'Chandigarh', 'Kolkata', 'Raipur', 'Delhi',
       'Visakhapatnam', 'Cape Town', 'Bangalore', 'Chennai', 'Abu Dhabi',
       'Pune', 'Hyderabad', 'Indore', 'Kimberley', 'Sharjah',
       'Johannesburg', 'Port Elizabeth', 'Jaipur', 'Dharamsala', 'Dubai',
       'Ahmedabad', 'Bloemfontein', 'Durban', 'Navi Mumbai', 'Centurion',
       'Bengaluru', 'Nagpur', 'Cuttack', 'Ranchi', 'East London']


pipe = pickle.load(open('ipl_predictor.pkl', 'rb'))
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select The Batting Team", sorted(teams))
with col2:
    bowling_team = st.selectbox("Select The Bowling Team", sorted(teams))

city = st.selectbox('Select Host City', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs Completed')
with col5:
    wickets = st.number_input('Wickets Out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets_left = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'BattingTeam':[batting_team],'BowlingTeam':[bowling_team],'City':[city],'runs_left':[runs_left],
                             'balls_left':[balls_left],'wickets_left':[wickets_left],'total_run_x':[target],'crr':[crr],'rrr':[rrr]})

    #st.table(input_df)
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.text(batting_team + "- " + str(win * 100) + "%")
    st.text(bowling_team + "- " + str(loss * 100) + "%")