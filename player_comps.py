import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


pd.set_option('display.max_columns', None)


imputer = KNNImputer(n_neighbors=3)
# Load the dataset
# data = pd.read_csv('/Users/nick/sleepertoolsversion2/combine_data/combine_and_fantasy.csv')
# print(data.info(verbose=True))
#
# seasons = np.arange(2022,2024).tolist()
# df = nfl.import_combine_data(seasons, ['QB','WR','RB','TE'])
# print(df.sort_values(by='season', ascending=False).head())
#
# df["first_name"] = df.player_name.apply(lambda x: x.split(" ")[0])
# df["last_name"] = df.player_name.apply(lambda x: " ".join(x.split(" ")[1::]))
#
#
# # Remove non-alpha numeric characters from first/last names.
# df["first_name"] = df.first_name.apply(
#     lambda x: "".join(c for c in x if c.isalnum())
# )
# df["last_name"] = df.last_name.apply(
#     lambda x: "".join(c for c in x if c.isalnum())
# )
#
# # Recreate full_name to fit format "Firstname Lastname" with no accents
# df["merge_name"] = df.apply(
#     lambda x: x.first_name + " " + x.last_name, axis=1
# )
# df["merge_name"] = df.merge_name.apply(lambda x: x.lower())
# df.drop(["first_name", "last_name"], axis=1, inplace=True)
#
# df["merge_name"] = df.merge_name.apply(lambda x: unidecode.unidecode(x))
#
# # Create Column to match with RotoGrinders
# df["merge_name"] = df.merge_name.apply(
#     lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:6])
df = pd.read_csv('https://raw.githubusercontent.com/stranger9977/player_comps_tool/master/apicombine.csv')
df = df[(df['season'] >= 2012) & (df['season'] <= 2024)]
df = df[(df['draftGrade'] >= 70) | (df['nflComparison'].notnull())]

df = df[['player_name','headshot','season', 'pos', 'ht', 'wt', 'forty','vertical','broad_jump','cone','shuttle','headshot', 'draftGrade','productionScore','nflComparison', 'bio','school']]
# df = df.merge(api_df, on = 'merge_name')
# df = df[['player_name','headshot','season', 'pos', 'ht', 'wt', 'forty','vertical','broad_jump','cone','shuttle', 'draftGrade','productionScore']]
print(df['headshot'].head())

# df['headshot_url'] = df['headshot'].apply(lambda url: url.replace('{formatInstructions}', 'f_auto,q_85'))

# Find the row for the player you want to fill in the height and weight values for
player_row = df[df['player_name'] == 'Elijah Moore']

# Fill in the height and weight values using loc
df.loc[player_row.index, 'ht'] = '5-10'
df.loc[player_row.index, 'wt'] = 184
# df.dropna(subset=['ht','wt'],inplace=True)
# def convert_height_to_cm(height):
#
#     # Split the height string into feet and inches
#     feet, inches = height.split('-')
#
#     # Convert feet and inches to centimeters
#     total_inches = round(int(feet) * 12 + int(inches),2)
#     height_in_cm = round(total_inches * 2.54, 2 )
#
#     return height_in_cm
#
#
# df['ht'] = df['ht'].apply(lambda x: convert_height_to_cm(x))
st.title("Combine Data Player Comps")

# Get the available years from the dataset
years = sorted(df['season'].unique())

# Create a dropdown menu for selecting a year
selected_year = st.selectbox("Select a Year", years)

# Filter the dataset based on the selected year
df_year = df[df['season'] == selected_year]

positions = years = sorted(df['pos'].unique())

selected_position = st.selectbox('Select A Position', positions)

df_position = df_year[df_year['pos']== selected_position]
# Get the available players from the filtered dataset
players = sorted(df_position['player_name'].unique())

# Create a dropdown menu for selecting a player
selected_player = st.selectbox("Select a player", players)

# Get the position for the selected player
selected_position = df_year[df_year['player_name'] == selected_player]['pos'].iloc[0]

# Create a dropdown menu for selecting the number of neighbors (n)
n_options = list(range(1, 11))

selected_n = st.selectbox("Select the number of neighbors", n_options)

# Create a button to run the knn_neighbors function

def knn_neighbors(player_name, pos, n):
    # Filter the DataFrame by pos
    pos_df = df[df['pos'] == pos]
    # Find the player's season
    player_season = pos_df.loc[pos_df['player_name'] == player_name, 'season'].values[0]

    # Drop all other players with the same season as the input player
    pos_df = pos_df.loc[~((pos_df['player_name'] != player_name) & (pos_df['season'] == player_season))]
    # Drop columns with null values for the input player
    player_df = pos_df[pos_df['player_name'] == player_name]
    player_df.info(verbose=True)
    null_cols = player_df.isnull().any()
    null_cols = null_cols[null_cols].index.tolist()
    features = ['ht', 'wt', 'forty', 'vertical', 'broad_jump','cone','shuttle','draftGrade','productionScore']
    features = [f for f in features if f not in null_cols]
    pos_df = pos_df.dropna(subset=features)
    # Rank the feature values relative to all players at the position
    ranks_df = pos_df[['player_name'] + features]
    ranks_df.set_index('player_name', inplace=True)
    ranks_df = ranks_df[features].rank(pct=True, na_option='keep')
    invert_cols = ['forty','cone','shuttle']
    invert_cols = [f for f in invert_cols if f not in null_cols]
    invert_ranks_df = ranks_df[invert_cols]
    ranks_df[invert_cols] = 1 - invert_ranks_df

    # Scale the feature values using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(ranks_df)

    # Fit a KNN model on the original feature values
    knn = NearestNeighbors(n_neighbors=n)
    knn.fit(pos_df[features])

    # Find the n neighbors of the input player
    player_features = player_df[features]
    distances, indices = knn.kneighbors(player_features)

    # Create a DataFrame of the neighbors and their similarity scores
    neighbors_df = pos_df.iloc[indices[0]].reset_index(drop=True)
    neighbors_df['similarity_score'] = 1 / (1 + distances[0])

    # Create a DataFrame for the input player and concatenate it with the neighbors DataFrame
    input_player_df = player_df.reindex(neighbors_df.columns).dropna()
    input_player_df['similarity_score'] = 1.0
    input_player_df.dropna(inplace=True)
    output_df = pd.concat([input_player_df, neighbors_df])

    # Drop columns with null values for the input player from the output DataFrame
    output_df.drop(null_cols, axis=1, inplace=True)

    # Convert the percentile ranks to 0-100 scale

    ranks_df.columns = [f'{col}_percentile' for col in ranks_df.columns]
    print(ranks_df.sort_values(by='wt_percentile', ascending=False))
     # Concatenate the DataFrames side by side
    # Concatenate the DataFrames side by side
    output_df = output_df.merge(ranks_df, right_index=True, left_on='player_name', how='left')

    print(output_df)
    # Define the variables that will be plotted
    variables = [var for var in features if var not in null_cols]
    plot_variables = [col for col in output_df.columns if col.endswith('_percentile')]

    # Define the data for each player
    player_names = output_df['player_name'].tolist()
    player_data = []
    for i, row in output_df.iterrows():
        player_data.append([row[col] for col in plot_variables])

    # Create the spider chart
    angles = np.linspace(0, 2 * np.pi, len(plot_variables), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # close the polygon
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    for i, (name, data) in enumerate(zip(player_names, player_data)):
        data = np.concatenate((data, [data[0]]))  # close the polygon
        ax.plot(angles, data, label=name)
        ax.fill(angles, data, alpha=0.1)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi,  variables)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Player Similarity')
    st.pyplot(plt.gcf())
    # output_df[['player_name' ,'nflComparison' + features ]]
    player_bio = player_df.loc[player_df.index[0], 'bio']
    columns_to_display_1 = ['player_name'] + features[0:5]
    columns_to_display_2 = ['player_name'] +  ['school'] + features[5:] + ['nflComparison']

    st.write(output_df[columns_to_display_1])
    st.write(output_df[columns_to_display_2])

    st.write(player_bio)

    # Return the DataFrame with the KNN neighbors and their similarity scores



if st.button("Find Neighbors"):
    # Call the knn_neighbors function with the selected inputs
    result = knn_neighbors(selected_player, selected_position, selected_n)

    # Display the result (you can customize the output as needed)


