from visualization.glicko2 import Glicko2, WIN, LOSS, DRAW
import pandas as pd
from pathlib import Path
import click
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_glicko_scores(df):
    all_players = set(pd.concat([df["player_1"], df["player_2"]]))

    env = Glicko2(tau=0.5)

    ratings = {}
    new_ratings = {}

    for player in all_players:
        agent, n_games = player.split(":")
        n_games = int(n_games)
        ratings[player] = env.create_rating(), agent, n_games

    for player in ratings.keys():
        p_df = df[df["player_1"] == player]

        games = []
        for i in p_df.iloc:
            p2 = (i["player_2"])
            score = i["score"]
            if (int(score) == -1):
                score = LOSS
            elif (int(score) == 0):
                score = DRAW
            elif (int(score) == 1):
                score = WIN
            else:
                exit("Should never be here!")
            games.append([score, ratings[p2][0]])

        p_df = df[df["player_2"] == player]

        for i in p_df.iloc:
            p2 = (i["player_1"])
            score = i["score"]
            if (int(score) == -1):
                score = WIN
            elif (int(score) == 0):
                score = DRAW
            elif (int(score) == 1):
                score = LOSS
            else:
                exit("Should never be here!")

            games.append([score, ratings[p2][0]])
        rated = env.rate(ratings[player][0], games)
        new_ratings[player] = rated, ratings[player][1], ratings[player][2]
        print(player, rated)

    map_results = {"n_games": [],
                   "glicko2": [],
                   "glicko2_upper": [],
                   "glicko2_lower": [],
                   "agent_name": [],
                   }
    for player in new_ratings.keys():
        r, agent, n_games = new_ratings[player]
        map_results["n_games"].append(n_games)
        map_results["glicko2"].append(r.mu)
        map_results["glicko2_lower"].append(r.mu - r.phi)
        map_results["glicko2_upper"].append(r.mu + r.phi)
        map_results["agent_name"].append(agent)



    df_results_all = pd.DataFrame(data=map_results)

    return df_results_all



import numpy
def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
            raise ValueError(
                "Input vector needs to be bigger than window size.")
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError(
                "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=numpy.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=numpy.ones(window_len,'d')
        else:
                w=eval('numpy.'+window+'(window_len)')
        y=numpy.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):

    df = pd.read_csv(input_filepath, index_col=0)

    df_glicko = calculate_glicko_scores(df)

    # replicate random player for all possible n_games
    random_glicko = df_glicko[df_glicko["n_games"] < 0]

    df_glicko = df_glicko[df_glicko["n_games"] > 0]

    n_games = set(df_glicko["n_games"])

    map_results = {"n_games": [],
                   "glicko2": [],
                   "glicko2_upper": [],
                   "glicko2_lower": [],
                   "agent_name": [],
                   }


    for game in n_games:
        map_results["n_games"].append(game)
        map_results["glicko2"].append(random_glicko["glicko2"].to_numpy()[0])
        map_results["glicko2_lower"].append(random_glicko["glicko2_lower"].to_numpy()[0])
        map_results["glicko2_upper"].append(random_glicko["glicko2_upper"].to_numpy()[0])
        map_results["agent_name"].append(random_glicko["agent_name"].to_numpy()[0])

    df_random = pd.DataFrame(data = map_results)

    df_glicko = df_glicko.append(df_random)

    path = Path(input_filepath).parents[0]
    name = Path(input_filepath).stem
    df_filename = f"{path}/{name}_glicko.csv"


    df_glicko.to_csv(df_filename)
    agent_names = list(set(df_glicko["agent_name"]))
    agent_names.sort()
    for colour, agent in enumerate(agent_names):
        df_results = df_glicko[df_glicko["agent_name"] == agent]
        df_results = df_results.sort_values(by=['n_games'])

        colour = sns.color_palette("deep")[colour]
        x = (df_results["n_games"].to_numpy())
        y = (df_results["glicko2"].to_numpy())
        if(y.size > 50):
            y = smooth(y)
        print(x.shape, y.shape)

        plt.plot(x, y, color=colour,
                 label=agent)
        # plt.plot(xfit, yfit, '-', color='gray')
        #
        y_lower = df_results["glicko2_lower"].to_numpy()
        y_upper = df_results["glicko2_upper"].to_numpy()
        if(y_upper.size > 50):
            y_lower = smooth(y_lower)
            y_upper = smooth(y_upper)
        plt.fill_between(df_results["n_games"], y_lower,
                         y_upper,
                         color=colour,
                         alpha=0.5)
    plt.legend()
    plt.ylabel("Glicko2 score")
    plt.xlabel(
        "Number of training games")
    plt.savefig(f"reports/figures/{name}.pdf", bbox_inches='tight')
    #plt.savefig(f"reports/figures/{name}.png", bbox_inches='tight')
    #plt.show()


if __name__ == '__main__':
    main()
