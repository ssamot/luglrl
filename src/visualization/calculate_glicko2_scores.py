from glicko2 import Glicko2, WIN, LOSS, DRAW
import pandas as pd
from pathlib import Path
import click
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_glicko_scores(input_filepath):
    # project_dir = Path(__file__).resolve().parents[2]
    df = pd.read_csv(input_filepath, index_col=0)

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
    return new_ratings
    

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):

    new_ratings = calculate_glicko_scores(input_filepath)

    path = Path(input_filepath).parents[0]
    name = Path(input_filepath).stem
    df_filename = f"{path}/{name}_glicko.csv"

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
    df_results_all.to_csv(df_filename)
    for colour, agent in enumerate(set(df_results_all["agent_name"])):
        df_results = df_results_all[df_results_all["agent_name"] == agent]
        df_results = df_results.sort_values(by=['n_games'])

        colour = sns.color_palette("deep")[colour]

        plt.plot(df_results["n_games"], df_results["glicko2"], color=colour,
                 label=agent)
        # plt.plot(xfit, yfit, '-', color='gray')
        #
        plt.fill_between(df_results["n_games"], df_results["glicko2_lower"],
                         df_results["glicko2_upper"],
                         color=colour,
                         alpha=0.5)
    plt.legend()
    plt.ylabel("Glicko2 score")
    plt.xlabel(
        "Number of training games (with negative games indicating uniform random player)")
    plt.savefig(f"reports/figures/{name}.pdf", bbox_inches='tight')
    plt.savefig(f"reports/figures/{name}.png", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
