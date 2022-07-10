from glicko2 import Glicko2, WIN, LOSS, DRAW
import pandas as pd
from pathlib import Path
import click
from matplotlib import pyplot as plt
import seaborn as sns


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    #project_dir = Path(__file__).resolve().parents[2]
    df = pd.read_csv(input_filepath, index_col=0)
    all_players = set(pd.concat([df["player_1"], df["player_2"]]))

    env = Glicko2(tau=0.5)

    ratings = {}
    new_ratings = {}

    for player in all_players:
        split = player.split("_")[-1]
        if(split == "Random"):
            split = -10000
        else:
            split = int(split)
        ratings[player] = env.create_rating(), split

    for player in ratings.keys():
        p_df = df[df["player_1"] == player]
        p_df = p_df[p_df["position"] == 0]
        games = []
        for i in p_df.iloc:
            p2 = (i["player_2"])
            score = i["score"]
            if (int(score) == -1):
                score == LOSS
            elif (int(score) == 0):
                score == DRAW
            elif (int(score) == 1):
                score = WIN
            else:
                print("WTF")
            # print(score)
            games.append([score, ratings[p2][0]])

        p_df = df[df["player_2"] == player]
        p_df = p_df[p_df["position"] == 1]

        for i in p_df.iloc:
            p2 = (i["player_1"])
            score = i["score"]
            if (int(score) == -1):
                score == WIN
            elif (int(score) == 0):
                score == DRAW
            elif (int(score) == 1):
                score = LOSS
            else:
                print("WTF")
            # print(score)
            games.append([score, ratings[p2][0]])

        rated = env.rate(ratings[player][0], games)
        new_ratings[player] = rated, ratings[player][1]
        #print(player, rated)
    # start the visualisation

    X = []
    Y = []
    Y_MIN = []
    Y_MAX = []
    for player in new_ratings.keys():
        r, n_games = new_ratings[player]
        print(r,n_games)
        X.append(n_games)
        Y_MIN.append(r.mu - r.phi)
        Y_MAX.append(r.mu + r.phi)
        Y.append(r.mu)

    new_X = [x for _, x in sorted(zip(X, X))]
    new_Y = [y for y, _ in sorted(zip(Y, X), key=lambda pair: pair[1])]
    new_Y_MAX = [y for y, _ in sorted(zip(Y_MAX, X), key=lambda pair: pair[1])]
    new_Y_MIN = [y for y, _ in sorted(zip(Y_MIN, X), key=lambda pair: pair[1])]

    #print(sorted(zip(Y, X), key=lambda pair: pair[1]))
    #exit()

    colour = sns.color_palette("deep")[1]

    plt.plot(new_X, new_Y, color=colour)
    # plt.plot(xfit, yfit, '-', color='gray')
    #
    plt.fill_between(new_X, new_Y_MIN, new_Y_MAX,
                    color=colour,
                    alpha=0.5)
    #plt.set_xlabel(r"%s, $\tau=%.2f$, $p-value=%.2f$" % (feature, kt[0], kt[1]))
    #plt.set_ylabel(outcomes[0])
    plt.show()


if __name__ == '__main__':
    main()
