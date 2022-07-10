import keras
from keras import layers
import keras.backend as K
import numpy as np
from tqdm import trange
import shutil
from pathlib import Path
import graycode
from matplotlib import pyplot as plt



if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    model = keras.models.load_model(f"{project_dir}/models/autoencoder_test.keras")
    encoder = keras.models.load_model(
        f"{project_dir}/models/encoder_test.keras")
    codes = graycode.gen_gray_codes(6)
    print(codes)

    ps = []
    for i, code in enumerate(codes):
        code = graycode.tc_to_gray_code(i)
        c = '{:06b}'.format(graycode.tc_to_gray_code(i))
        p  = np.fromstring(",".join(c),count = 6, dtype="int", sep=",")
        #print(c, '{:06b}'.format(graycode.tc_to_gray_code(i)), )
        ps.append(p)
    ps = np.array(ps)

    end = encoder.predict(ps)


    colours = []
    for i in range(len(ps)):
        c =  np.linalg.norm(ps[i] - ps[0])**2
        d = np.linalg.norm(end[i] - end[0])
        print(i, ps[i], end[i],c, np.abs(d -c ))
        colours.append(c)

    #print(end.shape)
    #colours = list(range(len(codes)))
    #colours = codes
    #print(colours)
    plt.scatter(end.T[0], end.T[1], c = colours )
    plt.show()

    # for i in range(len(end)-1):
    #     d_pred = np.linalg.norm(end[i] - end[i+1])
    #
    #     d = np.linalg.norm(ps[i] - ps[i+1])
    #     print(d, d_pred)