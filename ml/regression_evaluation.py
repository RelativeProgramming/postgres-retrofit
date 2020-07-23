

import os
import sys
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import regularizers
# from sklearn import decomposition

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, 'core/')

import db_connection as db_con
import eval_utils as utils
import json

MODEL_SAVE_PATH = '/tmp/best_model'
OUTPUT_NAME = 'food_cat_classify_'

VEC_TABLE_TEMPL = '{vec_table}'
VEC_TABLE1_TEMPL = '{vec_table1}'
VEC_TABLE2_TEMPL = '{vec_table2}'

ITERATIONS = 20


def load_config(file_name):
    f = open(file_name, 'r')
    config = json.load(f)
    f.close()
    return config


def output_result(result, filename):
    f = open(filename, 'w')
    json.dump(result, f)
    return


def construct_data_list(query_result):
    id_lookup = dict()
    count = 0
    for elem in query_result:  # id value vec [vec2]
        if type(elem[1]) == type(None):
            continue
        count += 1
        if len(elem) == 3:
            value = float(elem[1])
            vec = utils.parse_bin_vec(elem[2])
            id_lookup[elem[0]] = (value, vec / np.linalg.norm(vec))
        if len(elem) == 4:
            value = float(elem[1])
            v1 = utils.parse_bin_vec(elem[2])
            v2 = utils.parse_bin_vec(elem[3])
            combined = np.array(
                [v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)]).flatten()
            id_lookup[elem[0]] = (value, combined / np.linalg.norm(combined))
    print("Data Entries: ", count)
    return id_lookup


def vec_dict2matrix(vecs, max_size=10000000):
    data = []
    key_data = list(vecs.keys())
    key_data = key_data[:max_size]
    for i in range(len(key_data)):
        data.append(vecs[key_data[i]][1])
    data = np.array(data)
    return key_data, data


def get_data(query, size, con, cur):
    samples = []
    cur.execute(query)
    id_lookup = construct_data_list(cur.fetchall())
    id_keylist = list(id_lookup.keys())
    random.shuffle(id_keylist)
    train_data = dict()
    for elem in id_keylist[:size]:
        train_data[elem] = id_lookup[elem]

    # key_data, data = vec_dict2matrix(train_data)
    # pca = decomposition.PCA(n_components=300)
    # reduced = pca.fit_transform(data)
    # for i, key in enumerate(key_data):
    #     train_data[key] = (train_data[key][0], reduced[i])

    return train_data


def classify(train_data, test_data):
    x = np.array([x[1] for x in train_data])
    y = np.array([[x[0]] for x in train_data])
    x_test = np.array([x[1] for x in test_data])
    y_test = np.array([[x[0]] for x in test_data])

    backup_model_name = MODEL_SAVE_PATH + utils.create_timestamp() + '.weights'

    model = Sequential()
    # The Input Layer :
    model.add(Dense(300, kernel_initializer='normal',
                    input_dim=x.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    # The Hidden Layers :
    model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, kernel_initializer='normal', activation='relu'))

    # The Output Layer :
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam',
                  metrics=['mean_absolute_error'])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
    checkpointer = ModelCheckpoint(
        filepath=backup_model_name, verbose=1, save_best_only=True, monitor='val_loss')
    model.fit(x, y, epochs=500, batch_size=32,
              validation_split=0.1, callbacks=[es, checkpointer])
    score = model.evaluate(x_test, y_test)

    os.remove(backup_model_name)

    # TODO predict on test data
    # TODO execute multiple times
    return score[1]  # TODO return mu and std


def main(argc, argv):
    # Arguments: db_config_file, regression_config_file
    if argc < 3:
        print('Not enough arguments')
        return
    db_config_file = argv[1]
    db_config = db_con.get_db_config(db_config_file)
    con, cur = db_con.create_connection(db_config)

    ml_config_file = argv[2]
    ml_config = load_config(ml_config_file)
    query = ml_config["query"]
    train_data_size = ml_config["train-size"]
    table_names = ml_config["table_names"]
    output_folder = ml_config["output_folder"]

    query = utils.get_vector_query(
        query, table_names, VEC_TABLE_TEMPL, VEC_TABLE1_TEMPL, VEC_TABLE2_TEMPL)

    data = get_data(query, train_data_size, con, cur)
    input_data = list(data.values())
    errors = []
    for i in range(ITERATIONS):
        random.shuffle(input_data)
        train_data = input_data[:int(len(input_data) * 0.9)]
        test_data = input_data[int(len(input_data) * 0.9):]
        error = classify(train_data, test_data)
        errors.append(error)
        print('Error', np.mean(errors), '+/-', np.std(errors))
        print("Errors: ", errors)
    mu = np.mean(errors)
    std = np.std(errors)

    output_result(dict({"retro_vecs": {'mu': mu, 'std': std, 'scores': errors}}),
                  output_folder + OUTPUT_NAME + "retro_vecs" + '.json')

    return mu, std, errors


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)

# EXAMPLE QUERY

# SELECT r.id, budget, d.vector
# FROM movies
# INNER JOIN retro_vecs AS r ON concat('movies.title#', replace(movies.title, ' ', '_')) = r.word
# INNER JOIN deepwalk_vecs AS d ON concat('movies.title#', replace(movies.title, ' ', '_')) = d.word
