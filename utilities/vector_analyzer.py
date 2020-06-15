import sys
import json
import base64
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import db_connection as db_con


def main(argc, argv):
    print('Load file')
    file = open('../output/groups.json')
    print('Load json content')
    groups = json.load(file)
    group_sugar_values = groups['products.sugars_100g'][0]['inferred_elements']
    for value in group_sugar_values:
        group_sugar_values[value] = np.fromstring(base64.decodestring(bytes(group_sugar_values[value]['vector'], 'ascii')), dtype='float32')

    print('Load database content')
    db_config = db_con.get_db_config('../config/db_config.json')
    con, cur = db_con.create_connection(db_config)
    query = "SELECT DISTINCT products.product_name, products.sugars_100g::varchar, retro_vecs.vector FROM products JOIN retro_vecs" +\
            " ON ('products.product_name#' || products.product_name) = retro_vecs.word WHERE products.sugars_100g IS NOT NULL"
    cur.execute(query)
    product_values = dict()
    for name, sugar, vec in cur.fetchall():
        if product_values.get(sugar) is None:
            product_values[sugar] = []
        product_values[sugar].append((np.frombuffer(vec, dtype='float32'), name))

    number_to_test = '0'
    first = product_values.get(number_to_test)[0][0]
    for vec, name in product_values.get(number_to_test):
        similarity = cosine_similarity(vec.reshape(1, -1), first.reshape(1, -1))
        # print(cosine_similarity(vec.reshape(1, -1), group_sugar_values[number_to_test].reshape(1, -1)))
        print(name, ': ', similarity)

    print('! set breakpoint here !')


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
