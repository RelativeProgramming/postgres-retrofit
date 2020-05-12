import base64
import json
import re
from word2number import w2n
from bisect import bisect_left

import numpy as np


number_embeddings = dict()
embedded_numbers = []


def parse_groups(group_filename, vectors_encoded=True):
    f = open(group_filename)
    groups = json.load(f)
    for key in groups:
        for group in groups[key]:
            for key in group['elements']:
                if group['type'] == 'categorial':
                    if vectors_encoded:
                        group['elements'][key]['vector'] = np.fromstring(base64.decodestring(
                            bytes(group['elements'][key]['vector'], 'ascii')), dtype='float32')
                    else:
                        group['elements'][key]['vector'] = group['elements'][key]['vector']
            if 'inferred_elements' in group:
                if group['type'] == 'categorial':
                    for key in group['inferred_elements']:
                        if vectors_encoded:
                            group['inferred_elements'][key]['vector'] = np.fromstring(base64.decodestring(
                                bytes(group['inferred_elements'][key]['vector'], 'ascii')), dtype='float32')
                        else:
                            group['inferred_elements'][key]['vector'] = group['inferred_elements'][key]['vector']
    return groups


def get_data_columns_from_group_data(groups):
    result = set()
    for key in groups:
        if groups[key][0]['type'] == 'categorial':
            result.add((key, groups[key][0]['data_type']))
    return list(result)


def get_column_data_from_label(label, type):
    if type == 'column':
        try:
            table_name, column_name = label.split('.')
            return table_name, column_name
        except:
            print('ERROR: Can not decode %s into table name and column name' %
                  (label))
            return
    if type == 'relation':
        try:
            c1, c2 = label.split('~')
            c1_table_name, c1_column_name = c1.split('.')
            c2_table_name, c2_column_name = c2.split('.')
            return c1_table_name, c1_column_name, c2_table_name, c2_column_name
        except:
            print('ERROR: Can not decode relation label %s ' % (label))
            return


def get_label(x, y): return '%s#%s' % (x, y)


def tokenize_sql_variable(name):
    return "regexp_replace(%s, '[\.#~\s\xa0,\(\)/\[\]:]+', '_', 'g')" % (name)


def tokenize(term):
    if type(term) == str:
        return re.sub('[\.#~\s,\(\)/\[\]:]+', '_', str(term))
    else:
        return ''


def get_terms(columns, con, cur):
    result = dict()
    for column, data_type in columns:
        table_name, column_name = column.split(
            '.')  # TODO get this in an encoding save way
        # construct sql query
        sql_query = "SELECT %s::varchar FROM %s" % (column_name, table_name)
        cur.execute(sql_query)
        if data_type == 'number':
            result[column] = [x[0] for x in cur.fetchall()]
        else:
            result[column] = [tokenize(x[0]) for x in cur.fetchall()]
        result[column] = list(set(result[column]))  # remove duplicates
    return result


def construct_index_lookup(list_obj):
    result = dict()
    for i in range(len(list_obj)):
        result[list_obj[i]] = i
    return result


def get_dist_params(vectors):
    # returns the distribution parameter for vector elments
    m_value = 0
    count = 0
    values = []
    for key in vectors:
        max_inst = 0
        for term in vectors[key]:
            m_value += np.mean(vectors[key][term])
            values.extend([x for x in vectors[key][term]])
            max_inst += 1
            count += 1
            if max_inst > 100:
                break
    m_value /= count
    s_value = np.mean((np.array(values) - m_value) ** 2)
    return m_value, s_value


def execute_threads_from_pool(thread_pool, verbose=False):
    while (len(thread_pool) > 0):
        try:
            next = thread_pool.pop()
            if verbose:
                print('Number of threads:', len(thread_pool))
            next.start()
            next.join()
        except:
            print("Warning: threadpool.pop() failed")
    return


def get_vectors_for_present_terms_from_group_file(data_columns, groups_info):
    result_present = dict()
    dim = 0
    for column, data_type in data_columns:
        group = groups_info[column][0]['elements']
        group_extended = groups_info[column][0]['inferred_elements']
        result_present[column] = dict()
        for term in group:
            result_present[column][term] = np.array(
                group[term]['vector'], dtype='float32')
            dim = len(result_present[column][term])
        for term in group_extended:
            result_present[column][term] = np.array(
                group_extended[term]['vector'], dtype='float32')
            dim = len(result_present[column][term])
    return result_present, dim


def get_terms_from_vector_set(vec_table_name, con, cur):
    print("Getting terms from vector table:")
    QUERY_TMPL = "SELECT word, vector, id FROM %s WHERE id >= %d AND id < %d"
    BATCH_SIZE = 500000
    term_dict = dict()
    min_id = 0
    max_id = BATCH_SIZE
    while True:
        print("%s to %s..." % (min_id, max_id))
        query = QUERY_TMPL % (vec_table_name, min_id, max_id)
        cur.execute(query)
        term_list = [x for x in cur.fetchall()]
        if len(term_list) < 1:
            break
        for (term, vector, freq) in term_list:
            splits = term.split('_')
            current = [term_dict, None, -1]
            i = 1
            while i <= len(splits):
                subterm = '_'.join(splits[:i])
                if subterm in current[0]:
                    current = current[0][subterm]
                else:
                    current[0][subterm] = [dict(), None, -1]
                    current = current[0][subterm]
                i += 1
            current[1] = vector
            current[2] = freq
        min_id = max_id
        max_id += BATCH_SIZE
    return term_dict


def text_to_vec(term, vec_bytes, terms, tokenization_strategy):
    """
    Encodes a term to a 300-dimensional byte vector using the word embeddings given in "terms"

    :return: a boolean that indicates, if the vector was inferred and
        the vector itself in the form vector.tobytes()
    """
    if vec_bytes is not None:
        return False, vec_bytes
    else:
        if term is None:
            return True, None

        splits = [x.replace('_', '') for x in term.split('_')]
        i = 1
        j = 0
        current = [terms, None, -1]
        vector = None
        last_match = (0, None, -1)
        count = 0
        while (i <= len(splits)) or (type(last_match[1]) != type(None)):
            subword = '_'.join(splits[j:i])
            if subword in current[0]:
                current = current[0][subword]
                if current[1] is not None:
                    last_match = (i, np.fromstring(
                        bytes(current[1]), dtype='float32'), current[2])
            else:
                if type(last_match[1]) != type(None):
                    if type(vector) != type(None):
                        if tokenization_strategy == 'log10':
                            vector += last_match[1] * \
                                      np.log10(last_match[2])
                            count += np.log10(last_match[2])
                        else:  # 'simple' or different
                            vector += last_match[1]
                            count += 1
                    else:
                        if tokenization_strategy == 'log10':
                            vector = last_match[1] * \
                                     np.log10(last_match[2])
                            count += np.log10(last_match[2])
                        else:  # 'simple' or different
                            vector = last_match[1]
                            count += 1
                    j = last_match[0]
                    i = j
                    last_match = (0, None, -1)
                else:
                    j += 1
                    i = j
                current = [terms, None, -1]
            i += 1
        if type(vector) != type(None):
            vector /= count
            return True, vector.tobytes()
        else:
            return True, None


def num_to_vec_one_hot(num, min_value, max_value, column_vec):
    """
    Encodes a number to a 300-dimensional byte vector using byte-wise one-hot encoding

    -> divides the range [min_value, max_value] in 300 equally spaced sub-ranges, that are used for encoding
    """
    vec = np.zeros(300, dtype='float32')
    if max_value >= num >= min_value:
        range_size = (max_value - min_value) / 300
        index = int((num - min_value) // range_size)
        vec[min(299, index)] = 1.0
    if column_vec is not None:
        cv = np.frombuffer(column_vec, dtype='float32')
        vec += cv
        vec /= 2
    return vec.tobytes()


def num_to_vec_we_regression(num):
    if number_embeddings.get(num) is not None:
        return number_embeddings[num]
    else:
        closest_numbers = get_neighbors(embedded_numbers, num)
        result = np.zeros(300, dtype='float32')
        for i in closest_numbers:
            vec = np.frombuffer(number_embeddings[i], dtype='float32')
            result += vec
        result /= len(closest_numbers)
        return result.tobytes()


def generate_random_vec():
    vec = np.zeros(300, dtype='float32')
    for i in range(300):
        vec[i] = (np.random.random()*2)-1.0
    return vec.tobytes()


def initialize_numeric_word_embeddings(cur, we_table_name):
    """
    Traverses all word embeddings that represent numbers and saves them to number_embeddings.

    This function should be called when using the we-regression numeric tokenization strategy
    """
    if len(number_embeddings.keys()) == 0:
        print('Initializing numeric word embeddings')
        w2n.print_function = None
        query = 'SELECT word::varchar, vector FROM %s' % we_table_name
        cur.execute(query)
        result = dict()
        count = 0
        for word, vector in cur.fetchall():
            try:
                number = w2n.word_to_num(word)
                if number is not None:
                    if result.get(number) is None:
                        result[number] = []
                    result[number].append(vector)
                    count += 1
            except ValueError:
                pass
        print("Numeral word embeddings found: %s" % count)
        for key in result.keys():
            vectors = list(map(lambda x: np.frombuffer(x, dtype='float32'), result[key]))
            final_vec = np.zeros(300, dtype='float32')
            for vec in vectors:
                final_vec += vec
            final_vec /= len(vectors)
            number_embeddings[key] = final_vec.tobytes()
            embedded_numbers.append(key)
        embedded_numbers.sort()


def get_neighbors(sorted_list, value):
    """
    Returns the neighboring value(s) to value, that are contained in sorted_list.

    If two numbers are equally close, return the smallest number.

    Source: https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    """
    pos = bisect_left(sorted_list, value)
    if pos == 0:
        return [sorted_list[0]]
    if pos == len(sorted_list):
        return [sorted_list[-1]]
    before = sorted_list[pos - 1]
    after = sorted_list[pos]
    return [before, after]
