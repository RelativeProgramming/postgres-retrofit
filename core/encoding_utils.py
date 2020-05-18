import numpy as np
from word2number import w2n
from bisect import bisect_left

number_embeddings = dict()
embedded_numbers = []


def text_to_vec(term, vec_bytes, terms, tokenization_settings):
    """
    Encodes a term to a 300-dimensional vector using the word embeddings given in "terms"

    :return: a boolean that indicates, if the vector was inferred and
        the vector itself in the form vector.tobytes()
    """
    tokenization_strategy = tokenization_settings['TEXT_TOKENIZATION']
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
        while i <= len(splits) or last_match[1] is not None:
            subword = '_'.join(splits[j:i])
            if subword in current[0]:
                current = current[0][subword]
                if current[1] is not None:
                    last_match = (i, np.fromstring(bytes(current[1]), dtype='float32'), current[2])
            else:
                if last_match[1] is not None:
                    if vector is not None:
                        if tokenization_strategy == 'log10':
                            vector += last_match[1] * np.log10(last_match[2])
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
        if vector is not None:
            vector /= count
            return True, vector.tobytes()
        else:
            return True, None


def num_to_vec_one_hot(num, min_value, max_value, column_vec):
    """
    Encodes a number to a 300-dimensional vector using a one-hot encoding

    -> divides the range [min_value, max_value] in 300 equally spaced sub-ranges, that are used for encoding

    :return: vector in the form vector.tobytes()
    """
    if max_value >= num >= min_value:
        range_size = (max_value - min_value) / 300
        index = int((num - min_value) // range_size)
        return bucket_to_vec_one_hot(min(299, index), column_vec)
    else:
        return np.zeros(300, dtype='float32')


def bucket_to_vec_one_hot(bucket, column_vec):
    """
    Encodes a bucket index to a 300-dimensional vector using one-hot encoding with the values 0.0 and 1.0.

    :return: vector in the form vector.tobytes()
    """
    if bucket_valid(bucket):
        vec = np.zeros(300, dtype='float32')
        vec[bucket] = 1.0
        if column_vec is not None:
            cv = np.frombuffer(column_vec, dtype='float32')
            vec += cv
            vec /= 2
        return vec.tobytes()
    else:
        return np.zeros(300, dtype='float32').tobytes()


def num_to_vec_one_hot_gaussian(num, min_value, max_value, sd):
    """
    Encodes a number to a 300-dimensional vector using a one-hot encoding with a gaussian filter

    -> divides the range [min_value, max_value] in 300 equally spaced sub-ranges, that are used for encoding

    :return: vector in the form vector.tobytes()
    """
    if max_value >= num >= min_value:
        range_size = (max_value - min_value) / 300
        index = int((num - min_value) // range_size)
        return bucket_to_vec_one_hot_gaussian(min(299, index), sd)
    else:
        return np.zeros(300, dtype='float32')


def bucket_to_vec_one_hot_gaussian(bucket, sd):
    """
    Encodes a bucket index to a 300-dimensional vector using one-hot encoding with a gaussian filter

    :return: vector in the form vector.tobytes()
    """
    if bucket_valid(bucket):
        vec = np.zeros(300, dtype='float32')
        for x in range(300):
            vec[x] = gaussian(x*0.5, sd, bucket*0.5)  # the factor 0.2 streches the function
        return vec.tobytes()
    else:
        return np.zeros(300, dtype='float32').tobytes()


def num_to_vec_we_regression(num):
    """
    Encodes a number to a 300-dimensional vector using word embeddings of nearby numbers.

    :return: vector in the form vector.tobytes()
    """
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


def num_to_vec_unary(num, min_value, max_value):
    """
    Encodes a number to a 300-dimensional vector using unary encoding with the values -1.0 and 1.0.

    -> divides the range [min_value, max_value] in 300 equally spaced sub-ranges, that are used for encoding

    :return: vector in the form vector.tobytes()
    """
    if max_value >= num >= min_value:
        range_size = (max_value - min_value) / 300
        index = int((num - min_value) // range_size)
        return bucket_to_vec_unary(min(299, index))
    else:
        return np.zeros(300, dtype='float32').tobytes()


def bucket_to_vec_unary(bucket):
    """
    Encodes a bucket index to a 300-dimensional vector using unary encoding with the values -1.0 and 1.0.

    :return: vector in the form vector.tobytes()
    """
    if bucket_valid(bucket):
        vec = np.zeros(300, dtype='float32')
        if bucket < 299:
            vec[bucket+1:300] -= 1.0
        vec[0:bucket+1] = 1.0
        return vec.tobytes()
    else:
        return np.zeros(300, dtype='float32').tobytes()


def generate_random_vec():
    """
    Generates a random 300-dimensional vector, that is randomly filled with values between -1.0 and 1.0

    :return: vector in the form vector.tobytes()
    """
    vec = np.zeros(300, dtype='float32')
    for i in range(300):
        vec[i] = (np.random.random() * 2) - 1.0
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


def bucket_valid(bucket):
    """
    checks if given bucket index is inside the valid range [0, 300)
    """
    return 0 <= bucket < 300


def gaussian(x, sd, mean):
    return (1/(sd*np.sqrt(2*np.pi)))*np.power(np.e, -0.5*np.square((x-mean)/sd))
