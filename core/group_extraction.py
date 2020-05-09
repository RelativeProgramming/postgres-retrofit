#!/usr/bin/python3

import numpy as np
import json
import sys
import networkx as nx
from itertools import combinations
import base64

import config
import db_connection as db_con
import retro_utils as utils


def get_graph(path, graph_type='gml'):
    g = None
    if graph_type == 'gml':
        g = nx.read_gml(path)
    return g


def get_group(name, group_type, vector_dict, extended=None, query='', export_type='full'):
    elems = []
    if group_type == 'categorial':
        if export_type == 'full':
            elems = vector_dict
        else:
            elems = list(vector_dict.keys())
    else:
        if export_type == 'full':
            elems = vector_dict
        else:
            elems = list(vector_dict.keys())
    result = {
        'name': name,
        'type': group_type,
        'elements': elems,
        'query': query
    }
    if extended != None:
        if export_type == 'full':
            result['inferred_elements'] = extended
        else:
            result['inferred_elements'] = list(extended.keys())
    return result


def get_column_groups(graph, we_table_name, terms, con, cur, tokenization_strategy):
    print("Column relation extraction started:")
    result = dict()
    # construct query
    for node in graph.nodes:
        columns_attr = graph.nodes[node]['columns']
        types_attr = graph.nodes[node]['types']
        column_names = zip(columns_attr, types_attr) if type(columns_attr) == list and type(types_attr) == list \
            else [(columns_attr, types_attr)]
        for column_name, column_type in column_names:
            print('Processing %s.%s ...' % (node, column_name))
            vec_dict_fit = dict()
            vec_dict_inferred = dict()

            # Process numeric values
            if column_type == 'number':
                min_value = 0
                max_value = 0
                min_query = "SELECT min(%s) FROM %s" % \
                            ('%s.%s' % (node, column_name), node)
                cur.execute(min_query)
                min_value = cur.fetchall()[0][0]
                max_query = "SELECT max(%s) FROM %s" % \
                            ('%s.%s' % (node, column_name), node)
                cur.execute(max_query)
                max_value = cur.fetchall()[0][0]

                query = "SELECT %s::varchar FROM %s" % \
                        ('%s.%s' % (node, column_name), node)
                cur.execute(query)
                for res in cur.fetchall():
                    term = res[0]
                    if term is None:
                        continue

                    num = float(term)
                    vec = utils.num_to_vec_one_hot(num, min_value, max_value)
                    vec_dict_inferred[term] = dict()
                    vec_dict_inferred[term]['vector'] = base64.encodebytes(
                        vec).decode('ascii')

            else:  # Process string values
                query = "SELECT %s, we.vector, we.id FROM %s LEFT OUTER JOIN %s AS we ON %s = we.word" % (
                    utils.tokenize_sql_variable('%s.%s' % (node, column_name)),
                    node, we_table_name,
                    utils.tokenize_sql_variable('%s.%s' % (node, column_name)))
                cur.execute(query)
                term_vecs = cur.fetchall()

                for (term, vec_bytes, vec_id) in term_vecs:
                    if vec_bytes is not None and column_type != "number":
                        vec_dict_fit[term] = dict()
                        vec_dict_fit[term]['vector'] = base64.encodebytes(
                            vec_bytes).decode('ascii')
                        vec_dict_fit[term]['id'] = int(vec_id)
                    else:
                        if term is None:
                            continue

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
                            vec_dict_inferred[term] = dict()
                            vec_dict_inferred[term]['vector'] = base64.encodebytes(
                                vector.tobytes()).decode('ascii')
            result['%s.%s' % (node, column_name)] = [get_group(
                '%s.%s' % (node, column_name), 'categorial', vec_dict_fit, extended=vec_dict_inferred)]
            # here a clustering approach could be done
    return result


def get_row_groups(graph, we_table_name, con, cur):
    print("Row relation extraction started...")
    result = dict()
    for node in graph.nodes:
        columns = graph.nodes[node]['columns']
        types = graph.nodes[node]['types']
        if type(columns) != list or type(types) != list:
            continue
        columns_types = zip(columns, types)
        for (col1, type1), (col2, type2) in combinations(columns_types, 2):
            rel_name = '%s.%s~%s.%s' % (node, col1, node, col2)
            col1_query_symbol = ('%s.%s' % (node, col1)) if type1 == "number" \
                else utils.tokenize_sql_variable('%s.%s' % (node, col1))
            col2_query_symbol = ('%s.%s' % (node, col2)) if type2 == "number" \
                else utils.tokenize_sql_variable('%s.%s' % (node, col2))
            vec_dict = dict()

            if not (type1 == "number" or type2 == "number"):
                we_query = (
                                   "SELECT %s::varchar, %s::varchar, v1.vector, v2.vector, v1.id, v2.id "
                                   + "FROM %s INNER JOIN %s AS v1 ON %s::varchar = v1.word "
                                   + "INNER JOIN %s AS v2 ON %s::varchar = v2.word") % (
                               col1_query_symbol,
                               col2_query_symbol,
                               node,
                               we_table_name,
                               col1_query_symbol,
                               we_table_name,
                               col2_query_symbol)  # returns (term1, term2, vector1, vector2)
                cur.execute(we_query)
                for (term1, term2, vec1_bytes, vec2_bytes, vec1_id, vec2_id) in cur.fetchall():
                    key = '%s~%s' % (term1, term2)
                    vec_dict[key] = dict()
                    vec_dict[key]['ids'] = [int(vec1_id), int(vec2_id)]

            complete_query = "SELECT %s::varchar, %s::varchar FROM %s" % (col1_query_symbol, col2_query_symbol, node)
            new_group = get_group(rel_name, 'relational',
                                  vec_dict, query=complete_query)
            if rel_name in result:
                result[rel_name].append(new_group)
            else:
                result[rel_name] = [new_group]
    return result


def get_relation_groups(graph, we_table_name, con, cur):
    # Assumption: two tables are only direct related by one foreign key relation
    print("Table relation extraction started:")
    result = dict()
    for (node1, node2, attrs) in graph.edges.data():
        table1, table2 = node1, node2
        key_col1, key_col2 = attrs['col1'], attrs['col2']
        columns_attr1 = graph.nodes[node1]['columns']
        column_names1 = columns_attr1 if type(columns_attr1) == list else [
            columns_attr1
        ]
        columns_attr2 = graph.nodes[node2]['columns']
        column_names2 = columns_attr2 if type(columns_attr2) == list else [
            columns_attr2
        ]
        types_attr1 = graph.nodes[node1]['types']
        types1 = types_attr1 if type(types_attr1) == list else [
            types_attr1
        ]
        types_attr2 = graph.nodes[node2]['types']
        types2 = types_attr2 if type(types_attr2) == list else [
            types_attr2
        ]
        for (col1, type1) in zip(column_names1, types1):
            col1_query_symbol = ('%s.%s' % (table1, col1)) if type1 == "number" \
                else utils.tokenize_sql_variable('%s.%s' % (table1, col1))

            for (col2, type2) in zip(column_names2, types2):
                print('Process %s.%s~%s.%s ...' % (node1, col1, node2, col2))
                col2_query_symbol = ('%s.%s' % (table2, col2)) if type2 == "number" \
                    else utils.tokenize_sql_variable('%s.%s' % (table2, col2))
                # conect source with target
                rel_name = ''
                vec_dict = dict()
                rel_name = '%s.%s~%s.%s' % (node1, col1, node2, col2)
                we_query = ''
                complete_query = ''
                if attrs['name'] == '-':
                    we_query = (
                                       "SELECT %s::varchar, %s::varchar, v1.vector, v2.vector, v1.id, v2.id "
                                       + "FROM %s INNER JOIN %s ON %s.%s = %s.%s "
                                       + "INNER JOIN %s AS v1 ON %s::varchar = v1.word "
                                       + "INNER JOIN %s AS v2 ON %s::varchar = v2.word") % (
                                   col1_query_symbol,
                                   col2_query_symbol,
                                   table1, table2, table1, key_col1, table2, key_col2,
                                   we_table_name, col1_query_symbol,
                                   we_table_name, col2_query_symbol)  # returns (term1, term2, vector1, vector2)
                    # construct complete query for reconstruction
                    complete_query = "SELECT %s::varchar, %s::varchar FROM %s INNER JOIN %s ON %s.%s = %s.%s " \
                                     % (col1_query_symbol,
                                        col2_query_symbol,
                                        table1, table2, table1, key_col1,
                                        table2, key_col2)
                else:
                    pkey_col1 = graph.nodes[node1]['pkey']
                    pkey_col2 = graph.nodes[node2]['pkey']
                    rel_tab_name = attrs['name']
                    we_query = ("SELECT %s::varchar, %s::varchar, v1.vector, v2.vector, v1.id, v2.id "
                                + "FROM %s INNER JOIN %s ON %s.%s = %s.%s "
                                + "INNER JOIN %s ON %s.%s = %s.%s "
                                + "INNER JOIN %s AS v1 ON %s::varchar = v1.word "
                                + "INNER JOIN %s AS v2 ON %s::varchar = v2.word") % (
                                   col1_query_symbol,
                                   col2_query_symbol,
                                   table1, rel_tab_name, table1, pkey_col1,
                                   rel_tab_name, key_col1, table2, table2, pkey_col2, rel_tab_name, key_col2,
                                   we_table_name, col1_query_symbol,
                                   we_table_name, col2_query_symbol)  # returns (term1, term2, vector1, vector2)
                    # construct complete query for reconstruction
                    complete_query = ("SELECT %s::varchar, %s::varchar FROM %s " +
                                      "INNER JOIN %s ON %s.%s = %s.%s "
                                      + "INNER JOIN %s ON %s.%s = %s.%s") % (
                                         col1_query_symbol,
                                         col2_query_symbol, table1,
                                         rel_tab_name, table1, pkey_col1,
                                         rel_tab_name, key_col1, table2,
                                         table2, pkey_col2, rel_tab_name,
                                         key_col2)
                if not (type1 == "number" or type2 == "number"):
                    cur.execute(we_query)
                    for (term1, term2, vec1_bytes, vec2_bytes, vec1_id,
                         vec2_id) in cur.fetchall():
                        key = '%s~%s' % (term1, term2)
                        vec_dict[key] = dict()
                        vec_dict[key]['ids'] = [int(vec1_id), int(vec2_id)]

                new_group = get_group(
                    attrs['name'], 'relational', vec_dict, query=complete_query)
                if rel_name in result:
                    result[rel_name].append(new_group)
                else:
                    result[rel_name] = [new_group]
    return result


def output_groups(groups, filename):
    f = open(filename, 'w')
    f.write(json.dumps(groups))
    f.close()
    return


def update_groups(groups, new_groups):
    for key in new_groups:
        if key in groups:
            groups[key] += new_groups[key]
        else:
            groups[key] = new_groups[key]
    return groups


def main(argc, argv):
    db_config = db_con.get_db_config(path=argv[2])
    con, cur = db_con.create_connection(db_config)

    # get retrofitting config
    conf = config.get_config(argv)

    print('Start loading graph...')
    graph = get_graph(path=conf['SCHEMA_GRAPH_PATH'])
    print('Retrieved graph data')

    groups = dict()

    we_table_name = conf['WE_ORIGINAL_TABLE_NAME']

    # get terms (like radix tree)
    terms = utils.get_terms_from_vector_set(we_table_name, con, cur)

    # get groups of values occuring in the same column
    groups = update_groups(groups, get_column_groups(
        graph, we_table_name, terms, con, cur, conf['TOKENIZATION']))

    # get all relations between text values in two columns in the same table
    groups = update_groups(groups, get_row_groups(
        graph, we_table_name, con, cur))

    # get all relations in the graph
    groups = update_groups(groups, get_relation_groups(
        graph, we_table_name, con, cur))

    # export groups
    print('Export groups ...')
    output_groups(groups, conf['GROUPS_FILE_NAME'])


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
