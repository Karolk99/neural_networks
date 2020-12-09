import pandas as pd
import numpy as np


def create_list_from_csv(column_no, csv_file, separator=';'):
    data = pd.read_csv(filepath_or_buffer=csv_file, sep=separator)
    return data._get_column_array(column_no)


def create_list_from_file(file, row_type, separator=';'):
    table = open(file)
    result = []
    good_data = False
    for line in table:
        columns = line.split(separator)
        if columns[1] == '2020-01-02':
            good_data = True
        if good_data and columns[2] == row_type:
            result.append(float(columns[3]))
    table.close()
    return result


def write_to_file(list1, list2, noumber):
    file = open("test.txt", 'w+')
    for i in range(noumber):
        s = str(list1[i]) + '         ' + str(list2[i])
        file.write(s)
        s = '\n'
        file.write(s)

