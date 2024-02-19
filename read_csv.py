import csv


csv.field_size_limit(100000000)


def read(filename:str):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            data.append(line[1])
    return data
