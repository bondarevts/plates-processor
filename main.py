import csv
from typing import List

PLATES_DESCRIPTION_CSV = 'plates-example.csv'


def get_names() -> List[str]:
    with open(PLATES_DESCRIPTION_CSV, newline='') as csv_file:
        next(csv_file, None)

        return [record[1] for record in csv.reader(csv_file, dialect='excel')]


def main():
    print(get_names(), len(get_names()))


if __name__ == '__main__':
    main()
