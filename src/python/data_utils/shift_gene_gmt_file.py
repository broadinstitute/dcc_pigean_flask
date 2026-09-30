

#!/usr/bin/env python3

# run command
# 
# python3 remove_second_column.py in-file out-file
#

import argparse
import csv


def main():
    parser = argparse.ArgumentParser(
        description="Remove the second column from a tab-delimited file."
    )
    parser.add_argument("in_file", help="Input tab-delimited file")
    parser.add_argument("out_file", help="Output file")

    args = parser.parse_args()

    with open(args.in_file, "r", newline="") as infile, \
         open(args.out_file, "w", newline="") as outfile:

        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t", lineterminator="\n")

        for row in reader:
            if len(row) >= 2:
                del row[1]

            writer.writerow(row)


if __name__ == "__main__":
    main()