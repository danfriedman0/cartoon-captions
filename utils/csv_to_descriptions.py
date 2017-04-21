# Write the descriptions from the csv file to a text file

import csv

def convert(fn_in, fn_out):
    descriptions = []
    with open(fn_in, 'r') as f_in:
        reader = csv.reader(f_in)
        for i,row in enumerate(reader):
            if i < 3:
                continue
            id_num = row[0].split('.')[0]
            description = ' '.join(row[9:12])
            if len(description) <= 3:
                continue
            descriptions.append((id_num, description))

    with open(fn_out, 'w') as f_out:
        for id_num, description in descriptions:
            f_out.write(id_num + '\t' + description + '\n')

    print 'Wrote {} descriptions'.format(len(descriptions))


def main():
    fn_in = '/home/lily/daf62/cartoons/data/cartoonDescriptions.csv'
    fn_out = '/home/lily/daf62/cartoons/data/descriptions.txt'
    convert(fn_in, fn_out)

if __name__ == '__main__':
    main()

        
