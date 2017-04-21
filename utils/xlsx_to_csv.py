# Convert an excel file to csv
import xlrd
import csv
import sys
import codecs


def convert(fn_in, fn_out):
    wb = xlrd.open_workbook(fn_in)
    sh = wb.sheet_by_index(0)
    with codecs.open(fn_out, 'w') as f_out:
        wr = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        for row_num in xrange(sh.nrows):
            vals = []
            for val in sh.row_values(row_num):
                if isinstance(val, basestring):
                    vals.append(val.encode('utf-8'))
                else:
                    vals.append(val)
            try:
                wr.writerow(vals)
            except:
                print vals
                return



def main():
    fn_in = '/home/lily/daf62/cartoons/data/ContestIDMapping.xlsx'
    fn_out = '/home/lily/daf62/cartoons/data/id_map.csv'
    if len(sys.argv) > 1:
        fn_in = sys.argv[1]
    if len(sys.argv) > 2:
        fn_out = sys.argv[2]
    convert(fn_in, fn_out)

if __name__ == '__main__':
    main()

