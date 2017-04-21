import codecs

lines = []

f_in = '/home/lily/daf62/cartoons/data/captions.txt'
f_out = '/home/lily/daf62/cartoons/data/_captions.txt'

with codecs.open(f_in, 'r', encoding='latin-1') as f:
    for line in f:
        split = line.split('\t')
        if len(split) != 2:
            continue
        num, description = split
        try:
            idx = int(num)
            lines.append(line)
        except:
            continue

with codecs.open(f_out, 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line)
