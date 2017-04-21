# Write a json file mapping cartoon descriptions to captions
import json
import codecs
import csv

def is_number(s):
    try:
        int(s)
        return True
    except:
        return False

def make_dict(descriptions_fn, captions_fn, id_map_fn, fn_out):
    num_to_id = {}
    with open(id_map_fn, 'r') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i == 0:
                continue
            id_ = int(float(row[0]))
            num = int(float(row[2]))
            num_to_id[num] = id_

    id_to_description = {}
    with codecs.open(descriptions_fn, 'rb', encoding='utf-8') as f:
        for line in f:
            try:
                num, description = line.split('\t')
            except:
                print line
                continue
            if len(description) == 0:
                break
            id_ = int(num)
            id_to_description[id_] = description

    num_to_captions = {}
    with codecs.open(captions_fn, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                num, caption = line.split('\t')
            except:
                continue
            if not is_number(num):
                continue
            num = int(num)
            if num not in num_to_captions:
                num_to_captions[num] = []
            num_to_captions[num].append(caption)


    pairs = []
    for num in num_to_captions:
        if num not in num_to_id:
            print 'missing id:', num
            continue
        id_ = num_to_id[num]
        if id_ not in id_to_description:
            print 'missing description:', id_
            continue
        description = id_to_description[id_]
        pairs.append((description, num_to_captions[num]))

    with codecs.open(fn_out, 'wb', encoding='utf-8') as f_out:
        json.dump(pairs, f_out)


descriptions_fn = '/home/lily/daf62/cartoons/data/descriptions.txt'
captions_fn = '/home/lily/daf62/cartoons/data/captions.txt'
id_to_map_fn = '/home/lily/daf62/cartoons/data/id_map.csv'
fn_out = '/home/lily/daf62/cartoons/data/dataset.json'

make_dict(descriptions_fn, captions_fn, id_to_map_fn, fn_out)
