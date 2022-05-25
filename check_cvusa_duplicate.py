from PIL import Image
import hashlib
import os
import json

# Change this line yo YOUR dataset directory
dataset_path = os.path.join("CVUSA", "dataset")

csv_file = os.path.join(dataset_path, "splits", "train-19zl.csv")

md5_sat_dict = {}
md5_grd_dict = {}

duplicate_sat_dict = {}
duplicate_grd_dict = {}

csv = open(csv_file, 'r')

all_sat = []
all_grd = []

for i, entry in enumerate(csv.readlines()):
    data = entry.strip().split(",")
    sat = data[0]
    grd = data[1]

    all_sat.append(sat)
    all_grd.append(grd)

    sat_hash = hashlib.md5(Image.open(os.path.join(dataset_path, sat)).tobytes()).hexdigest()
    grd_hash = hashlib.md5(Image.open(os.path.join(dataset_path, grd)).tobytes()).hexdigest()

    #for sat
    if sat_hash not in md5_sat_dict.keys():
        md5_sat_dict[sat_hash] = [i]
    else:
        dup_items = []
        for k in md5_sat_dict[sat_hash]:
            dup_items.append(all_sat[k])
        dup_items.append(sat)
        duplicate_sat_dict[sat_hash] = dup_items

        md5_sat_dict[sat_hash].append(i)

    #for grd
    if grd_hash not in md5_grd_dict.keys():
        md5_grd_dict[grd_hash] = [i]
    else:
        dup_items = []
        for k in md5_grd_dict[grd_hash]:
            dup_items.append(all_grd[k])
        dup_items.append(grd)
        duplicate_grd_dict[grd_hash] = dup_items

        md5_grd_dict[grd_hash].append(i)

csv.close()

print(duplicate_grd_dict)
print(duplicate_sat_dict)

with open("duplicate_grd.json", "w") as outfile:
    json.dump(duplicate_grd_dict, outfile)

with open("duplicate_sat.json", "w") as outfile:
    json.dump(duplicate_sat_dict, outfile)
