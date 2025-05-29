from scipy.optimize import curve_fit

import json
data = {
    "layernum[2]_bsz1_seq1024": 44.7532958984375,
    "layernum[2]_bsz2_seq1024": 60.48530311584473,
    "layernum[2]_bsz3_seq1024": 76.12503051757812,
    "layernum[2]_bsz4_seq1024": 91.3835060119629,
    "layernum[2]_bsz5_seq1024": 106.40481262207031,
    "layernum[2]_bsz6_seq1024": 124.3981315612793,
    "layernum[2]_bsz7_seq1024": 138.51839599609374,
    "layernum[2]_bsz8_seq1024": 154.392236328125,
    "layernum[2]_bsz9_seq1024": 169.49546661376954,
    "layernum[2]_bsz10_seq1024": 185.3256591796875,
    "layernum[2]_bsz11_seq1024": 203.16634368896484,
    "layernum[2]_bsz12_seq1024": 218.39861145019532,
    "layernum[4]_bsz1_seq1024": 75.3896484375,
    "layernum[4]_bsz2_seq1024": 101.76360931396485,
    "layernum[4]_bsz3_seq1024": 130.1878860473633,
    "layernum[4]_bsz4_seq1024": 156.3745086669922,
    "layernum[4]_bsz5_seq1024": 182.56463928222655,
    "layernum[4]_bsz6_seq1024": 213.85322113037108,
    "layernum[4]_bsz7_seq1024": 240.69671630859375,
    "layernum[4]_bsz8_seq1024": 266.33167419433596,
    "layernum[4]_bsz9_seq1024": 293.07745666503905,
    "layernum[4]_bsz10_seq1024": 320.1240966796875,
    "layernum[4]_bsz11_seq1024": 351.4576812744141,
    "layernum[4]_bsz12_seq1024": 378.5106994628906,
    "layertype_0_bsz1_seq1024": 15.31817626953125,
    "layertype_other_bsz1_seq1024": 14.116943359375,
    "layertype_0_bsz2_seq1024": 10.31957654953003,
    "layertype_other_bsz2_seq1024": 9.603498458862305,
    "layertype_0_bsz3_seq1024": 9.010475921630862,
    "layertype_other_bsz3_seq1024": 7.354058329264319,
    "layertype_0_bsz4_seq1024": 8.123875331878663,
    "layertype_other_bsz4_seq1024": 6.598125839233397,
    "layertype_0_bsz5_seq1024": 7.615982666015624,
    "layertype_other_bsz5_seq1024": 6.048997192382814,
    "layertype_0_bsz6_seq1024": 7.4545907974243155,
    "layertype_other_bsz6_seq1024": 5.823840332031253,
    "layertype_0_bsz7_seq1024": 7.298451450892857,
    "layertype_other_bsz7_seq1024": 5.191439383370534,
    "layertype_0_bsz8_seq1024": 6.996214866638185,
    "layertype_other_bsz8_seq1024": 5.3065998077392535,
    "layertype_0_bsz9_seq1024": 6.865666113959417,
    "layertype_other_bsz9_seq1024": 5.101497395833336,
    "layertype_0_bsz10_seq1024": 6.739921875,
    "layertype_other_bsz10_seq1024": 5.05272216796875,
    "layertype_0_bsz11_seq1024": 6.740515344793147,
    "layertype_other_bsz11_seq1024": 4.9886369185014185,
    "layertype_0_bsz12_seq1024": 6.671337000528971,
    "layertype_other_bsz12_seq1024": 4.8572102864583355
}

def linear_func(x, a, b):
    return a * x + b

data_divided = {k: v / 3 for k, v in data.items()}
json.dump(data_divided, open("data_divided.json", "w"), indent=4)

x_data = []
y_data = []

for k in data_divided.keys():
    if k.startswith("layertype_0_bsz"):
        x_data.append(int(k.split('_')[2][3:]))
        y_data.append(data_divided[k] * x_data[-1])
    elif k.startswith("layertype_other_bsz"):
        pass
        # x_data.append(int(k.split('_')[2][3:]) + 0.5)  # Offset for 'other' layer type
    else:
        continue

popt, _ = curve_fit(linear_func, x_data, y_data)
print(f"Fitted parameters: {popt}")

x_data = []
y_data = []
for k in data_divided.keys():
    if k.startswith("layertype_other_bsz"):
        x_data.append(int(k.split('_')[2][3:]))
        y_data.append(data_divided[k] * x_data[-1])
    else:
        continue
popt_other, _ = curve_fit(linear_func, x_data, y_data)
print(f"Fitted parameters for other layer type: {popt_other}")
