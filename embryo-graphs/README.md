# Embryo Graphs

## Install requirements
Use `requirements.txt` with `-f https://data.pyg.org/whl/torch-1.13.1+cu117.html`

## How to run
### You will need
1. The outputs from the segmentation algorithm which should be stored in `data/unity/t8` as `.txt` files and look something like...
```
183 202 4 0.9999244213104248 [(189, 242), (194, 241), (201, 239), (207, 237), (209, 236), (212, 234), (218, 228), (223, 218), (224, 215), (224, 199), (222, 193), (219, 187), (215, 180), (213, 177), (210, 174), (206, 171), (201, 168), (194, 164), (190, 163), (173, 163), (161, 169), (158, 171), (148, 181), (145, 185), (143, 189), (142, 192), (142, 210), (143, 222), (145, 226), (150, 231), (155, 235), (157, 236), (166, 240), (169, 241), (179, 242)]
142 178 8 0.9203842282295227 [(164, 229), (167, 228), (170, 226), (174, 223), (176, 221), (178, 218), (181, 213), (183, 209), (185, 202), (189, 179), (190, 171), (190, 165), (188, 160), (180, 144), (174, 139), (172, 138), (155, 130), (152, 129), (148, 128), (139, 128), (134, 129), (129, 131), (122, 136), (118, 139), (117, 140), (105, 155), (96, 167), (94, 170), (93, 174), (93, 187), (94, 190), (98, 198), (107, 210), (108, 211), (122, 222), (129, 226), (133, 228), (136, 229)]
...
```
2. The outputs from Unity which should be a `.csv` file containing adjacency matrices which should look something like this...
```
id,adjacency
1,  0 0.6172954 0.7363851 1.26234 0 0 0 0/ 0.6172954 0 2.34963 0 0 0 2.098766 2.83061/ 0.7363851 2.34963 0 0 0.6098611 0 0.3850897 0/ 1.26234 0 0 0 0 0.1334248 0 0/ 0 0 0.6098611 0 0 1.017972 1.86063 0/ 0 0 0 0.1334248 1.017972 0 1.223803 0.4090188/ 0 2.098766 0.3850897 0 1.86063 1.223803 0 0/ 0 2.83061 0 0 0 0.4090188 0 0/
...
```
3. A `.csv` file of clinical output data with path `data/clinical.csv` and the headings...
```
id,t8_path,grade,pgt,outcome,simplified_grade,simplified_outcome,tPNf,t2,t3,t4,t5,t6,t7,t8
```
### Steps to run
1. Run `datasets/create_graph_dataset.ipynb` to generate a CSV of adjacency matrices and bounding boxes.
2. Run `generate_bottlenecks.py` to generate bottlenecks for CNN training
3. Train and evaluate the baseline CNN with `train_baseline.py` and the GNN with `train.py`.
4. Train and evaluate the other baselines by running `other_baselines.ipynb`.
