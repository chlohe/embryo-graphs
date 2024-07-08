# Cell Segmentation 3D

## Dataset format

The scripts expect a dataset organised into directories in the format
```
datasets
├── train
    ├── stacks
    	├── clinic 1
      	    ├── embryo 1        (an embryo id e.g. "1234")
    	    	├── timestamp 1 (e.g. "1.4") 
    	    	    ├── F0.jpg  (focal planes)
    	    	    ├── F15.jpg
    	    	    ├── F-15.jpg
    	    	    ...
    	        ├── timestamp 2
    	        ├── timestamp 3
    	        ...
 	    ├── embryo 2
 	    ├── embryo 3
 	    ...
        ├── clinic 2
        ├── clinic 3
    ├── seg
        ├── clinic 1
            ├── annotation.json  (in the format EMBRYO-ID_TIMESTAMP.json. For example, if the ID is "1234" and the timestamp is "10.0" then the file is "1234_10.0.json")
    ...
├── val
├── test
```
The JSONs were created using our modified version of VIA.

## How to run
- To train, run `train_maskrcnn2d.py`.
- To do inference, run `predict_frame.py` for a single frame, or `predict_timelapse.py` for an entire timelapse. The output of these scripts are serialised predictions in the form of `.txt` files. Each line of these files represent a different cell in the format
```
center_x center_y center_z confidence [(outline_point_1_x, outline_point_1_y), (outline_point_2_x, outline_point_2_y)... ]
```
and can be imported and viewed using the Unity viewer.
- To evaluate different NMS algorithms, run `notebooks/miccai_eval.ipynb`.
