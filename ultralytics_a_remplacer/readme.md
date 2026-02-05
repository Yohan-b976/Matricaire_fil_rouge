In order to see the performance in counting at each epoch of training and on the validation set, some ultralytics files need to be replaced.
As of 05/02/2026, the computed counting metric is the ratio between the number of predicted boxes and the number of ground truth boxes.
As of 05/02/2026, the best model selected at the end of training IS NOT selected on the ratio, but on mAP50-95 as this is the default in YOLO.

Instructions for file replacement :
- Locate the ultralytics folder (if installed with miniconda, it should be in miniconda3/lib/python3.13/site-packages where python3.13 can be any version you're using)
- In ultralytics/utils, replace metrics.py and plotting.py
- In ultralytics/models/yolo/detect, replace val.py

For reference, I have ultralytics 8.4.6, python 3.13.9 and torch 2.9 (should still work on slightly different versions).