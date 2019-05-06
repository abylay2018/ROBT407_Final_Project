# ROBT407_Final_Project
Airbus Ship Detection Challenge

This project is for Kaggle competiton [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection).

It can help you quickly get a **baseline solution**, which is not bad.

![infer_example](https://github.com/abylay2018/ROBT407_Final_Project/blob/master/images/infer_example.jpg)



## File strcture

    airbus                         
    data balancing                
    turn rle to matrices of image size
    |
    Data augmentation   
    |
    model and trainning log
    log and visualization script
    configure file and .pkl (.pkl not be uploaded)
    |
    generate your submission
    reference .csv file





## Steps

#### 1. Generate image labels from rle 

Run functions of transforms in airbus.py

![dataset annotation](https://github.com/pascal1129/kaggle_airbus_ship_detection/blob/master/images/annotation.png)



#### 2. Get augmented data

Various methods of augmentation (rotating, flipping, shifying, etc.) were used

![dataset annotation](https://github.com/pascal1129/kaggle_airbus_ship_detection/blob/master/444.png)

  
#### 2. Constructing a model and training

U-NET model for segmenting objects on image was used to detect ships.

![dataset annotation](https://github.com/pascal1129/kaggle_airbus_ship_detection/blob/master/333.jpg)




#### 6. Get the final submission

Run `./3_submit/get_final_csv.py`.
