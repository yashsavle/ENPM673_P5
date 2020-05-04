# Visual Odometry

This Project Implements the concept of Visual Odometry. The program calculates the orientation and position of a robot by using camera parameters and video frames

## Dependencies

```bash
1. OpenCV(3.4.2) or earlier. You can use the latest version if SIFT is enabled.
2. Python 3.6 or higher
3. Scipy
```

## Usage
**p5.py** is the main program using userdefined functions
**predef.py** is using predefined functions
**compare.py** compare the two results
**video.py** is to make the video

Download the Dataset from [here](https://drive.google.com/drive/folders/1hAds4iwjSulc-3T88m9UDRsc6tBFih8a?usp=sharing)
Once downloaded extract the the zip file and place all the py in Oxford_dataset.
To run simply type following commands in terminal
```python
python p5.py
```
A folder will be created which will store the output plots for each frame. Also a text file is generated which will store the Camera Co-ordinates. Make sure that there are no previous text files present before running the program.
Once completed use compare.py and video.py to visualize the results.

## Link to Output
[link](https://drive.google.com/file/d/1slKDtfBjthnQRE8EN6aEy1KtiORKv4S4/view)
