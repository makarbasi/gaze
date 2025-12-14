# Gaze Estimation Android Demo

## Introduction

This is a real-time gaze estimation demo running on Qualcomm GPU. It supports multi-person high-quality face detection, facial landmark detection, headpose estimation and gaze estimation, and still runs at real-time when all functionalities are enabled.

![image1.png](https://s2.loli.net/2022/10/17/3Yva8x6DTm9N5Go.png)

Video demo link: https://drive.google.com/file/d/1z4He0zVHlPRfaDlUv0mS623lYrsOppJv/view?usp=sharing

## Usage

**NOTICE: This demo can only run on Qualcomm GPU.**

Download model files from: [Google Drive](https://drive.google.com/drive/folders/1jEL7o55bmsSCmJX9ihByaHyBNZYX0CF7?usp=sharing). There should be three .dlc files.

Push all three .dlc model files to device folder `/data/local/tmp/`, etc. `adb push face_detection.dlc /data/local/tmp/`.   

Compile the project or simply install the .apk, and enjoy the demo!
