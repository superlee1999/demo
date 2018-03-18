# demo (Windwos only)
Instructions:
What does this dome do?
	- run prebuit APTTestMain.exe
	- the programm will pop up 3 windows
		- FG: motion segmentation view
		- Background: background statistic model view (auto adapted/updated overtime)
		- Tracking: object tracking with trajectories plot

	- APTTestMain.exe (simply run the exe with open your PC webcam and capture live video)
	- APTTestMain.exe <input_video> (run the exe again a local video file, e.g xxx.avi xxx.mp4)
	- NOTICE:
		- before running the exe, please set environment variable APT_CONFIG_DIR to Configuration folder (containing cctv_config.json)
			set APT_CONFIG_DIR=PATH_TO_Configuration

How to compile:
1. prerequisite
	- VS2015 with VC14 compiler
	- boost library >= 1.61.0 (static build preferred available from https://sourceforge.net/projects/boost/files/boost/1.63.0/boost_1_63_0.zip/download) 
	- opencv library >= 3.4.0 (available from https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.1/opencv-3.4.1-vc14_vc15.exe/download)
2. adapt project props file
	- Projects\VC14.0\Common.props (adapt the variable path defined in this file to your local directory)
3. compile x64 bit version and set TestMain as startup programm

