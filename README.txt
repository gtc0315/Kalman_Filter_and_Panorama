This project is written by Tianchang Gu for the course project Unscented Kalman Filter and Panorama.

###steps to run UKF###
1. modify line 9-12 in main.py 
	change dataset to the correct number
	change training to False if there is no vicon data, default=False
2. change direction to current path and run
	python main.py
3. close the figure to end the program

###steps to run panorama###
1. modify line 10-14 in panorama.py 
	change dataset to the correct number
	change training to False if there is no vicon data, default=False
	change use_vic to True to generate panorama with vicon data, default=False
	change skip to a big value (like 9) to speed up panorama, default=0(use all images)
2. change direction to current path and run
	python panorama.py
3. close the figure to end the program


###special library requirements###
numpy, pickle, PIL, matplotlib, transforms3d
python2.7

###folder contents###
There are three library for the cleaness of the code.
	BasicQuaternionFunctions.py
	ukf_lib.py
	panorama_lib.py
	draw.py
	test.py

There are two main scripts for ukf and panorama.
	main.py
	panorama.py

There is one pickle file for storing euler orientation from ukf.
	ukf_data_1

There is a cache image for storing image
	img_panorama_1.jpeg

