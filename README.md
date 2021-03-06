# CS289 Group Project S - Final Deliverables

### Group Member Info

| Name       | SID |
| ----------- | ----------- |
| Jaewon Lee | 3035373160 |
| Ioannis Vouvakis Manousakis | 3035352698 |

For the early deadline project, we had created a large dataset of synthetic astronomical observations of the planets of the solar system, the Sun, and the Moon. Among the recorded quantities were the right ascension, and the declination of the considered celestial bodies. The data were collected for a period starting from -1500 BC and ending in 2020 AD. Time was expressed in Julian date (JD), and at least one measurement per celestial body was taken daily. More details about this can be found [here](https://github.com/ioannis-vm/CS289_2020_ProjectS_TeamJupyter).

Our goal for the final project was to create models that can accurately predict RA and Dec given a specific time expressed in JD.

Our results are summarized in our writeup, which is available here. The following instructions describe the necessary steps you need to take if you would like to reproduce our results.

### Reproducibility of our results

The repository has the following file hierarchy:

```
[Folder] Mars_fourier_lasso
	[Folder] Data
		Mars_test.df
		Mars_training.df
	Mars_Fourier_featurization.ipynb
[Folder] Geocentric Model Param Recovery
	[Folder] Mars
		Mars_Train_Test.py
		Mars.py
[Folder] Heliocentric Model Param Recovery
	[Folder] 1. Earth Model
		Earth_LongRange.py
	[Folder] 2. Planets Model
		Mars_Train_Test.py
		Planets.py
[Folder] Auxiliary 
	Filter_Data.ipynb
	OrbitPlots.ipynb
```

Due to the large size of the required input data, they have not been hosted on Github. You can find them [here](https://drive.google.com/drive/folders/1UF7OOssdIMld-oOjBigWpc_-oSErPLyi?usp=sharing).

After downloading the input data folders, they need to be placed in specific locations in the file tree, and the auxiliary jupyter notebooks need to be placed in the Heliocentric Data folder. The final file tree should be the following:

```
[Folder] Mars_fourier_lasso
	[Folder] Data
		Mars_test.df
		Mars_training.df
	Mars_Fourier_featurization.ipynb
[Folder] Geocentric Model Param Recovery
	[Folder] Mars
		Mars_Train_Test.py
		Mars.py
	*[Folder] Geo_Data* <-- here!
[Folder] Heliocentric Model Param Recovery
	[Folder] 1. Earth Model
		Earth_LongRange.py
	[Folder] 2. Planets Model
		Mars_Train_Test.py
		Planets.py
	*[Folder] Heliocentric_Data* <-- and here!
		Filter_Data.ipynb
		OrbitPlots.ipynb
```



