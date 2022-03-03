# EmotionsInSSW

DATA AND PROGRAMS

- Figures 1 (except 1A), 2 and 4: Use Matlab notebook Figures.mlx, which reads data from existing files
- Figures 5 and 6: Use Matlab notebook Figures56.mlx; data in the notebook are to be copy-pasted from xlsx spreadsheets “CograngerHR_aggregate1to24s**.xlsx, where ** = heart rate response delay (0, 5, …, 30s). See notebook on how to copy from xlsx (theh xlsx files contain Granger causality tests and p values for all (valid) subjects and sessions, assuming heart rate response delay of ** seconds; only test statistics and p values go into the Matlab book). Other data pasted in the notebook: (i) earnings per subject (only valid subject-sessions for which valid heart rate records exist); (ii) list of session numbers for MMIP-calibrated sessions.
  o	Remark: the data in “CograngerHR_aggregate1to24s**.xlsx are output from the method “cogranger” function in the python program
  o	FWE correction added manually to panel titles, based on 6 simultaneous tests
  o	Stars (indicating significance of slope coefficients) added manually
- Figure 3: Python program
