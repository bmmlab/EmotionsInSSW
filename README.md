# EmotionsInSSW

DATA AND PROGRAMS

- Figures 1 (except 1A), 2 and 4: Use Matlab notebook Figures.mlx, which reads data from existing files
- Figures 5 and 6: Use Matlab notebook Figures56.mlx; data in the notebook are to be copy-pasted from xlsx spreadsheets “CograngerHR_aggregate1to24s**.xlsx, where ** = heart rate response delay (0, 5, …, 30s). See notebook on how to copy from xlsx (theh xlsx files contain Granger causality tests and p values for all (valid) subjects and sessions, assuming heart rate response delay of ** seconds; only test statistics and p values go into the Matlab book). Other data pasted in the notebook: (i) earnings per subject (only valid subject-sessions for which valid heart rate records exist); (ii) list of session numbers for MMIP-calibrated sessions.
  o	Remark: the data in “CograngerHR_aggregate1to24s**.xlsx are output from the method “cogranger” function in the python program
  o	FWE correction added manually to panel titles, based on 6 simultaneous tests
  o	Stars (indicating significance of slope coefficients) added manually
- Figure 3: (Left and Right Panels): Use Matlab notebook Fig3.mlx, where data are copied from P-value_aggregate.csv.
- PYTHON: DataProcessing.py. Takes SCR data, HR (Heart Rate) data and markets data, processes them, merges them in appropriate ways – thereby generating the input to the Matlab notebooks above, among others), and does analysis of cointegration (SCR) and VAR/Granger causality (HR)
  o	Part 1 reads markets (Flex-E-Markets) output data, generates variables of interest (change in holdings & cash, mispricing, bid-ask spread, adds dividends from a different file, 
    	Output files are created along the way, read to add variables, saved again, read again… (legacy of building and checking the program in pieces)
  o	Part 2 reads earnings, volume, final holdings inline, and checks against data from Flex-e-Markets
    	PLOTS:
      •	Earnings (re-produced in Matlab notebook)
      •	Volume (not in paper)
      •	Final holdings (not in paper)
  o	Part 3 reads SCR data, merges with markets data, and analyzes using co-integration
  o	Part 4 reads HR, aligns with markets data using time_adjust to allow for putative HR response delay and analyzes using VAR and Granger causality
