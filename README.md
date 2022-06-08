# EmotionsInSSW

DATA AND PROGRAMS

- Figures 1 (except 1A), 2 and 4: Use Matlab notebook Figures.mlx, which reads data from existing files
- Figures 5 and 6: Use Matlab notebook Figures5DETAILED.mlx and Figures6DETAILED.mlx; data in the notebook are to be copy-pasted from xlsx spreadsheets “CograngerHR_aggregate1to24s**.xlsx, where ** = heart rate response delay (0, 5, …, 30s). See notebook on how to copy from xlsx (theh xlsx files contain Granger causality tests and p values for all (valid) subjects and sessions, assuming heart rate response delay of ** seconds; only test statistics and p values go into the Matlab book). Other data pasted in the notebook: (i) earnings per subject (only valid subject-sessions for which valid heart rate records exist); (ii) list of session numbers for MMIP-calibrated sessions. THESE NOTEBOOKS ALSO CREATE TABLES IN 3RD SECTION OF ELECTRONIC COMPANION (see EC2Fig5.docx and EC2Fig6.docx, which contain thhe output of the notebooks).
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

PYTHON PROGRAM INPUT AND OUTPUT FILES:

Input data:
- 'Official ' + str(s) + '.csv' (markets output data)
- 'Div_' + str(s) + '_official.csv' (dividend payments)
- 'session' + str(s) + '_sub' + str(p) + '.csv' (sessions 1-12) and 'sessionEX' + str(ss) + 'M' + str(p) + '.csv' (sessions 21-24) (RAW SCR data, per subject/session)
- 'S' + str(s) + '_R' + str(p) + '.csv' (local heart rate estimates, per second, processed from raw 512Hz ECG as explained in the paper)

Output data:
- 'saveit_official ' + str(s) + ' v2.csv' (markets, variables of interest, dividends
- 'save_earnings'+str(i+21)+'.csv' (earnings, volume, final holdings0
- 'GSR in sec session ' + str(s) + str(p) + '.csv' (SCR data per second)
- 'coint' + str(s) + '_s' + str(p) + '.csv' (Has all the data to do cointegration analysis between SCR and variables of interest)
- "CointSCR_AssetValueHolding.csv", "CointSCR_CashHolding.csv", "CointSCR_BASpread.csv" (cointegration data used in cointegration tests SCR-AssetValueHolding, SCR-CashHHolding, SCRR-BASpread, just to check whether data are right (checked through, example, plotting)
- "CointSCR_aggregate.csv" collects all the data for cointegration tests
- "P-value_aggregate.csv" (P values of SCR-"variable of interest" cointegration test, used to relate to earnings; used to produce Fig 3)
- 'scrvarForPlot.csv' (data for plot of cointegration with variable of interest for a particular pair (participant, session)
- 'Session '+str(s)+' P'+str(p)+'.csv' = 'S' + str(s) + '_R' + str(p) + '.csv'  with correct time alignment for markets data
- 'HRcoint' + str(s) + '_s' + str(p) + '.csv' (Like 'coint' + str(s) + '_s' + str(p) + '.csv', but for HR data; Has all the data to do VAR+Granger Causality analysis between SCR and variables of interest)
- "CograngerHR_aggregate1to24s10.csv" (Exemplar output file of Granger Causality tests for all sessions assuming 10s HR response delay (copied into matlab notebook to produce Figures 5&6)
