
Note: 

The signals are sampled in 60Hz.

In Preprocessing a 8th Order chebyshev II filter is implemented in reality.
(Correction required to Liang 2018. 

fl = 0.5Hz & fH = 10Hz. 
(Assume HR normally 50-200bpm. ~(0.83 - 3.33 Hz))
Important => fH set as 10Hz otherwise APG features are lost. 

Amplitude normalized using Minimax normalization applied to the signals. 
For the comparability of the ratio features we calculate. 

We want to extract multiple signal portions. 
Ideally focus on a 1 minute interval at the middle of the signal. 

2.1s --> 60 * 2.1 = 126 ~150 sample window is obtained for the analysis. 

Tunable Parameters.





