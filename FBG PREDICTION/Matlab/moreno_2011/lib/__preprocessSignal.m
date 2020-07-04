function [cheby_filtered_ppg] = preprocessSignal(raw_PPG)

    %%% Adding a 4th Order Chebyshev II Filter - Liang2018 %%%
    % Filter Parameters
    order = 4;
    fl = 0.5;
    fH = 10;
    Fs = 125;
    Fn = Fs /2 ;
    [A,B,C,D] = cheby2(order,20,[fl fH]/Fn);
    [filter_SOS,g] = ss2sos(A,B,C,D);
    cheby_filtered_ppg = filtfilt(filter_SOS,g,raw_PPG);

    % Normalizing the Figures. 
    ppgMax = max(cheby_filtered_ppg);
    ppgMin = min(cheby_filtered_ppg);
    cheby_filtered_ppg = (cheby_filtered_ppg - ppgMin) / (ppgMax - ppgMin);
    
end