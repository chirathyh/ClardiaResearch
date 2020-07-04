function [cheby_filtered_ppg] = hr_preprocessSignal(raw_PPG)
    
    order = 6;
    fl = 0.01;
    fH = 4;
    Fs = 62;
    Fn = Fs /2 ;
    [A,B,C,D] = cheby1(order/2,20,[fl fH]/Fn);
    [filter_SOS,g] = ss2sos(A,B,C,D);
    cheby_filtered_ppg = filtfilt(filter_SOS,g,raw_PPG);
    
end