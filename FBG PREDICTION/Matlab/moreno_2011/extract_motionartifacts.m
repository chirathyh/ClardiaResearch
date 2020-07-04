clear all;
close all;

noisy_signals = ["TROIKA_Training_data/DATA_01_TYPE01.mat", "TROIKA_Training_data/DATA_02_TYPE02.mat",...
    "TROIKA_Training_data/DATA_03_TYPE02.mat", "TROIKA_Training_data/DATA_04_TYPE02.mat",...
    "TROIKA_Training_data/DATA_05_TYPE02.mat", "TROIKA_Training_data/DATA_06_TYPE02.mat",...
    "TROIKA_Training_data/DATA_07_TYPE02.mat", "TROIKA_Training_data/DATA_08_TYPE02.mat",...
    "TROIKA_Training_data/DATA_09_TYPE02.mat", "TROIKA_Training_data/DATA_10_TYPE02.mat",...
    "TROIKA_Training_data/DATA_11_TYPE02.mat", "TROIKA_Training_data/DATA_12_TYPE02.mat"...
    ]; 

snr_arr = [];
for x = 1:1:12
    
    disp(noisy_signals(x));
    S = load(noisy_signals(x));
    PPG = S.sig(2,:);
    
    PPG = PPG(1,1:7500);
    
    PPG = downsample(PPG,2);
    PPG = PPG(1,1:3600);
    
    figure
    subplot(4,1,1)
    plot(PPG);
    
    frame_length = 75;
    savitzky_golay_filtered = sgolayfilt(PPG,4,frame_length);
    n = PPG - savitzky_golay_filtered;
    
    %m1 = mean(PPG(1,1:1800));
    %m2 = mean(PPG(1,1801:3600));
    %m  = [ones(1,1800)*m1 ones(1,1800)*m2];
    %M = movmean(PPG,frame_length);
    r = snr(savitzky_golay_filtered,PPG-savitzky_golay_filtered);
    snr_arr = [snr_arr r];
    
    bp = 1:150:3450;
    detrended_signal = detrend(PPG,1,bp);
    d = PPG - detrended_signal; 
    
   
    PPG_1 = PPG(1,1:1800);
    PPG_2 = PPG(1,1801:3600);
    F = -0.53;
    fs = 62.5;
    sg_fil1 = sgolayfilt(PPG_1, 4, frame_length);
    n1 = PPG_1 - sg_fil1;
    sg_fil2 = sgolayfilt(PPG_2, 4, frame_length);
    n2 = PPG_2 - sg_fil2;
    exp_n1 = movmean(n1,45);
    var_n1 = sqrt(movvar(n1,45)) * F;
    exp_n2 = movmean(n2,70);
    var_n2 = sqrt(movvar(n2,45)) * F;
    base_n1 = exp_n1 + var_n1;
    base_n2 = exp_n2 + var_n2;
    base_noise = [base_n1 base_n2];
    y = bandpass(base_noise,[2 26],fs,'ImpulseResponse','iir');
    dynamic_var = y + d;
    
    subplot(4,1,2)
    plot(base_noise);      
    subplot(4,1,3)
    plot(d);  
    subplot(4,1,4)
    plot(y+d);

end
disp(mean(snr_arr));