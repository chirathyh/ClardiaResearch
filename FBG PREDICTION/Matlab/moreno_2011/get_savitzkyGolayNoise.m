function [n,m, d, M, signal_snr, dynamic_var] = get_savitzkyGolayNoise(id)
    noisy_signals = ["TROIKA_Training_data/DATA_01_TYPE01.mat", "TROIKA_Training_data/DATA_02_TYPE02.mat",...
    "TROIKA_Training_data/DATA_03_TYPE02.mat", "TROIKA_Training_data/DATA_04_TYPE02.mat",...
    "TROIKA_Training_data/DATA_05_TYPE02.mat", "TROIKA_Training_data/DATA_06_TYPE02.mat",...
    "TROIKA_Training_data/DATA_07_TYPE02.mat", "TROIKA_Training_data/DATA_08_TYPE02.mat",...
    "TROIKA_Training_data/DATA_09_TYPE02.mat", "TROIKA_Training_data/DATA_10_TYPE02.mat",...
    "TROIKA_Training_data/DATA_11_TYPE02.mat", "TROIKA_Training_data/DATA_12_TYPE02.mat"...
    ]; 

    S = load(noisy_signals(id));
    PPG = S.sig(2,:);
    
    PPG = PPG(1,1:7500);
    %PPG = PPG(1,7500:15000); %second minute.
    
    PPG = downsample(PPG,2);
    PPG = PPG(1,1:3600);
    
    frame_length = 75;
    
    savitzky_golay_filtered = sgolayfilt(PPG,4,frame_length);
    n = PPG - savitzky_golay_filtered;
    
    m1 = mean(PPG(1,1:1800));
    m2 = mean(PPG(1,1801:3600));
    m  = [ones(1,1800)*m1 ones(1,1800)*m2];
    
    bp = 1:150:3450;
    detrended_signal = detrend(PPG,1,bp);
    d = PPG - detrended_signal; 
    
    M = movmean(PPG,frame_length);
    
    signal_snr = snr(savitzky_golay_filtered,PPG-savitzky_golay_filtered);
    
    %%%%%%%%% Dynamic Variance Noise Model%%%%%%%%%%%%
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
    var_n2 = sqrt(movvar(n2,70)) * F;
    base_n1 = exp_n1 + var_n1;
    base_n2 = exp_n2 + var_n2;
    base_noise = [base_n1 base_n2];
    y = bandpass(base_noise,[2 26],fs,'ImpulseResponse','iir');
    dynamic_var = y + d;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
end
