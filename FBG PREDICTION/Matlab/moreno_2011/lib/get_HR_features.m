function F_HR = get_HR_features(filtered_PPG)
    
    %removing the DC offset
    signal_mean = mean(filtered_PPG);
    filtered_PPG = filtered_PPG - signal_mean;
    
    %figure;
    %plot(filtered_PPG);
    
    zero_crossings = [];

    for i =2:length(filtered_PPG)
        if filtered_PPG(i-1) >= 0 && filtered_PPG(i) <= 0
            zero_crossings = [zero_crossings i];
        elseif filtered_PPG(i-1) <= 0 && filtered_PPG(i) >= 0
            zero_crossings = [zero_crossings i];
        else
        end
    end
    
    shifted_zerocrossing = zero_crossings(1,2:end);
    normal_zerocrossing  = zero_crossings(1,1:end-1);
    hrv = shifted_zerocrossing - normal_zerocrossing;
    
%     figure;
%     plot(hrv);
%     hold on;

    f_HRmean = mean(hrv);
    f_HRvar  = var(hrv);
    f_HRskew = skewness(hrv);
    f_HRiqr  = iqr(hrv);
     
    F_HR = [f_HRmean f_HRvar f_HRskew f_HRiqr];
   
end