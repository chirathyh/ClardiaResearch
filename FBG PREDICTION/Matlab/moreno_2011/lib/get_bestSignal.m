function [start_index,end_index] = get_bestSignal(array, type)
    % The function calculated the 1 minute signal in between recorded
    % signal
    % Type 1 = ppg, 2 = spo2.
    
    if type == 1
        %sampled at 60Hz 3600 samples / min
        tot_sample = 3600;
        ppg_len = length(array);
   
        if ppg_len <= 3600
            start_index = 1;
            end_index = ppg_len;
        else
            start_index = floor((ppg_len / 2)) - (tot_sample / 2);
            end_index   = floor((ppg_len / 2)) + (tot_sample / 2);
        end
        
    elseif type == 2
        %sampled at 1Hz 60 samples / min
        tot_sample = 60; %since the spo2 take some time to settle
        spo2_len = length(array);
        %disp(spo2_len);
        if spo2_len <= 80
            start_index = 20;
            end_index = spo2_len;
        else
            start_index = floor((spo2_len / 2)) - (tot_sample / 2);
            end_index   = floor((spo2_len / 2)) + (tot_sample / 2);
        end
        
    else
        disp('Warning: Type of signal not specified.')
    end
    
    
end