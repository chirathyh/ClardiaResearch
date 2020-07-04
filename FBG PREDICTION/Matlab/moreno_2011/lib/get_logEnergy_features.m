function F_LOGENERGY = get_logEnergy_features(frame_array)
    n_frame_log_energy = [];
    
    for n =1:length(frame_array)
        n_frame = frame_array{n}';   
        le = log(sum(n_frame.^2)); %calc log energy
        n_frame_log_energy = [n_frame_log_energy le];
    end
    
    f_LE_var = var(n_frame_log_energy);
    f_LE_iqr = iqr(n_frame_log_energy);
    
    n_frame_log_energy = n_frame_log_energy - mean(n_frame_log_energy);
    
    sys = ar(n_frame_log_energy,5,'yw');
    f_AR_LE = sys.A;
    f_AR_LE = f_AR_LE(1,2:end);
    
    F_LOGENERGY = [f_AR_LE f_LE_var f_LE_iqr];
    
end