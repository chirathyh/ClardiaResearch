function F_FRAME_SE = get_frameSE_features(frame_array)
    
    n_frame_SE = [];
    for n =1:length(frame_array)
        n_frame = frame_array{n}';
        se = pentropy(n_frame,62,'Instantaneous',false);
        n_frame_SE = [n_frame_SE se];
    end
    F_FRAME_SEmean = mean(n_frame_SE);
    F_FRAME_SEvar  = var(n_frame_SE);
    F_FRAME_SEskew = skewness(n_frame_SE);
    F_FRAME_SEiqr  = iqr(n_frame_SE);
    F_FRAME_SE = [F_FRAME_SEmean, F_FRAME_SEvar, F_FRAME_SEskew,F_FRAME_SEiqr]; 
end