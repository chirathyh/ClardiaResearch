function f_FRAME_KTE = get_frameKTE_features(frame_array)
% calculate the frame kte based features. 
    n_kte_mean = [];
    n_kte_var  = [];
    n_kte_skew = [];
    n_kte_iqr  = [];
    frame_kte  = [];
    for n =1:length(frame_array)
        n_frame = frame_array{n}';
        for i=2:length(n_frame)-1
            frame_kte(i) = n_frame(i)^2 - n_frame(i-1) * n_frame(i+1);
        end
        % append the features to arrays. 
        n_kte_mean = [n_kte_mean mean(frame_kte)];
        n_kte_var  = [n_kte_var var(frame_kte)];
        n_kte_skew = [n_kte_skew skewness(frame_kte)];
        n_kte_iqr  = [n_kte_iqr iqr(frame_kte)]; 
    end
    f_FRAME_KTEmean = mean(n_kte_mean);
    f_FRAME_KTEvar  = mean(n_kte_var);
    f_FRAME_KTEskew = mean(n_kte_skew);
    f_FRAME_KTEiqr  = mean(n_kte_iqr);
    f_FRAME_KTE = [f_FRAME_KTEmean, f_FRAME_KTEvar, f_FRAME_KTEskew, f_FRAME_KTEiqr];
end