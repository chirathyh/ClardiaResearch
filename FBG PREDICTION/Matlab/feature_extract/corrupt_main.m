% Chirath Hettiarachchi 
% March 2020
% Automatic Feature Extraction based on 

clear all;
close all;

addpath(genpath('lib/'));

% select dataset
DATASET = 2;
if DATASET == 1
    subject = csvread('data_handler/data1.csv',1,0);
    file_prefix = 'data1/AAA';
else
    subject = csvread('data_handler/data2.csv',1,0);
    file_prefix = 'data2/X_';
end

[m,n] = size(subject);
feature_vector = [];

SKEWNESS_THRESHOLD = 0.10;

features = [];
reject_count = 0;

for patient = 1:1:m
    % load the signal
    subject_ID = int2str(subject(patient,1));
    name = strcat('input/ppg/',file_prefix, subject_ID, '.csv');
    wave = csvread(name,1,0)';
    
    % select the 1 min interval in the middle.
    [start_point, end_point] = get_bestSignal(wave,1);
    select_perSubject = 0;
    
    %%%%%%%%%%%%%%%%%%
    figure
    subplot(2,1,1)
    plot(wave(1,start_point:end_point));
    
    noise_id = rem(subject(patient,1),12);
    [n,m, d, M, signal_snr, dynamic_var] = get_savitzkyGolayNoise(noise_id + 1);
    corrupt_m = [zeros(1,start_point-1) dynamic_var zeros(1,size(wave,2)-end_point+1)];
    corrupt_m = corrupt_m(1,1:size(wave,2));
    wave = wave + corrupt_m ;
    
    wave = awgn(wave,10,'measured');
    %alpha = other_noise(1,start_point+700:start_point+1000);
    %beta  = other_noise(1,start_point+1800:start_point+2150);
    %gamma = other_noise(1,start_point+2650:start_point+3000);
    %main_noise = [zeros(1,start_point+700) alpha zeros(1,800) beta zeros(1,500) gamma zeros(1,size(wave,2)-3000)];
    %main_noise = main_noise(1,1:size(wave,2));
    %wave = wave + main_noise;
    
    subplot(2,1,2)
    plot(wave(1,start_point:end_point));
    %%%%%%%%%%%%%%%%%%
    
    PPGsignal = preprocessSignal(wave);

    loop_number = 0;
    for segment = start_point:75:end_point-149
        loop_number = loop_number + 1;
        signal_segment = PPGsignal(1,segment:segment+149);
        sample_skew = skewness(signal_segment);
        
        if sample_skew > SKEWNESS_THRESHOLD
            %calculate the main features. 
            [extracted_features,status] = feature_extractv3(signal_segment, subject_ID);
            if status == 1
                reject_count = reject_count + 1;
            else
                
                % fix overlapping subject ID between datasets.
                if DATASET == 1
                    patient_id = subject(patient,1);
                else 
                    patient_id = subject(patient,1) + 30;
                end
                
                full_features = [patient_id, subject(patient,2:end), loop_number, extracted_features];
                features = [features; full_features];
                select_perSubject = select_perSubject + 1;
            end 
        end
        
    end 
    fprintf('\n Subject = %s, Count = %d',subject_ID, select_perSubject)
end
fprintf('\n Total Rejected Signals = %d',reject_count)
csvwrite('corrupt_output/corrupt_features_dataset2.csv',features);


