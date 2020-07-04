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
    PPGsignal = preprocessSignal(wave);
    
    % select the 1 min interval in the middle.
    [start_point, end_point] = get_bestSignal(PPGsignal,1);
    select_perSubject = 0;

    loop_number = 0;
    for segment = start_point:75:end_point-149 %149
        loop_number = loop_number + 1;
        signal_segment = PPGsignal(1,segment:segment+149); %149
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
csvwrite('output/features_dataset2.csv',features);


