% PPG Signal Feature Extraction Script 
% Chirath Hettiarachci 
% 23rd November 2018

% 1 SubjectID
% 2 Gender
% 3 Age
% 4 Height
% 5 Weight
% 6 HR
% 7 BMI
% 8 SystolicBP
% 9 DiastolicBP
% 10 Hypertension
% 11 Diabetes
% 12 Signal 1 SQI
% 13 Signal 2 SQI
% 14 Signal 3 SQI
% 15 Selected Signal

addpath(genpath('lib/'));

SIGNAL_PATH = '/Users/chirathhettiarachchi/tensorflow/datasets/clardia3/PPG/signals/';
subject = csvread('diabetesPatientInfo.csv',1,0);
%subject = csvread('HypertensionPatientInfo.csv',1,0);
subject1 = csvread('normalPatientInfo.csv',1,0);
[m,n] = size(subject);
[m1,n1] = size(subject1);

feature_vector = [];

for patient = 1:1:m
    subject_ID = int2str(subject(patient,1));
    selected = int2str(subject(patient,15));
    tempSubject = strcat(SIGNAL_PATH,subject_ID,'_',selected,'.txt'); 
    
    raw_PPG = load(tempSubject);
    PPGsignal = preprocessSignal(raw_PPG); 
    
    % find all the maximum, minimum points
    [high_peakValues, high_indexes] = findpeaks(PPGsignal);
    negPPGsignal = -1 * PPGsignal;
    [neg_peakValues, neg_indexes] = findpeaks(negPPGsignal); 
    
    % PPG signal peak & Lowest point
    peak_array = [];
    low_array = [];   
    
    %figure
    %plot(PPGsignal);
    %hold on;
    
    % Beat Seperation
    % Alternative => hr = floor((1000 * 60) / subject(patient,6)) ;
    %disp(hr);
    % 1s - 1000 datapoints 
    % IF 60bpm -> 1ps -> 1000 window (One peak to peak data points) 
    % IF 120bpm -> 2ps -> 500  window
    % 52bpm - 106bpm -> 564 window
    
    hr = 525;
    for x = 1:hr:length(PPGsignal)
        [identifiedMax,pos] = max(PPGsignal(x:1:x+hr-1));
        [identifiedMin,minpos] = min(PPGsignal(x:1:x+hr-1));
        peakPosition = pos + x;
        lowPosition = minpos + x;
        if sum(ismember(high_indexes,peakPosition - 1)) == 1
            peak_array = [peak_array peakPosition];
            %plot(peakPosition, PPGsignal(peakPosition),'o','MarkerSize',12)
            %hold on;
        end
        if sum(ismember(neg_indexes,lowPosition - 1)) == 1
            low_array = [low_array lowPosition];
            %plot(lowPosition, PPGsignal(lowPosition),'X','MarkerSize',12)
            %hold on;
            
        end
    end
    
    % Necessary Alterations for the low_array, If two minimums are near 
    % Note : Selecting which points should be deleted. 
    delTrough = [];
    for z = 2:1:length(low_array)
        first = PPGsignal(low_array(z-1));
        second = PPGsignal(low_array(z));
        if low_array(z) - low_array(z-1) < 525
            if first < second
                delTrough = [delTrough z];
            else
                delTrough = [delTrough z-1];
            end
        end
    end 
    
  % Deleting unwanted detected lows.
    for y = length(delTrough):-1:1
        low_array(delTrough(y)) = [];
    end
    
    % Adjustemnt for the Peak Array
    delPeak = [];
    for z = 2:1:length(peak_array)
        first = PPGsignal(peak_array(z-1));
        second = PPGsignal(peak_array(z));
        if peak_array(z) - peak_array(z-1) < 525
            if first < second
                delPeak = [delPeak z-1];
            else
                delPeak = [delPeak z];
            end
        end
    end
    
    for y = length(delPeak):-1:1
        peak_array(delPeak(y)) = [];
    end
    
    %FINAL CUTOFF POINTS ACQUIRED - Plotting
    for point = 1:length(low_array)
        %line([low_array(point) low_array(point)],[min(PPGsignal) max(PPGsignal)],'Color','red','LineStyle','--');
        %line([peak_array(point) peak_array(point)],[min(PPGsignal) max(PPGsignal)],'Color','blue','LineStyle','--');
    end
    
    
    % PPG Features 
    
    % identifiedPoints - Start with a trough , end with peak
    allidentifiedPoints = [low_array peak_array];
    allidentifiedPoints = sort(allidentifiedPoints);
    low_start = low_array(1);
    low_end = low_array(end);
    for x = 1:1:length(allidentifiedPoints)
        if allidentifiedPoints(x) == low_start
            startIndex = x;
        end
        if allidentifiedPoints(x) == low_end
            endIndex = x;
        end
    end
    identifiedPoints = allidentifiedPoints(startIndex:endIndex);
    for x = 1:1:length(identifiedPoints)
        %plot(identifiedPoints(x), PPGsignal(identifiedPoints(x)),'o','MarkerSize',12)
        %hold on;
    end
    %disp(identifiedPoints);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % peak2peakInterval (p - P)
    % pulseInterval (T - T) 
    % Rise Time (T to P, peak - trough) 
    % Amplitude
    
    beats = ( length(identifiedPoints) - 1 ) / 2 ;
    % calculated inter beat
    p2p = 0; pi = 0;
    % when number of beats > 2
    if beats == 1
        p2p = -1;
        pi = -1;
    elseif beats == 2
        p2p = p2p + identifiedPoints(4) - identifiedPoints(2);
        pi =  pi + identifiedPoints(3) - identifiedPoints(1);
    else 
        for b = 1:1:beats - 1
            p2p = p2p + identifiedPoints(2*b+2) - identifiedPoints(2*b);
            pi =  pi + identifiedPoints(2*b+1) - identifiedPoints(2*b-1);
        end
        p2p = p2p / (beats - 1);
        pi = pi / (beats - 1);
    end
    
    % calculated per beat
    rt = 0; amp = 0;
    if beats == 1
        rt = rt + identifiedPoints(2) - identifiedPoints(1);
        amp = amp + PPGsignal(identifiedPoints(2)) - PPGsignal(identifiedPoints(1));
    else
        for b = 1:1:beats
            rt =  rt + identifiedPoints(2*b) - identifiedPoints(2*b-1);
            amp = amp + PPGsignal(identifiedPoints(2*b)) - PPGsignal(identifiedPoints(2*b-1));
        end
        rt = rt / beats;
        amp = amp / beats;
    end
    
    extractedFeatures = [round(p2p) round(pi) round(rt) round(amp)];
    %disp(extractedFeatures);
    
    
    % Save the extracted features to a CSV 
    temp = [subject(patient,:) extractedFeatures PPGsignal(1:2100) ];
    feature_vector = [feature_vector ; temp];
%     % a wave
%     % b wave
%     % c
%     % d
%     % e - can also be used to detect the dicrotic notch
%     % f - to detect the dicrotic peak
%     
%     % Extracted Signal to obtain the VPG & APG
    %extractedPPGSegment = PPGsignal(low_start:low_end);
    extractedPPGSegment = PPGsignal(low_array(1):low_array(2));
    
    figure
    subplot(3,1,1);
    plot(extractedPPGSegment),grid;
    
    % 1st derivative
    x = 1:1:length(extractedPPGSegment);
    dy = diff(extractedPPGSegment)./diff(x);
    dy = preprocessSignal(dy); 
    subplot(3,1,2);
    plot(dy), grid;
    title('1st Derivative');
    
    % 2nd derivative
    x2 = 1:1:length(dy);
    dy2 = diff(dy)./diff(x2);
    dy2 = preprocessSignal(dy2); 
    subplot(3,1,3);
    plot(dy2), grid;
    hold on;
    title('2nd Derivative'); 
    
    dy2 = smooth(dy2);
%     
%     % extraction of a,b,c,d,e,f
%     [apg_high_peakValues, apg_high_indexes] = findpeaks(dy2);
%     %negdy2 = -1 * dy2;
%     inverteddy2 = 1.01 * max(dy2) - dy2;
%     [apg_neg_peakValues, apg_neg_indexes] = findpeaks(inverteddy2); 
%     
%     [a,apos] = max(dy2);
%     for x = 1:1:length(apg_high_indexes)
%         if apg_high_indexes(x) > apos
%            pointHigh = x;
%            break;
%         end
%     end
%     
%     c = apg_high_indexes(pointHigh);
%     e = apg_high_indexes(pointHigh + 1);
%     
%     for x = 1:1:length(apg_neg_indexes)
%         if apg_neg_indexes(x) > apos
%            pointLow = x;
%            break;
%         end
%     end
%     
%     b = apg_neg_indexes(pointLow);
%     d = apg_neg_indexes(pointLow + 1);
%     f = apg_neg_indexes(pointLow + 2);
%     
%         
%     plot(apos, a,'o','MarkerSize',10)
%     hold on;
%     plot(b, dy2(b),'x','MarkerSize',10)
%     hold on;
%     plot(c, dy2(c),'o','MarkerSize',10)
%     hold on;
%     plot(d, dy2(d),'x','MarkerSize',10)
%     hold on;
%     plot(e, dy2(e),'o','MarkerSize',10)
%     hold on;
%     plot(f, dy2(f),'x','MarkerSize',10)
% 
% 
%     arr = [apos,b,c,d,e,f];
%     if apos < b && b < c && c < d && d < e && e < f 
%         disp("true");
%     else 
%         disp("flase");
%     end
        
end


% for patient = 1:1:m1
%     subject_ID = int2str(subject1(patient,1));
%     selected = int2str(subject1(patient,15));
%     tempSubject = strcat(SIGNAL_PATH,subject_ID,'_',selected,'.txt'); 
%     
%     raw_PPG = load(tempSubject);
%     PPGsignal = preprocessSignal(raw_PPG); 
%     
%     % find all the maximum, minimum points
%     [high_peakValues, high_indexes] = findpeaks(PPGsignal);
%     negPPGsignal = -1 * PPGsignal;
%     [neg_peakValues, neg_indexes] = findpeaks(negPPGsignal); 
%     
%     % PPG signal peak & Lowest point
%     peak_array = [];
%     low_array = [];   
%     
%     %figure
%     %plot(PPGsignal);
%     %hold on;
%     
%     % Beat Seperation
%     % Alternative => hr = floor((1000 * 60) / subject(patient,6)) ;
%     %disp(hr);
%     % 1s - 1000 datapoints 
%     % IF 60bpm -> 1ps -> 1000 window (One peak to peak data points) 
%     % IF 120bpm -> 2ps -> 500  window
%     % 52bpm - 106bpm -> 564 window
%     
%     hr = 525;
%     for x = 1:hr:length(PPGsignal)
%         [identifiedMax,pos] = max(PPGsignal(x:1:x+hr-1));
%         [identifiedMin,minpos] = min(PPGsignal(x:1:x+hr-1));
%         peakPosition = pos + x;
%         lowPosition = minpos + x;
%         if sum(ismember(high_indexes,peakPosition - 1)) == 1
%             peak_array = [peak_array peakPosition];
%             %plot(peakPosition, PPGsignal(peakPosition),'o','MarkerSize',12)
%             %hold on;
%         end
%         if sum(ismember(neg_indexes,lowPosition - 1)) == 1
%             low_array = [low_array lowPosition];
%             %plot(lowPosition, PPGsignal(lowPosition),'X','MarkerSize',12)
%             %hold on;
%             
%         end
%     end
%     
%     % Necessary Alterations for the low_array, If two minimums are near 
%     % Note : Selecting which points should be deleted. 
%     delTrough = [];
%     for z = 2:1:length(low_array)
%         first = PPGsignal(low_array(z-1));
%         second = PPGsignal(low_array(z));
%         if low_array(z) - low_array(z-1) < 525
%             if first < second
%                 delTrough = [delTrough z];
%             else
%                 delTrough = [delTrough z-1];
%             end
%         end
%     end 
%     
%   % Deleting unwanted detected lows.
%     for y = length(delTrough):-1:1
%         low_array(delTrough(y)) = [];
%     end
%     
%     % Adjustemnt for the Peak Array
%     delPeak = [];
%     for z = 2:1:length(peak_array)
%         first = PPGsignal(peak_array(z-1));
%         second = PPGsignal(peak_array(z));
%         if peak_array(z) - peak_array(z-1) < 525
%             if first < second
%                 delPeak = [delPeak z-1];
%             else
%                 delPeak = [delPeak z];
%             end
%         end
%     end
%     
%     for y = length(delPeak):-1:1
%         peak_array(delPeak(y)) = [];
%     end
%     
%     %FINAL CUTOFF POINTS ACQUIRED - Plotting
%     for point = 1:length(low_array)
%         %line([low_array(point) low_array(point)],[min(PPGsignal) max(PPGsignal)],'Color','red','LineStyle','--');
%         %line([peak_array(point) peak_array(point)],[min(PPGsignal) max(PPGsignal)],'Color','blue','LineStyle','--');
%     end
%     
%     
%     % PPG Features 
%     
%     % identifiedPoints - Start with a trough , end with peak
%     allidentifiedPoints = [low_array peak_array];
%     allidentifiedPoints = sort(allidentifiedPoints);
%     low_start = low_array(1);
%     low_end = low_array(end);
%     for x = 1:1:length(allidentifiedPoints)
%         if allidentifiedPoints(x) == low_start
%             startIndex = x;
%         end
%         if allidentifiedPoints(x) == low_end
%             endIndex = x;
%         end
%     end
%     identifiedPoints = allidentifiedPoints(startIndex:endIndex);
%     for x = 1:1:length(identifiedPoints)
%         %plot(identifiedPoints(x), PPGsignal(identifiedPoints(x)),'o','MarkerSize',12)
%         %hold on;
%     end
%  
%     %disp(identifiedPoints);
%     
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     
%     % peak2peakInterval (p - P)
%     % pulseInterval (T - T) 
%     % Rise Time (T to P, peak - trough) 
%     % Amplitude
%     
%     beats = ( length(identifiedPoints) - 1 ) / 2 ;
%     % calculated inter beat
%     p2p = 0; pi = 0;
%     % when number of beats > 2
%     if beats == 1
%         p2p = -1;
%         pi = -1;
%     elseif beats == 2
%         p2p = p2p + identifiedPoints(4) - identifiedPoints(2);
%         pi =  pi + identifiedPoints(3) - identifiedPoints(1);
%     else 
%         for b = 1:1:beats - 1
%             p2p = p2p + identifiedPoints(2*b+2) - identifiedPoints(2*b);
%             pi =  pi + identifiedPoints(2*b+1) - identifiedPoints(2*b-1);
%         end
%         p2p = p2p / (beats - 1);
%         pi = pi / (beats - 1);
%     end
%     
%     % calculated per beat
%     rt = 0; amp = 0;
%     if beats == 1
%         rt = rt + identifiedPoints(2) - identifiedPoints(1);
%         amp = amp + PPGsignal(identifiedPoints(2)) - PPGsignal(identifiedPoints(1));
%     else
%         for b = 1:1:beats
%             rt =  rt + identifiedPoints(2*b) - identifiedPoints(2*b-1);
%             amp = amp + PPGsignal(identifiedPoints(2*b)) - PPGsignal(identifiedPoints(2*b-1));
%         end
%         rt = rt / beats;
%         amp = amp / beats;
%     end
%     
%     extractedFeatures = [round(p2p) round(pi) round(rt) round(amp)];
%     %disp(extractedFeatures);
%     
%     
%     % Save the extracted features to a CSV 
%     temp = [subject1(patient,:) extractedFeatures PPGsignal(1:2100)];
%     feature_vector = [feature_vector ; temp]; 
% end



% Write to CSV 
%csvwrite('BigData.csv',feature_vector);



function [x,y] = ppgBeatExtracter(PPGsignal,low_array,n)
y = PPGsignal(low_array(n):low_array(n+1));
ylen = length(y);
x = 1:1:ylen;
x = x / ylen;
end
