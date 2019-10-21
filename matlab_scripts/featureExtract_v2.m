addpath(genpath('lib/'));
SIGNAL_PATH = '/Users/chirathhettiarachchi/tensorflow/datasets/clardia3/PPG/signals/';
%subject = csvread('diabetesPatientInfo.csv',1,0);
subject = csvread('HypertensionPatientInfo.csv',1,0);
%subject = csvread('normalPatientInfo.csv',1,0);
[m,n] = size(subject);
feature_vector = [];
disp(m)


for patient = 1:1:m
    subject_ID = int2str(subject(patient,1));
    selected = int2str(subject(patient,15));
    tempSubject = strcat(SIGNAL_PATH,subject_ID,'_',selected,'.txt'); 
    
    raw_PPG = load(tempSubject);
    new_PPGsignal = preprocessSignal(raw_PPG);
    PPGsignal = new_PPGsignal;

    figure
    subplot(4,1,1)
    plot(PPGsignal),grid;
    title(subject_ID);
    
    % 1st Derivative
    x = 1:1:length(PPGsignal);
    dy = diff(PPGsignal)./diff(x);
    dy = preprocessSignal(dy); 
    subplot(4,1,2)
    plot(dy),grid;
    title('VPG WAVE');
    
    % 2nd derivative
    x2 = 1:1:length(dy);
    dy2 = diff(dy)./diff(x2);
    dy2 = preprocessSignal(dy2); 
    
    offset = min(dy2);
    new_dy2 = dy2 + abs(offset)*0 ; %offset -> 0 
    
    %figure
    subplot(4,1,3);
    new_dy2 = smooth(new_dy2);
    plot(new_dy2), grid on;
    title('Accelerated Photoplethysmography Signal (APG)');
    xlabel('Time');
    ylabel('Amplitude');
    hold on;

    peak_loc = [];
    peak_val = [];
    trough_loc = [];
    trough_val = [];
    temp = [];
    for i=2:1:length(new_dy2)-1
        cur = new_dy2(i);
        prev = new_dy2(i-1);
        next = new_dy2(i+1);
        if prev < cur && cur > next
            peak_loc = [peak_loc i];
            peak_val = [peak_val new_dy2(i)];
        end
        if prev > cur && cur < next
            trough_loc = [trough_loc i];
            trough_val = [trough_val new_dy2(i)];
        end
    end

%     for i=1:1:length(peak_loc)
%          plot(peak_loc(i),peak_val(i),'r.','MarkerSize',12)
%          hold on;
%     end
%     for i=1:1:length(trough_loc)
%          plot(trough_loc(i),trough_val(i),'g.','MarkerSize',12)
%          hold on;
%     end

    max_finder = peak_val;
    [max1, ind1] = max(max_finder);
    max_finder(ind1) = -Inf;
    [max2, ind2] = max(max_finder);
    max_finder(ind2) = -Inf;
    [max3, ind3] = max(max_finder);
    
    
    %plot(peak_loc(ind1), max1,'bx','MarkerSize',14);
    %hold on;
    %plot(peak_loc(ind2), max2,'bx','MarkerSize',14);
    %old on;
 
    a = min(peak_loc(ind1),peak_loc(ind2));
    for i=1:1:length(peak_loc)
        if peak_loc(i) > a
            c = peak_loc(i);
            for ii=i+1:1:length(peak_loc)
                if (peak_loc(ii) - c )> 10  %threshold
                    e = peak_loc(ii);
                end
                break;
            end
            break;
        end
    end
    for i=1:1:length(trough_loc)
        if trough_loc(i) > a
            b = trough_loc(i);
            for ii=i+1:1:length(trough_loc)
                if (trough_loc(ii) - b )> 10    %threshold
                    d = trough_loc(ii);
                    for iii=ii+1:1:length(trough_loc)
                        if (trough_loc(iii) -d) > 10
                            f = trough_loc(iii);
                        end
                        break;
                    end
                end
                break;
            end
            break;
        end
    end
    
    %h = zeros(6,1);
    %h(1) = plot(a, new_dy2(a),'r.','MarkerSize',12);
    %h(2) = plot(b, new_dy2(b),'b.','MarkerSize',12);
    %h(3) = plot(c, new_dy2(c),'k.','MarkerSize',12);
    %h(4) = plot(d, new_dy2(d),'g.','MarkerSize',12);
    %h(5) = plot(e, new_dy2(e),'c.','MarkerSize',12);
    %h(6) = plot(f, new_dy2(f),'m.','MarkerSize',12);
    
    plot(a, new_dy2(a),'r.','MarkerSize',12);
    hold on;
    plot(b, new_dy2(b),'b.','MarkerSize',12);
    hold on;
    plot(c, new_dy2(c),'k.','MarkerSize',12);
    hold on;
    plot(d, new_dy2(d),'g.','MarkerSize',12);
    hold on;
    plot(e, new_dy2(e),'c.','MarkerSize',12);
    hold on;
    plot(f, new_dy2(f),'m.','MarkerSize',12);
    hold on;
    
    
    
    %Identifying the Second Peak
    selected_temp2 = max(peak_loc(ind1),peak_loc(ind2));
    if (a < peak_loc(ind3) && peak_loc(ind3) < selected_temp2) && (max3 > new_dy2(e))
        selected = peak_loc(ind3);
        a1 = selected;
        plot(selected, new_dy2(selected),'r.','MarkerSize',14);
    else
        plot(selected_temp2, new_dy2(selected_temp2),'r.','MarkerSize',14);
        a1 = selected_temp2;
    end
    %legend(h,'a-wave','b-wave','c-wave','d-Wave','e-wave','Diastolic Peak')
    
    %%%% PPG Main Signal %%%%
    dicro_notch = e;
    dicro_peak = f;
    
    PPGsignal = smooth(PPGsignal);
    [ppg_peak_val,ppg_peaks_loc] = findpeaks(PPGsignal);
    negPPGsignal = -1 * PPGsignal;
    [ppg_trough_val, ppg_trough_loc] = findpeaks(negPPGsignal);
    
    for i=1:1:length(ppg_peaks_loc)
        if ppg_peaks_loc(i) > a
            amp = ppg_peaks_loc(i);
            break;
        end
    end
    low1 = -1; low2 = -1;
    for i=1:1:length(ppg_trough_loc)
        if ppg_trough_loc(i) < a
            low1 = ppg_trough_loc(i);
        end
        if ppg_trough_loc(i) < a1
            low2 = ppg_trough_loc(i);
        end
    end
    amp1 = -1;
    for i=1:1:length(ppg_peaks_loc)
        if ppg_peaks_loc(i) > a1
            amp1 = ppg_peaks_loc(i);
            break;
        end
    end

    temp_PPGsignal = detrend(PPGsignal,'linear');
    PPGsignal = temp_PPGsignal;
    
    subplot(4,1,4)
    plot(PPGsignal),grid;
    title('Photoplethysmography Signal (PPG)');
    xlabel('Time');
    ylabel('Amplitude');
    hold on;
    
    %h = zeros(4,1);
    %h(1) = plot(dicro_notch, PPGsignal(dicro_notch),'b.','MarkerSize',14);
    %h(2) = plot(dicro_peak, PPGsignal(dicro_peak),'k.','MarkerSize',14);
    %h(3) = plot(low1, PPGsignal(low1),'g.','MarkerSize',14);
    %h(4) = plot(amp1, PPGsignal(amp1),'r.','MarkerSize',14);
    
    plot(dicro_notch, PPGsignal(dicro_notch),'b.','MarkerSize',14);
    plot(dicro_peak, PPGsignal(dicro_peak),'k.','MarkerSize',14);
    plot(amp, PPGsignal(amp),'r.','MarkerSize',14);
    if low1 ~= -1
        plot(low1, PPGsignal(low1),'g.','MarkerSize',14);
    end
    if amp1 ~= -1
        plot(amp1, PPGsignal(amp1),'r.','MarkerSize',14);
    end
    plot(low2, PPGsignal(low2),'g.','MarkerSize',14);
    
    %legend(h,'Dicrotic Notch','Diastolic Peak','PPG Signal Start','Systolic Peak')
    
    % FEATURE CALCULATION
    %Heights of the a,b,c,d,e waves
    A = new_dy2(a);
    B = new_dy2(b);
    C = new_dy2(c);
    D = new_dy2(d);
    E = new_dy2(e);
    
    % PPG Features
    if low1 ~= -1
        SysAmp = PPGsignal(amp) - low1;
    elseif amp1 ~= -1
        SysAmp = PPGsignal(amp1) - low1;
    else 
        SysAmp = -1;
    end
    DysAmp = PPGsignal(dicro_peak);
    height = subject(patient,4);
    %calculating PI, If PI cant be calculated then the p2p is taken
    if low1 ~= -1
        PI = low2 - low1;
    elseif amp1 ~= -1
        PI = amp1 - amp;
    else
        PI = -1;
    end
    PI_Sys = PI /SysAmp ;
    AI = DysAmp / SysAmp;
    adj_AI = (SysAmp - DysAmp) / SysAmp ;
    deltaT = dicro_peak - amp;
    ArtStiff = height / deltaT;
    
    %Calculate the RT
    if low1 ~= -1
        RT = amp - low1;
    elseif amp1 ~= -1
        RT = amp1 - low2;
    else
        RT = -1;
    end
    
    % Pulse Area Related Features
    pulse2 = PPGsignal(amp:dicro_notch);
    area2 = trapz(pulse2);
    pulse3 = PPGsignal(dicro_notch:low2);
    area3 = trapz(pulse3);
    if low1 ~= -1
        pulse1 = PPGsignal(low1:amp);
        area1 = trapz(pulse1);
        TotArea = area1 + area2 + area3;
        AreaRatio = (area1+area2) / area3;
    elseif amp1 ~= -1
        pulse1 = PPGsignal(low2:amp1);
        area1 = trapz(pulse1);
        TotArea = area1 + area2 + area3;
        AreaRatio = (area1+area2) / area3;
    else
        TotArea = -1;
        AreaRatio = -1;
    end
    
    %pulse width related features
    %halfAmp = SysAmp / 2;
    %index = find(PPGsignal == halfAmp);
    %disp(index)
    
    
    extractedFeatures = [B/A C/A D/A E/A (B-C-D-E)/A (B-E)/A (B-C-D)/A (C+D-B)/A (a1-a) SysAmp TotArea AreaRatio PI PI_Sys AI adj_AI ArtStiff RT];    
    temp = [subject(patient,:) extractedFeatures ];
    feature_vector = [feature_vector ; temp];
    
end

% Write to CSV 
%csvwrite('hypertensionAPGFeaturesV2.csv',feature_vector);
