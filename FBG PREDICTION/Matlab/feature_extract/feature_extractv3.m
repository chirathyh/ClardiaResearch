function [feature_vector, status] = feature_extractv3(PPGsignal, subject_ID)
            % The function calculates the PPG & APG based features. 
            % Todo : Summarize the features here. 
            
            THRESHOLD = 5; % Prev set to 10 in Fs = 125Hz
            
            % 1st Derivative
            x = 1:1:length(PPGsignal);
            dy = diff(PPGsignal)./diff(x);
            dy = preprocessSignal(dy); 
            % 2nd derivative
            x2 = 1:1:length(dy);
            dy2 = diff(dy)./diff(x2);
            dy2 = preprocessSignal(dy2); 
            offset = min(dy2);
            new_dy2 = dy2 + abs(offset)*0 ; %offset -> 0 
            new_dy2 = smooth(new_dy2);
            
            % Detecting the Peaks of the APG Waveform. 
            peak_loc   = [];
            peak_val   = [];
            trough_loc = [];
            trough_val = [];

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

            % Find the first three maximum points in desceding order. 
            % Roughly there will be two peaks in 2.1s APG waveform.
            % Select the two maximum peaks and then need to find the first occurence of the alpha wave.
            % max3 used for the second peak detection.
            
            max_finder = peak_val;
            [max1, ind1] = max(max_finder);
            max_finder(ind1) = -100000;
            [max2, ind2] = max(max_finder);
            max_finder(ind2) = -100000;
            [max3, ind3] = max(max_finder);

            % Use a threshold to select the other points. Threhold dependent on the dataset.
            % The default d,e,f waves set to one, if unchanged features not
            % extracted properly. 
            d = 1; e = 1; f = 1; 
            
            a = min(peak_loc(ind1),peak_loc(ind2));
            for i=1:1:length(peak_loc)
                if peak_loc(i) > a
                    c = peak_loc(i);
                    for ii=i+1:1:length(peak_loc)
                        if (peak_loc(ii) - c) > THRESHOLD  
                            e = peak_loc(ii);
                            break;
                        end   
                    end
                    break;
                end
            end
            for i=1:1:length(trough_loc)
                if trough_loc(i) > a
                    b = trough_loc(i);
                    for ii=i+1:1:length(trough_loc)
                        if (trough_loc(ii) - b ) > THRESHOLD
                            d = trough_loc(ii);
                            for iii=ii+1:1:length(trough_loc)
                                if (trough_loc(iii) - d) > THRESHOLD
                                    f = trough_loc(iii);
                                    break;
                                end
                            end
                        end
                        break;
                    end
                    break;
                end
            end

            %Identifying the Second Peak(a1) of the APG waveform.
            selected_temp2 = max(peak_loc(ind1),peak_loc(ind2));
            if (a < peak_loc(ind3) && peak_loc(ind3) < selected_temp2) && (max3 > new_dy2(e))
                selected = peak_loc(ind3);
                a1 = selected;
            else
                a1 = selected_temp2;
            end

            %%%% PPG Main Signal %%%%
            dicro_notch = e;
            dicro_peak  = f;
            PPGsignal   = smooth(PPGsignal);
            [ppg_peak_val,ppg_peaks_loc] = findpeaks(PPGsignal);
            negPPGsignal = -1 * PPGsignal;
            [ppg_trough_val, ppg_trough_loc] = findpeaks(negPPGsignal);

            % Detect main & second peak of PPG
            amp = 1;
            for i=1:1:length(ppg_peaks_loc)
                if ppg_peaks_loc(i) > a
                    amp = ppg_peaks_loc(i);
                    break;
                end
            end
            amp1 = -1;
            for i=1:1:length(ppg_peaks_loc)
                if ppg_peaks_loc(i) > a1
                    amp1 = ppg_peaks_loc(i);
                    break;
                end
            end
            
            % Detect the first trough
            low1 = -1; low2 = -1;
            for i=1:1:length(ppg_trough_loc)
                if ppg_trough_loc(i) < a
                    low1 = ppg_trough_loc(i);
                end
                if ppg_trough_loc(i) < a1
                    low2 = ppg_trough_loc(i);
                end
            end

            temp_PPGsignal = detrend(PPGsignal,'linear');
            PPGsignal = temp_PPGsignal;

            % unknow issue with low becoming negative.
            if low2 < 0
                low2 = 1;
            end
           
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
            
            %height = subject(patient,4);
            height = 0; %since the height isn't measured. 
            
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


            feature_vector = [B/A C/A D/A E/A (B-C-D-E)/A (B-E)/A (B-C-D)/A (C+D-B)/A (a1-a) SysAmp TotArea AreaRatio PI PI_Sys AI adj_AI ArtStiff RT];    
            
            % Delecting the quality of extracted features. 
            if (d == 1 || e == 0 || f == 1 || TotArea == -1 || AreaRatio == -1 || low1 == -1 || low2 == -1 || amp == -1 || amp1 == -1)
                status = 1;
            else
                status = 0;
                
%                 %display_plots();
%                 figure
%                 subplot(4,1,1)
%                 plot(PPGsignal),grid;
%                 graph_title = subject_ID;
%                 title(graph_title);
%                 
%                 subplot(4,1,2)
%                 plot(dy),grid;
%                 title('VPG WAVE');
%                 
%                 subplot(4,1,4);
%                 plot(new_dy2), grid on;
%                 title('Accelerated Photoplethysmography Signal (APG)');
%                 xlabel('Time');
%                 ylabel('Amplitude');
%                 hold on;
%                 plot(a, new_dy2(a),'r.','MarkerSize',12);
%                 hold on;
%                 plot(b, new_dy2(b),'b.','MarkerSize',12);
%                 hold on;
%                 plot(c, new_dy2(c),'k.','MarkerSize',12);
%                 hold on;
%                 plot(d, new_dy2(d),'g.','MarkerSize',12);
%                 hold on;
%                 plot(e, new_dy2(e),'c.','MarkerSize',12);
%                 hold on;
%                 plot(f, new_dy2(f),'m.','MarkerSize',12);
%                 hold on;
%                 plot(a1, new_dy2(a1),'r.','MarkerSize',14);
%                 %xlim([0 300])
%                 
%                 subplot(4,1,3)
%                 plot(PPGsignal),grid;
%                 title('Photoplethysmography Signal (PPG)');
%                 xlabel('Time');
%                 ylabel('Amplitude');
%                 hold on;
%                 plot(dicro_notch, PPGsignal(dicro_notch),'c.','MarkerSize',14);
%                 plot(dicro_peak, PPGsignal(dicro_peak),'m.','MarkerSize',14);
%                 plot(amp, PPGsignal(amp),'r.','MarkerSize',14);
%                 
%                 plot(low1, PPGsignal(low1),'b.','MarkerSize',14);
%                 plot(amp1, PPGsignal(amp1),'r.','MarkerSize',14);
%                 plot(low2, PPGsignal(low2),'b.','MarkerSize',14);
%                 %xlim([0 300])
%  
                
                
            end
 
