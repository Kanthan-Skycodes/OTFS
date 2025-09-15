clc; clear; close all;

%% Simulation Parameters
M = 6;                 % Number of subcarriers
N = 6;                 % Number of time slots
MN = M * N;             % Total symbols per frame
K = 2;                  % Modulation order (QPSK)
D = 0.05;               % Distortion requirement
snr_db = 0:5:30;        % SNR range in dB
num_realizations = 1e4; % Monte Carlo realizations

%% Channel Parameters
carrier_freq = 4e9;     % Carrier frequency (Hz)
delta_f = 15e3;         % Subcarrier spacing (Hz)
T = 1 / delta_f;        % Symbol duration (s)
max_speed = 100;        % Maximum UE speed (km/h)

%% Derived Parameters
Hb = -D*log2(D) - (1-D)*log2(1-D); % Binary entropy
R = K * (1 - Hb);                  % Target rate (Eq. 6-7)
EsN0_lin = 10.^(snr_db/10);        % Linear SNR values

%% Preallocate Results
outage_prob = zeros(size(snr_db));
lower_bound = zeros(size(snr_db));

%% Main Simulation Loop
for snr_idx = 1:length(snr_db)
    fprintf('Processing SNR = %d dB...\n', snr_db(snr_idx));
    EsN0 = EsN0_lin(snr_idx);
    outage_count = 0;
    
    for mc_iter = 1:num_realizations
        %% Generate Channel (EVA model)
        [chan_coef, delay_taps, Doppler_taps, P] = ...
            Generate_delay_Doppler_channel_parameters(N, M, carrier_freq, delta_f, T, max_speed);
        
        %% Construct DD Domain Channel Matrix (Eq. 4)
        H_DD = zeros(MN);
        F_N = dftmtx(N)/sqrt(N);
        
        for i = 1:P
            l_i = mod(delay_taps(i), M);        % Wrap delay
            Pi = circshift(eye(MN), l_i, 2);    % Delay permutation
            
            k_i = Doppler_taps(i);              % Real-valued Doppler
            alpha = exp(1j*2*pi*k_i/(M*N));
            Delta = diag(alpha.^(0:MN-1));      % Doppler diagonal
            
            term = kron(F_N, eye(M)) * Pi * Delta * kron(F_N', eye(M));
            H_DD = H_DD + chan_coef(i) * term;
        end
        H_DD = H_DD / sqrt(P); % Normalize total power

        %% Stable Capacity Calculation (Eq. 5)
        H_product = H_DD' * H_DD;
        H_product = (H_product + H_product') / 2; % Hermitian
        eigenvalues = eig(H_product);            % Real eigenvalues
        
        % Numerical stable log-det
        max_eig = max(eigenvalues);
        if max_eig > 0
            scaled_EsN0 = EsN0 / max_eig;
            log_det = sum(log2(1 + scaled_EsN0 * eigenvalues));
            log_det = log_det + log2(max_eig) * MN; % Rescale
        else
            log_det = 0;
        end
        C = log_det / (M*N); % Normalized capacity
        
        %% Outage Check
        if C < R
            outage_count = outage_count + 1;
        end
    end
    
    %% Outage Probability
    outage_prob(snr_idx) = outage_count / num_realizations;
    
    %% Lower Bound (Eq. 17)
    gamma = P * (2^(K*(1 - Hb)) - 1) / EsN0; 
    lower_bound(snr_idx) = 1 - exp(-gamma) * sum(gamma.^(0:P-1)./factorial(0:P-1));
end

%% Plot Results
figure;
semilogy(snr_db, outage_prob, '-o', 'LineWidth', 2, 'DisplayName', 'Simulation');
hold on;
semilogy(snr_db, lower_bound, '-s', 'LineWidth', 2, 'DisplayName', 'Lower Bound');
grid on;
xlabel('SNR (dB)');
ylabel('Outage Probability');
legend('Location', 'best');
title('OTFS Outage Probability with EVA Channel (M=16, N=16)');
set(gca, 'YScale', 'log');



%% -----------------------------------------------------------------------
%% Subfunction: Generate_delay_Doppler_channel_parameters
%% -----------------------------------------------------------------------
function [chan_coef, delay_taps, Doppler_taps, taps] = Generate_delay_Doppler_channel_parameters(N, M, car_fre, delta_f, T, max_speed)
    % Delay & Doppler resolutions
    one_delay_tap   = 1/(M * delta_f);
    one_doppler_tap = 1/(N * T);

    % Path delays (EVA model)
    delays = [0 30 150 310 370 710 1090 1730 2510] * 1e-9; % in seconds
    taps   = length(delays);

    % Quantize delays to nearest tap
    delay_taps = round(delays / one_delay_tap);

    % Power Delay Profile (dB)
    pdp = [0 -1.5 -1.4 -3.6 -0.6 -9.1 -7.0 -12.0 -16.9];
    pow_prof = 10.^(pdp/10);
    pow_prof = pow_prof / sum(pow_prof);

    % Rayleigh channel coeffs (one-tap Rayleigh) for each path
    % then cascade two to get cascaded fading if you wish:
    % Uncomment next lines to model cascaded Rayleigh:
    % h1 = sqrt(pow_prof) .* (randn(1,taps) + 1i*randn(1,taps))/sqrt(2);
    % h2 = sqrt(pow_prof) .* (randn(1,taps) + 1i*randn(1,taps))/sqrt(2);
    % chan_coef = h1 .* h2;
    chan_coef = sqrt(pow_prof) .* (randn(1,taps) + 1i*randn(1,taps))/sqrt(2);

    % Doppler taps via Jake's model
    max_UE_speed = max_speed * (1000/3600); % km/h to m/s
    Doppler_vel  = (max_UE_speed * car_fre) / 3e8;
    max_Doppler_tap = Doppler_vel / one_doppler_tap;
    Doppler_taps = max_Doppler_tap * cos(2*pi*rand(1, taps));
end
