clc; clear;

%% Parameters
M = 4; N = 2; L = 2; % OTFS grid and channel taps
SNR_dB = 0:2:20;
num_trials = 1000;
bits_per_symbol = 2; % QPSK

%% BER storage
ber_precoded = zeros(size(SNR_dB));

%% QPSK modulation
generate_qpsk = @(bits) ((1 - 2*bits(1:2:end)) + 1j*(1 - 2*bits(2:2:end))) / sqrt(2);

%% QPSK demodulation
function bits = qpsk_demod(symbols)
    symbols = symbols * sqrt(2); % Undo normalization
    bits_i = real(symbols) < 0;
    bits_q = imag(symbols) < 0;
    bits = zeros(2*length(symbols), 1);
    bits(1:2:end) = bits_i;
    bits(2:2:end) = bits_q;
end

%% OTFS Modulation
function x_tf = otfs_modulate(x_dd, M, N)
    X_tf = ifft(ifft(x_dd, N, 2), M, 1);  % ISFFT
    x_tf = reshape(X_tf, [], 1);         % Serialize
end

%% OTFS Demodulation
function x_dd = otfs_demodulate(y_tf, M, N)
    Y_tf = reshape(y_tf, M, N);               % De-serialize
    x_dd = fft(fft(Y_tf, M, 1), N, 2);        % SFFT
end

%% Vandermonde Precoding
function x_precoded = apply_precoding(x_dd, M, N)
    MN = M * N;
    alpha = exp(1j * 2 * pi * (0:MN-1) / MN);  % Length-MN base vector
    P = zeros(MN, MN);
    for i = 1:MN
        P(:, i) = alpha.'.^(i-1);  % Now each column is 64x1
    end
    P = P / norm(P, 'fro'); % Normalize
    x_precoded = reshape(P * x_dd(:), M, N);  % Apply precoding
end

%% ML Detector
function x_hat = ml_detector(y, H_eff, constellation)
    K = size(H_eff, 2);
    [grid{1:K}] = ndgrid(constellation);
    x_all = reshape(cat(K+1, grid{:}), [], K).';
    dists = vecnorm(y - H_eff * x_all, 2, 1).^2;
    [~, idx] = min(dists);
    x_hat = x_all(:, idx);
end

%% Simulation Loop
for snr_idx = 1:length(SNR_dB)
    snr = SNR_dB(snr_idx);
    total_err = 0;
    total_bits = 0;

    for t = 1:num_trials
        % Generate bits and QPSK symbols
        bits = randi([0 1], M*N*bits_per_symbol, 1);
        symbols = generate_qpsk(bits);
        x_dd = reshape(symbols, M, N);

        % Precoding
        x_dd_precoded = apply_precoding(x_dd, M, N);

        % OTFS Modulation
        x_tf = otfs_modulate(x_dd_precoded, M, N);

        % Frequency-selective Channel
        H = zeros(M*N);
        delays = randi([0 M*N/4], L, 1);
        h = (randn(L,1) + 1j*randn(L,1));
        h = h / norm(h); % Normalize channel energy

        for l = 1:L
            H = H + h(l) * circshift(eye(M*N), delays(l), 1);
        end

        noise_var = 10^(-snr/10);
        noise = sqrt(noise_var/2) * (randn(M*N,1) + 1j*randn(M*N,1));
        y_tf = H * x_tf + noise;

        % OTFS Demodulation
        x_dd_hat = otfs_demodulate(y_tf, M, N);

        % Effective channel: approximate using same H (assuming linear system)
        % You can improve this by calculating H_eff for the delay-Doppler domain
        x_dd_hat_vec = x_dd_hat(:);
        constellation = [1+1j, 1-1j, -1+1j, -1-1j] / sqrt(2);

        % ML Detection
        x_hat = ml_detector(x_dd_hat_vec, eye(M*N), constellation);

        bits_hat = qpsk_demod(x_hat);

        % BER computation
        total_bits = total_bits + length(bits);
        total_err = total_err + sum(bits ~= bits_hat);
    end

    ber_precoded(snr_idx) = total_err / total_bits;
    fprintf("SNR = %d dB, BER = %e\n", snr, ber_precoded(snr_idx));
end

%% Plotting
semilogy(SNR_dB, ber_precoded, 'b-o', 'LineWidth', 1.5); hold on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('BER of Linearly Precoded OTFS with ML Detection');
grid on;
legend('Precoded OTFS');
