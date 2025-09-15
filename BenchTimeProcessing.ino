#include <arduinoFFT.h>
#include <math.h>
#include <algorithm>
#include <esp_heap_caps.h>

#define N 128
#define SAMPLING_FREQ 3200
#define SIGNAL_FREQ 50
#define OVERLAP 0.5
#define EPS 1e-12

// Arrays for FFT (using float for optimization)
float vReal[N];
float vImag[N];
float inputSignal[N];

// FFT object
ArduinoFFT<float> FFT(vReal, vImag, N, SAMPLING_FREQ);

// Function to calculate statistical moments (non-normalized)
float calculateMoment(float* data, int size, float mean, int order) {
    float moment = 0.0;
    for (int i = 0; i < size; i++) {
        moment += powf(fabsf(data[i] - mean), order);  // Absolute value as in Python
    }
    return moment / size;
}

// Function to calculate entropy with histogram
float calculateEntropyHistogram(float* data, int size, float bins = 50) {
    // Find min and max for bins
    float min_val = *std::min_element(data, data + size);
    float max_val = *std::max_element(data, data + size);
    
    // Create histogram
    float bin_width = (max_val - min_val) / bins;
    int histogram[bins];
    for (int i = 0; i < bins; i++) histogram[i] = 0;
    
    for (int i = 0; i < size; i++) {
        int bin_index = (int)((data[i] - min_val) / bin_width);
        if (bin_index >= bins) bin_index = bins - 1;
        if (bin_index < 0) bin_index = 0;
        histogram[bin_index]++;
    }
    
    // Calculate entropy
    float entropy = 0.0;
    float total = size;
    for (int i = 0; i < bins; i++) {
        if (histogram[i] > 0) {
            float p = (float)histogram[i] / total;
            entropy -= p * logf(p);
        }
    }
    
    return entropy;
}

// Function to calculate spectral entropy
float calculateSpectralEntropy(float* spectrum, int size, float total_energy) {
    float entropy = 0.0;
    for (int i = 0; i < size; i++) {
        float p = spectrum[i] / total_energy;
        if (p > EPS) {
            entropy -= p * logf(p);
        }
    }
    return entropy;
}

// Variables for continuous RAM monitoring
size_t min_ram = UINT32_MAX;
size_t max_ram = 0;

void printRamUsage(const char* stage) {
    size_t free_ram = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    size_t used_ram = heap_caps_get_total_size(MALLOC_CAP_8BIT) - free_ram;
    
    // Update min and max
    min_ram = (free_ram < min_ram) ? free_ram : min_ram;
    max_ram = (free_ram > max_ram) ? free_ram : max_ram;
    
    Serial.printf("[RAM @ %s] Free: %6d bytes | Used: %6d bytes\n", 
                  stage, free_ram, used_ram);
}

void setup() {
    Serial.begin(115200);
    while (!Serial); // Wait for serial connection
    Serial.println("System started. Periodic measurements every 2 seconds...");
    printRamUsage("Setup complete");
}

void loop() {
    printRamUsage("Loop start");
    unsigned long startTime = micros();

    // 1. Generate clean sine wave and remove DC offset
    float sum_signal = 0.0;
    for (int i = 0; i < N; i++) {
        inputSignal[i] = 100.0 * sinf(2.0 * PI * SIGNAL_FREQ * i / (float)SAMPLING_FREQ);
        sum_signal += inputSignal[i];
    }
    
    // Remove DC offset
    float dc_offset = sum_signal / N;
    for (int i = 0; i < N; i++) {
        inputSignal[i] -= dc_offset;
        vReal[i] = inputSignal[i];
        vImag[i] = 0.0;
    }
    printRamUsage("Signal generated");

    // 2. Time-domain feature calculation
    float peak = *std::max_element(inputSignal, inputSignal + N);
    float trough = *std::min_element(inputSignal, inputSignal + N);
    float mean = 0.0, absMean = 0.0, rms = 0.0;

    for (int i = 0; i < N; i++) {
        mean += inputSignal[i];
        absMean += fabsf(inputSignal[i]);
        rms += inputSignal[i] * inputSignal[i];
    }
    mean /= N;
    absMean /= N;
    rms = sqrtf(rms / N);

    // Standard deviation and variance
    float stdDev = 0.0;
    for (int i = 0; i < N; i++) {
        stdDev += powf(inputSignal[i] - mean, 2);
    }
    stdDev = sqrtf(stdDev / N);
    float variance = stdDev * stdDev;

    // Statistical moments
    float skewness = calculateMoment(inputSignal, N, mean, 3) / powf(stdDev, 3);
    float kurtosis_val = calculateMoment(inputSignal, N, mean, 4) / powf(stdDev, 4) - 3;

    // Amplitude features
    float crestFactor = (rms > EPS) ? peak / rms : 0.0;
    float formFactor = (absMean > EPS) ? rms / absMean : 0.0;
    float impulseFactor = (absMean > EPS) ? peak / absMean : 0.0;

    // SMR (Square Mean Root)
    float smr = 0.0;
    for (int i = 0; i < N; i++) {
        smr += sqrtf(fabsf(inputSignal[i]));
    }
    smr = powf(smr / N, 2);

    // Higher-order moments
    float n5m = calculateMoment(inputSignal, N, mean, 5);
    float n6m = calculateMoment(inputSignal, N, mean, 6);

    // Time-domain entropy
    float timeEntropy = calculateEntropyHistogram(inputSignal, N);
    printRamUsage("Time-domain features");

    // 3. FFT and frequency-domain features
    FFT.windowing(FFT_WIN_TYP_HANN, FFT_FORWARD);
    FFT.compute(FFT_FORWARD);
    FFT.complexToMagnitude();
    printRamUsage("FFT computed");

    // Convert to power spectrum 
    float power_spectrum[N/2];
    float total_energy = EPS;
    for (int i = 0; i < N/2; i++) {
        power_spectrum[i] = vReal[i] * vReal[i];  // |X|^2
        total_energy += power_spectrum[i];
    }

    // Relative spectrum (normalized by total energy)
    float spec_rel[N/2];
    for (int i = 0; i < N/2; i++) {
        spec_rel[i] = power_spectrum[i] / total_energy;
    }

    // Frequency-domain variables
    float fft_mean = 0.0f, fft_std = 0.0f, fft_peak_val = 0.0f;
    float fft_entropy = 0.0f, fft_energy_low = 0.0f, fft_energy_high = 0.0f;
    float fft_fc = 0.0f, fft_rmsf = 0.0f, fft_rvf = 0.0f;

    // Calculate frequencies (Hz)
    float freqs[N/2];
    float max_freq = 0.0;
    for (int i = 0; i < N/2; i++) {
        freqs[i] = i * (SAMPLING_FREQ / (float)N);
        if (freqs[i] > max_freq) max_freq = freqs[i];
    }

    // Frequency-domain features
    for (int i = 0; i < N/2; i++) {
        fft_mean += spec_rel[i];
        if (spec_rel[i] > fft_peak_val) fft_peak_val = spec_rel[i];

        if (freqs[i] < 0.25 * max_freq) {
            fft_energy_low += spec_rel[i];
        } else {
            fft_energy_high += spec_rel[i];
        }

        fft_fc += freqs[i] * power_spectrum[i];
        fft_rmsf += freqs[i] * freqs[i] * power_spectrum[i];
    }

    fft_mean /= (N/2);
    fft_fc /= total_energy;
    fft_rmsf = sqrtf(fft_rmsf / total_energy);
    fft_rvf = sqrtf(fabsf(fft_rmsf * fft_rmsf - fft_fc * fft_fc));

    // Standard deviation FFT
    for (int i = 0; i < N/2; i++) {
        fft_std += powf(spec_rel[i] - fft_mean, 2);
    }
    fft_std = sqrtf(fft_std / (N/2));

    // Spectral entropy
    fft_entropy = calculateSpectralEntropy(power_spectrum, N/2, total_energy);
    printRamUsage("Frequency-domain features");

    // 4. Final time measurement
    unsigned long endTime = micros();

    // 5. Print results 
    Serial.println("\nTime-domain Features");
    Serial.printf("peak: %.4f\n", fabsf(peak));  
    Serial.printf("trough: %.4f\n", trough);
    Serial.printf("meanv: %.4f\n", mean);
    Serial.printf("sd: %.4f\n", stdDev);
    Serial.printf("rms: %.4f\n", rms);
    Serial.printf("skewn: %.4f\n", skewness);
    Serial.printf("kurto: %.4f\n", kurtosis_val);
    Serial.printf("crest: %.4f\n", crestFactor);
    Serial.printf("form: %.4f\n", formFactor);
    Serial.printf("smr: %.4f\n", smr);
    Serial.printf("impulse: %.4f\n", impulseFactor);
    Serial.printf("entropy: %.4f\n", timeEntropy);
    Serial.printf("var: %.4f\n", variance);
    Serial.printf("n5m: %.4f\n", n5m);
    Serial.printf("n6m: %.4f\n", n6m);

    Serial.println("\nFrequency-domain Features");
    Serial.printf("fft_mean: %.6f\n", fft_mean);
    Serial.printf("fft_std: %.6f\n", fft_std);
    Serial.printf("fft_peak: %.6f\n", fft_peak_val);
    Serial.printf("fft_entropy: %.6f\n", fft_entropy);
    Serial.printf("fft_energy_low: %.6f\n", fft_energy_low);
    Serial.printf("fft_energy_high: %.6f\n", fft_energy_high);
    Serial.printf("fft_fc: %.2f\n", fft_fc);
    Serial.printf("fft_rmsf: %.2f\n", fft_rmsf);
    Serial.printf("fft_rvf: %.2f\n", fft_rvf);

    Serial.println("\nPerformance");
    Serial.printf("Computation time: %lu µs\n", endTime - startTime);
    Serial.printf("Minimum free RAM: %d bytes\n", min_ram);
    Serial.printf("Maximum free RAM: %d bytes\n", max_ram);
    printRamUsage("End of loop");
    
    // 6. Debug: Print first 10 points of the signal
    Serial.println("First 10 signal points:");
    for(int i=0; i<10; i++) {
        Serial.printf("%.2f ", inputSignal[i]);
    }
    Serial.println("\n");
    
    // 7. Debug: Print significant FFT peaks
    Serial.println("Significant FFT peaks (>1.0):");
    for(int i=0; i<N/2; i++) {
        if(vReal[i] > 1.0) {
            float freq = i * (SAMPLING_FREQ / (float)N);
            Serial.printf("%.2f Hz: %.2f\n", freq, vReal[i]);
        }
    }

    Serial.println("Waiting 2 seconds for next measurement...");
    
    // Reset RAM counters for next cycle
    min_ram = UINT32_MAX;
    max_ram = 0;
    
    delay(2000);
};
