import os
import numpy as np
import wfdb
from functools import partial, reduce
from typing import Tuple, List, Optional, Callable
from itertools import takewhile

# Pure functions for data processing
def find_ecg_file(files: List[str]) -> Optional[str]:
    """Finds ECG file from list of files."""
    ecg_files = [f for f in files if f.endswith("_ECG.dat")]
    return ecg_files[0][:-4] if ecg_files else None

def load_record_data(record_path: str) -> Tuple[np.ndarray, int]:
    """Pure function to load WFDB record data."""
    record = wfdb.rdrecord(record_path)
    return record.p_signal[:, 0], record.fs

def truncate_signal(max_length: int) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a function that truncates signal to max_length if needed."""
    return lambda signal: signal[:max_length] if len(signal) > max_length else signal

def compose(*functions):
    """Function composition utility."""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def pipe(value, *functions):
    """Pipe value through sequence of functions."""
    return reduce(lambda acc, func: func(acc), functions, value)

def apply_with_params(func: Callable, *args, **kwargs) -> Callable:
    """Returns a function with pre-applied parameters."""
    return lambda data: func(data, *args, **kwargs)

# Core ECG processing functions (from the original paste.txt)
def center_ecg_signal(ecg_signal: np.ndarray) -> np.ndarray:
    """Centers ECG signal by subtracting mean value."""
    return ecg_signal - np.mean(ecg_signal)

def high_pass_filter(signal: np.ndarray, samp_rate: int, cutoff_freq: int = 50) -> np.ndarray:
    """Applies high-pass filter to signal."""
    tau = 1 / (2 * np.pi * cutoff_freq)
    alpha = tau / (tau + 1 / (2 * np.pi * samp_rate))

    def filter_step(acc, x):
        y_prev, x_prev, result = acc
        y = alpha * (y_prev + x - x_prev)
        return (y, x, result + [y])

    _, _, filtered = reduce(filter_step, signal, (0, 0, []))
    return np.array(filtered)

def remove_baseline_wander_fft(ekg_data: np.ndarray, sampling_rate: int = 4, cutoff_frequency: float = 0.5) -> np.ndarray:
    """Removes baseline wander using FFT."""
    from scipy.fft import fft, ifft

    spectrum = fft(ekg_data)
    freq = np.fft.fftfreq(len(ekg_data), 1 / sampling_rate)
    spectrum[(freq > -cutoff_frequency) & (freq < cutoff_frequency)] = 0
    return np.real(ifft(spectrum))

def estimate_threshold(signal: np.ndarray, perc: int = 95) -> float:
    """Estimates threshold for peak detection."""
    return np.percentile(signal, perc)

def find_Rpeaks(signal: np.ndarray) -> np.ndarray:
    """Finds R peaks in ECG signal."""
    threshold = estimate_threshold(signal)
    peaks = []
    is_peak = False
    current_peak_value = -np.inf
    current_peak_index = 0

    for i, value in enumerate(signal):
        if value > threshold:
            if not is_peak or value > current_peak_value:
                current_peak_value = value
                current_peak_index = i
            is_peak = True
        else:
            if is_peak:
                peaks.append(current_peak_index)
                is_peak = False
    return np.array(peaks)

def count_rpeaks_in_windows(signal: np.ndarray, r_peaks: np.ndarray, samp_rate: int, window_duration: float = 10.0) -> List[int]:
    """Counts R peaks in time windows."""
    signal_duration = len(signal) / samp_rate
    window_count = int(signal_duration / window_duration)
    r_peak_counts = [0] * window_count

    for r_peak in r_peaks:
        window_index = int(r_peak / (samp_rate * window_duration))
        if 0 <= window_index < window_count:
            r_peak_counts[window_index] += 1

    return r_peak_counts

def compute_bpm(r_peaks: np.ndarray, samp_rate: int, window_size: int = 15) -> List[float]:
    """Computes BPM in time windows."""
    signal_duration = r_peaks[-1] / samp_rate
    num_windows = int(signal_duration // window_size)

    return [
        sum((i * window_size * samp_rate <= p < (i + 1) * window_size * samp_rate) for p in r_peaks) * (60 / window_size)
        for i in range(num_windows)
    ]

# Higher-order functions for processing pipeline
def create_processing_pipeline(fs: int, cutoff_frequency: float = 0.5) -> Callable:
    """Creates a processing pipeline for ECG data."""
    return compose(
        find_Rpeaks,
        apply_with_params(remove_baseline_wander_fft, fs, cutoff_frequency),
        center_ecg_signal
    )

def create_bpm_analysis(fs: int, window_size: int = 15) -> Callable:
    """Creates BPM analysis function."""
    return lambda r_peaks: compute_bpm(r_peaks, fs, window_size)

def create_window_counter(fs: int, window_duration: float = 10.0) -> Callable:
    """Creates window counter function."""
    return lambda signal_and_peaks: count_rpeaks_in_windows(
        signal_and_peaks[0], signal_and_peaks[1], fs, window_duration
    )

# Main processing function
def process_ecg_record(database_folder: str, record_name: str, max_samples: int = 6000000) -> dict:
    """
    Main function to process ECG record using functional programming approach.
    Returns dictionary with all results.
    """
    # Build paths
    database_path = os.path.join(database_folder, record_name)
    files = os.listdir(database_path)

    # Find ECG file
    ecg_base_name = find_ecg_file(files)
    if not ecg_base_name:
        raise FileNotFoundError(f"No ECG file found in {database_path}")

    record_path = os.path.join(database_path, ecg_base_name)

    # Load and process data
    ecg_data, fs = load_record_data(record_path)

    # Create processing pipeline
    truncate_fn = truncate_signal(max_samples)
    process_fn = create_processing_pipeline(fs)
    bpm_analysis_fn = create_bpm_analysis(fs)

    # Process signal through pipeline
    processed_signal = pipe(
        ecg_data,
        truncate_fn,
        center_ecg_signal,
        apply_with_params(remove_baseline_wander_fft, fs, 0.5)
    )

    # Find R peaks
    r_peaks = find_Rpeaks(processed_signal)

    # Calculate metrics
    r_peak_counts = count_rpeaks_in_windows(processed_signal, r_peaks, fs)
    bpm_values = bpm_analysis_fn(r_peaks)
    average_bpm = np.mean(bpm_values) if bpm_values else 0

    return {
        'fs': fs,
        'signal_length': len(ecg_data),
        'processed_signal': processed_signal,
        'r_peaks': r_peaks,
        'r_peak_counts': [count * 4 for count in r_peak_counts],
        'bpm_values': bpm_values,
        'average_bpm': average_bpm,
        'record_name': record_name
    }

# Evaluation functions (functional style)
def calculate_detection_accuracy(true_peaks: np.ndarray, detected_peaks: np.ndarray, threshold: float = 0.0001) -> Tuple[int, int, int, int]:
    """Calculates detection accuracy in functional style."""
    def is_detected(true_peak):
        return any(
            abs(1 - (true_peak + 1) / (detected_peak + 1)) < threshold
            for detected_peak in detected_peaks
        )

    correct_detections = sum(1 for peak in true_peaks if is_detected(peak))
    total_true = len(true_peaks)
    missed = total_true - correct_detections
    total_detected = len(detected_peaks)

    return correct_detections, missed, total_true, total_detected

def test_detection_on_file(file_path: str, sampfrom: int = 0, nsamp: int = 100000, threshold: float = 0.0001) -> Tuple[int, int, int, int]:
    """Tests detection on a specific file."""
    # Load data
    signal, _ = wfdb.rdsamp(file_path, channels=None, sampfrom=sampfrom, sampto=sampfrom+nsamp)
    true_peaks = wfdb.rdann(file_path, extension="atr", sampfrom=sampfrom, sampto=sampfrom+nsamp).sample

    # Process signal
    processed_signal = pipe(
        signal[:, 0],
        center_ecg_signal,
        find_Rpeaks
    )

    return calculate_detection_accuracy(true_peaks, processed_signal, threshold)

# Automated testing functions
def find_all_ecg_records(database_folder: str) -> List[str]:
    """Finds all ECG record directories in the database folder."""
    records = []
    if not os.path.exists(database_folder):
        return records

    for item in os.listdir(database_folder):
        item_path = os.path.join(database_folder, item)
        if os.path.isdir(item_path):
            # Check if directory contains ECG files
            files = os.listdir(item_path)
            if any(f.endswith("_ECG.dat") for f in files):
                records.append(item)

    return sorted(records)

def test_single_record(database_folder: str, record_name: str, test_config: dict) -> dict:
    """Tests a single ECG record and returns comprehensive results."""
    try:
        # Process the record
        results = process_ecg_record(
            database_folder,
            record_name,
            test_config.get('max_samples', 6000000)
        )

        # Test detection accuracy if annotation file exists
        detection_results = None
        record_path = os.path.join(database_folder, record_name)
        ecg_files = [f for f in os.listdir(record_path) if f.endswith("_ECG.dat")]

        if ecg_files:
            base_name = ecg_files[0][:-4]
            full_path = os.path.join(record_path, base_name)

            try:
                detection_results = test_detection_on_file(
                    full_path,
                    sampfrom=test_config.get('sampfrom', 0),
                    nsamp=test_config.get('nsamp', 100000),
                    threshold=test_config.get('threshold', 0.0001)
                )
            except Exception as e:
                print(f"Warning: Could not test detection for {record_name}: {e}")

        return {
            'record_name': record_name,
            'success': True,
            'fs': results['fs'],
            'signal_length': results['signal_length'],
            'r_peaks_count': len(results['r_peaks']),
            'average_bpm': results['average_bpm'],
            'bpm_values': results['bpm_values'],
            'detection_accuracy': detection_results,
            'error': None
        }

    except Exception as e:
        return {
            'record_name': record_name,
            'success': False,
            'error': str(e),
            'fs': None,
            'signal_length': None,
            'r_peaks_count': None,
            'average_bpm': None,
            'bpm_values': None,
            'detection_accuracy': None
        }

def run_batch_tests(database_folder: str, test_config: dict = None, record_filter: Callable = None) -> List[dict]:
    """Runs automated tests on all ECG records in the database folder."""
    if test_config is None:
        test_config = {
            'max_samples': 6000000,
            'sampfrom': 0,
            'nsamp': 100000,
            'threshold': 0.0001
        }

    # Find all records
    all_records = find_all_ecg_records(database_folder)

    # Apply filter if provided
    if record_filter:
        all_records = [record for record in all_records if record_filter(record)]

    print(f"Found {len(all_records)} ECG records to test")
    print(f"Records: {all_records}")

    results = []
    for i, record_name in enumerate(all_records, 1):
        print(f"\nProcessing {i}/{len(all_records)}: {record_name}")
        result = test_single_record(database_folder, record_name, test_config)
        results.append(result)

        # Print quick summary
        if result['success']:
            print(f"  ✓ Success - BPM: {result['average_bpm']:.1f}, R-peaks: {result['r_peaks_count']}")
            if result['detection_accuracy']:
                correct, missed, total_true, total_detected = result['detection_accuracy']
                accuracy = (correct / total_true * 100) if total_true > 0 else 0
                print(f"  ✓ Detection accuracy: {accuracy:.1f}% ({correct}/{total_true})")
        else:
            print(f"  ✗ Failed: {result['error']}")

    return results

def generate_test_report(results: List[dict], output_file: str = None) -> str:
    """Generates a comprehensive test report."""
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]

    report_lines = [
        "=" * 60,
        "ECG ANALYSIS TEST REPORT",
        "=" * 60,
        f"Total tests: {len(results)}",
        f"Successful: {len(successful_tests)}",
        f"Failed: {len(failed_tests)}",
        f"Success rate: {len(successful_tests)/len(results)*100:.1f}%",
        "",
        "SUCCESSFUL TESTS:",
        "-" * 30
    ]

    if successful_tests:
        for result in successful_tests:
            report_lines.extend([
                f"Record: {result['record_name']}",
                f"  Sampling rate: {result['fs']} Hz",
                f"  Signal length: {result['signal_length']} samples",
                f"  R-peaks detected: {result['r_peaks_count']}",
                f"  Average BPM: {result['average_bpm']:.2f}",
            ])

            if result['detection_accuracy']:
                correct, missed, total_true, total_detected = result['detection_accuracy']
                accuracy = (correct / total_true * 100) if total_true > 0 else 0
                sensitivity = (correct / total_detected * 100) if total_detected > 0 else 0
                report_lines.extend([
                    f"  Detection accuracy: {accuracy:.1f}%",
                    f"  Sensitivity: {sensitivity:.1f}%",
                    f"  True positives: {correct}",
                    f"  False negatives: {missed}",
                    f"  Total detected: {total_detected}"
                ])
            report_lines.append("")

    if failed_tests:
        report_lines.extend([
            "",
            "FAILED TESTS:",
            "-" * 30
        ])
        for result in failed_tests:
            report_lines.extend([
                f"Record: {result['record_name']}",
                f"  Error: {result['error']}",
                ""
            ])

    # Summary statistics
    if successful_tests:
        bpm_values = [r['average_bpm'] for r in successful_tests if r['average_bpm']]
        r_peak_counts = [r['r_peaks_count'] for r in successful_tests if r['r_peaks_count']]

        report_lines.extend([
            "",
            "SUMMARY STATISTICS:",
            "-" * 30,
            f"Average BPM across all records: {np.mean(bpm_values):.2f} ± {np.std(bpm_values):.2f}",
            f"BPM range: {min(bpm_values):.1f} - {max(bpm_values):.1f}",
            f"Average R-peaks per record: {np.mean(r_peak_counts):.0f} ± {np.std(r_peak_counts):.0f}",
        ])

        # Detection accuracy summary
        detection_results = [r['detection_accuracy'] for r in successful_tests if r['detection_accuracy']]
        if detection_results:
            accuracies = [(correct/total_true*100) for correct, missed, total_true, total_detected in detection_results if total_true > 0]
            report_lines.extend([
                f"Average detection accuracy: {np.mean(accuracies):.1f}% ± {np.std(accuracies):.1f}%",
                f"Detection accuracy range: {min(accuracies):.1f}% - {max(accuracies):.1f}%"
            ])

    report = "\n".join(report_lines)

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_file}")

    return report

def create_record_filter(patterns: List[str] = None, exclude_patterns: List[str] = None) -> Callable:
    """Creates a filter function for record names."""
    def filter_func(record_name: str) -> bool:
        # Include patterns
        if patterns:
            if not any(pattern in record_name for pattern in patterns):
                return False

        # Exclude patterns
        if exclude_patterns:
            if any(pattern in record_name for pattern in exclude_patterns):
                return False

        return True

    return filter_func

# Usage examples and main function
def main():
    """Main execution function with automated testing capabilities."""
    # Configuration for testing
    test_config = {
        'max_samples': 6000000,      # Limit signal length
        'sampfrom': 0,               # Start sample for annotation testing
        'nsamp': 100000,             # Number of samples for annotation testing
        'threshold': 0.0001          # Detection accuracy threshold
    }

    database_folder = "source1"

    print("ECG Analysis Automation System")
    print("=" * 40)

    # Option 1: Test all records
    print("\n1. Testing all records...")
    all_results = run_batch_tests(database_folder, test_config)

    # Option 2: Test filtered records (example: only records containing "100")
    print("\n2. Testing filtered records (containing '100')...")
    record_filter = create_record_filter(patterns=["100"])
    filtered_results = run_batch_tests(database_folder, test_config, record_filter)

    # Generate reports
    print("\n3. Generating reports...")
    report = generate_test_report(all_results, "ecg_test_report.txt")
    print(report)

    # Return results for further analysis
    return {
        'all_results': all_results,
        'filtered_results': filtered_results,
        'report': report
    }

# Additional utility functions for analysis
def compare_algorithms(database_folder: str, algorithms: dict, test_config: dict = None) -> dict:
    """Compare multiple detection algorithms on the same dataset."""
    if test_config is None:
        test_config = {'max_samples': 6000000, 'sampfrom': 0, 'nsamp': 100000, 'threshold': 0.0001}

    records = find_all_ecg_records(database_folder)
    comparison_results = {alg_name: [] for alg_name in algorithms.keys()}

    for record_name in records:
        print(f"Comparing algorithms on {record_name}...")

        for alg_name, alg_func in algorithms.items():
            try:
                result = test_single_record(database_folder, record_name, test_config)
                comparison_results[alg_name].append(result)
            except Exception as e:
                print(f"Error with {alg_name} on {record_name}: {e}")

    return comparison_results

if __name__ == "__main__":
    main()
