import time
import stabilizer_qec
import numpy as np
import matplotlib.pyplot as plt

def run_benchmarks():
    print("==================================================")
    print("Starting QEC Surface Code & Union-Find Decoder Benchmarks")
    print("Mode: Pure Data Noise (Perfect Measurements, 1 Round)")
    print("==================================================")

    # We test distances 3, 5, 7
    distances = [3, 5, 7]
    # Physical error rates to scan
    p_values = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]
    num_trials = 2000  # Higher number of trials to ensure statistical significance
    
    results = {d: [] for d in distances}

    start_time = time.time()

    print(f"{'Distance (d)':<15}{'Physical Error (p)':<22}{'Logical Error (p_L)':<22}{'Failures / Trials':<20}")
    print("-" * 79)

    for d in distances:
        code = stabilizer_qec.RotatedSurfaceCode(d)
        for p in p_values:
            failures = 0
            for _ in range(num_trials):
                if code.simulate_data_noise(p):
                    failures += 1
            
            logical_p = failures / num_trials
            results[d].append(logical_p)
            print(f"{d:<15}{p:<22.4f}{logical_p:<22.4f}{failures}/{num_trials:<20}")

    total_time = time.time() - start_time
    print("-" * 79)
    print(f"Benchmark completed in {total_time:.2f} seconds.")
    print("==================================================")

    # Plotting the threshold curve
    plt.figure(figsize=(10, 7))
    
    colors = {3: '#e06666', 5: '#3d85c6', 7: '#6aa84f'}
    markers = {3: 'o', 5: 's', 7: '^'}

    for d in distances:
        plt.plot(
            p_values, 
            results[d], 
            label=f"d = {d}", 
            color=colors[d], 
            marker=markers[d], 
            linewidth=2, 
            markersize=7
        )

    # Plot the breakeven line y = x
    plt.plot(p_values, p_values, '--', color='#7f8c8d', label="Breakeven (p_L = p)", alpha=0.8)

    plt.title("QEC Rotated Surface Code Threshold Curve (Pure Data Noise)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Physical Error Rate (p)", fontsize=12)
    plt.ylabel("Logical Error Rate ($p_L$)", fontsize=12)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(min(p_values) * 0.8, max(p_values) * 1.2)
    plt.ylim(min(p_values) * 0.01, 1.0)
    plt.legend(fontsize=11, loc='lower right')
    
    plot_path = "data_threshold_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Threshold plot saved successfully as '{plot_path}'.")

if __name__ == "__main__":
    run_benchmarks()
