import pandas as pd
import matplotlib.pyplot as plt

def plot_training_curves(csv_path="eval_metrics_history.csv"):
    """
    Reads the logged evaluation metrics and plots the Success Rate 
    and Error Rate curves per scenario across timesteps.
    """
    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    scenarios = df['scenario'].unique()
    
    for scenario in scenarios:
        data = df[df['scenario'] == scenario]
        # .to_numpy() to avoid indexing issues with pandas/matplotlib integration
        ax1.plot(data['step'].to_numpy(), data['success_rate'].to_numpy(), label=scenario, linewidth=2)
        ax2.plot(data['step'].to_numpy(), data['error_rate'].to_numpy(), label=scenario, linewidth=2)

    # Success rate plot formatting
    ax1.set_title("Success Rate Evolution by Scenario")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Success Rate (0.0 to 1.0)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Error rate plot formatting (Collision, OOB, Timeout)
    ax2.set_title("Error Rate Evolution (Collision + OOB + Timeout)")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Error Rate (0.0 to 1.0)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Plots saved successfully as 'training_curves.png'")
    plt.show()

if __name__ == "__main__":
    plot_training_curves()
