import os
import matplotlib.pyplot as plt

def save_plot(plot_func, filename, *args, **kwargs):
    """
    Saves a matplotlib plot to the `plots/` folder.
    """
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    filepath = os.path.join(plot_dir, filename)
    plot_func(*args, **kwargs)
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")
    plt.close()

def log_message(message):
    """
    Logs a message to the console and optionally a log file.
    """
    print(f"[LOG] {message}")
