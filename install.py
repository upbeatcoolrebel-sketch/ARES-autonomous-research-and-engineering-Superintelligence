import os
import sys
import subprocess
import json
from typing import Dict, Optional

# Default configuration
DEFAULT_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "vocab_size": 20000,
    "d_model": 256,
    "num_layers": 4,
    "num_heads": 8,
    "d_ff": 1024,
    "max_seq_len": 128,
    "batch_size": 8,
    "epochs": 20,
    "patience": 7,
    "data_sources": ["wikitext", "rss"],
    "rss_feeds": [
        "http://feeds.feedburner.com/analyticsinsight/ijEZ",
        "https://www.kdnuggets.com/feed",
        "https://blog.google/technology/ai/rss/",
        "https://marekrei.com/blog/feed",
        "https://paperswithcode.com/rss",
        "https://www.artificialintelligence-news.com/feed/",
        "https://machinelearningmastery.com/feed/"
    ],
    "max_samples": 10000,
    "beam_width": 7,
    "learning_rate": 0.0005,
    "weight_decay": 0.01,
    "save_path": "/content/drive/MyDrive" if "google.colab" in sys.modules else "./checkpoints"
}

CONFIG_FILE = "ares_config.json"

def install_dependencies():
    """Install required Python packages."""
    print("Installing dependencies...")
    packages = [
        "torch",
        "datasets",
        "feedparser",
        "requests",
        "beautifulsoup4"
    ]
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            return False
    return True

def load_config() -> Dict:
    """Load existing configuration or return default."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config: Dict):
    """Save configuration to file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {CONFIG_FILE}")

def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default value."""
    if default:
        return input(f"{prompt} [default: {default}]: ") or default
    return input(f"{prompt}: ")

def configure_settings():
    """Interactive configuration of settings."""
    config = load_config()
    
    print("\n=== ARES Configuration Setup ===")
    print("Leave blank to keep default values. Press Enter to skip to the end.\n")
    
    # Device and basic parameters
    config["device"] = get_user_input("Device (cuda/cpu)", config["device"])
    config["vocab_size"] = int(get_user_input("Vocabulary size", str(config["vocab_size"]))) or config["vocab_size"]
    config["d_model"] = int(get_user_input("Model dimension", str(config["d_model"]))) or config["d_model"]
    config["num_layers"] = int(get_user_input("Number of layers", str(config["num_layers"]))) or config["num_layers"]
    config["num_heads"] = int(get_user_input("Number of attention heads", str(config["num_heads"]))) or config["num_heads"]
    config["d_ff"] = int(get_user_input("Feed-forward dimension", str(config["d_ff"]))) or config["d_ff"]
    config["max_seq_len"] = int(get_user_input("Max sequence length", str(config["max_seq_len"]))) or config["max_seq_len"]
    
    # Training parameters
    config["batch_size"] = int(get_user_input("Batch size", str(config["batch_size"]))) or config["batch_size"]
    config["epochs"] = int(get_user_input("Number of epochs", str(config["epochs"]))) or config["epochs"]
    config["patience"] = int(get_user_input("Patience for early stopping", str(config["patience"]))) or config["patience"]
    config["learning_rate"] = float(get_user_input("Learning rate", str(config["learning_rate"]))) or config["learning_rate"]
    config["weight_decay"] = float(get_user_input("Weight decay", str(config["weight_decay"]))) or config["weight_decay"]
    
    # Data and generation
    config["max_samples"] = int(get_user_input("Max samples", str(config["max_samples"]))) or config["max_samples"]
    config["beam_width"] = int(get_user_input("Beam width for generation", str(config["beam_width"]))) or config["beam_width"]
    data_sources = get_user_input("Data sources (wikitext/rss, comma-separated)", ",".join(config["data_sources"])).split(",")
    config["data_sources"] = [s.strip() for s in data_sources if s.strip()]
    if "rss" in config["data_sources"]:
        new_feeds = get_user_input("RSS feed URLs (comma-separated)", ",".join(config["rss_feeds"])).split(",")
        config["rss_feeds"] = [f.strip() for f in new_feeds if f.strip()]
    
    # Save path
    config["save_path"] = get_user_input("Checkpoint save path", config["save_path"])
    
    save_config(config)
    return config

def update_ares_script(config: Dict):
    """Update ares.py with user-defined settings."""
    with open("ares.py", "r") as f:
        lines = f.readlines()
    
    # Update hyperparameters in ares.py
    for i, line in enumerate(lines):
        if line.strip().startswith("vocab_size ="):
            lines[i] = f"vocab_size = {config['vocab_size']}\n"
        elif line.strip().startswith("d_model ="):
            lines[i] = f"d_model = {config['d_model']}\n"
        elif line.strip().startswith("num_layers ="):
            lines[i] = f"num_layers = {config['num_layers']}\n"
        elif line.strip().startswith("num_heads ="):
            lines[i] = f"num_heads = {config['num_heads']}\n"
        elif line.strip().startswith("d_ff ="):
            lines[i] = f"d_ff = {config['d_ff']}\n"
        elif line.strip().startswith("max_seq_len ="):
            lines[i] = f"max_seq_len = {config['max_seq_len']}\n"
        elif line.strip().startswith("batch_size ="):
            lines[i] = f"batch_size = {config['batch_size']}\n"
        elif line.strip().startswith("epochs ="):
            lines[i] = f"epochs = {config['epochs']}\n"
        elif line.strip().startswith("patience ="):
            lines[i] = f"patience = {config['patience']}\n"
        elif line.strip().startswith("learning_rate ="):
            lines[i] = f"learning_rate = {config['learning_rate']}\n"  # Add to ares.py if not present
        elif line.strip().startswith("weight_decay ="):
            lines[i] = f"weight_decay = {config['weight_decay']}\n"  # Add to ares.py if not present
        elif line.strip().startswith("max_samples ="):
            lines[i] = f"max_samples = {config['max_samples']}\n"
        elif line.strip().startswith("beam_width ="):
            lines[i] = f"beam_width = {config['beam_width']}\n"
    
    with open("ares.py", "w") as f:
        f.writelines(lines)
    print("ares.py updated with new settings.")

def main():
    print("=== ARES Installation and Configuration ===")
    
    # Install dependencies
    if not install_dependencies():
        print("Installation failed. Please check the errors and try again.")
        return
    
    # Configure settings
    print("\nConfiguring ARES settings...")
    config = configure_settings()
    
    # Update ares.py with new settings
    update_ares_script(config)
    
    print("\nInstallation and configuration complete!")
    print(f"Run `python ares.py` to start ARES with your settings.")
    print(f"Configuration saved in {CONFIG_FILE}. Edit it manually if needed.")

if __name__ == "__main__":
    try:
        import torch  # Test import to ensure dependencies are available
        main()
    except ImportError:
        print("Error: Required libraries not found. Running dependency installation...")
        if install_dependencies():
            import torch
            main()
        else:
            print("Failed to resolve dependencies. Please install manually: pip install torch datasets feedparser requests beautifulsoup4")