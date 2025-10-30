import subprocess
import sys
import os

def install_requirements():
    """Install required packages for dashboard"""
    packages = [
        'streamlit',
        'plotly',
        'pandas'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting DeepFake Detection Dashboard...")
    print("📊 Dashboard will open in your browser")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "="*50)
    
    # Run streamlit
    os.system("streamlit run dashboard.py")

if __name__ == "__main__":
    print("🎭 DeepFake Detection Dashboard Setup")
    print("="*40)
    
    # Install requirements
    install_requirements()
    
    # Run dashboard
    run_dashboard()