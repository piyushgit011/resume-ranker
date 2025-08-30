# Advanced Resume Ranker with Multiple AI Agents
# Import the new advanced application
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.main_app import main

if __name__ == "__main__":
    main()
