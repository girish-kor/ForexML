"""ForexML entry point."""

import logging
import sys

from config import load_system_config
from system import ForexMLTradingSystem

# 1. Setup logging to output to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ForexML_Main")

def main():
    """Main entry point for the ForexML trading system."""
    logger.info("Initializing ForexML...")
    
    try:
        # Load Configuration
        config = load_system_config()
        logger.info("Configuration loaded successfully.")
        
        # Initialize Trading System
        system = ForexMLTradingSystem(config)
        logger.info("Trading system initialized. Starting execution loop...")
        
        # Run System
        system.run()
        
    except KeyboardInterrupt:
        # 2. Graceful Shutdown on Ctrl+C
        logger.info("KeyboardInterrupt detected. Shutting down ForexML gracefully...")
        # If your system class has a cleanup/shutdown method, call it here:
        # system.shutdown() 
        sys.exit(0)
        
    except Exception as e:
        # 3. Catch-all for unexpected crashes to log the stack trace
        logger.exception(f"A critical error occurred during execution: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()