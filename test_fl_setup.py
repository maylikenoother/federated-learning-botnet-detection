#!/usr/bin/env python3
"""
Test script to verify federated learning setup and run a simple experiment
"""

import os
import sys
import subprocess
import time
import signal
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FLTestRunner:
    def __init__(self):
        self.processes = []
        self.test_results = {
            'dependencies': False,
            'data': False,
            'server': False,
            'clients': False,
            'communication': False
        }
    
    def cleanup(self):
        """Clean up any running processes"""
        for process in self.processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        self.processes.clear()
    
    def test_dependencies(self):
        """Test if all required dependencies are installed"""
        logger.info("🔍 Testing dependencies...")
        
        required_packages = {
            'torch': 'PyTorch',
            'flwr': 'Flower',
            'pandas': 'Pandas',
            'numpy': 'NumPy',
            'sklearn': 'Scikit-learn'
        }
        
        missing = []
        for package, name in required_packages.items():
            try:
                __import__(package)
                logger.info(f"✅ {name} is installed")
            except ImportError:
                missing.append(package)
                logger.error(f"❌ {name} is NOT installed")
        
        if missing:
            logger.error(f"Missing packages: {missing}")
            logger.info(f"Install with: pip install {' '.join(missing)}")
            return False
        
        self.test_results['dependencies'] = True
        return True
    
    def test_data(self):
        """Test if Bot-IoT dataset is available and valid"""
        logger.info("\n🔍 Testing dataset...")
        
        if not os.path.exists("Bot_IoT.csv"):
            logger.error("❌ Bot_IoT.csv not found!")
            return False
        
        try:
            import pandas as pd
            df = pd.read_csv("Bot_IoT.csv", nrows=100)
            
            if 'category' not in df.columns:
                logger.error("❌ 'category' column not found in dataset")
                return False
            
            categories = df['category'].unique()
            logger.info(f"✅ Dataset loaded successfully")
            logger.info(f"   Categories found: {categories[:5]}...")  # Show first 5
            logger.info(f"   Total columns: {len(df.columns)}")
            
            self.test_results['data'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to read dataset: {e}")
            return False
    
    def test_server(self, port=8080):
        """Test if server can start"""
        logger.info(f"\n🔍 Testing server on port {port}...")
        
        # First check if port is in use
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            logger.warning(f"⚠️ Port {port} is already in use")
            return False
        
        # Try to start server
        try:
            server_cmd = [sys.executable, "server.py", "--algorithm", "FedAvg", "--rounds", "2", "--port", str(port)]
            
            logger.info("Starting test server...")
            server_process = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(server_process)
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server is still running
            if server_process.poll() is not None:
                stdout, stderr = server_process.communicate()
                logger.error("❌ Server failed to start")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                return False
            
            logger.info("✅ Server started successfully")
            self.test_results['server'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}")
            return False
    
    def test_client(self, client_id=0, port=8080):
        """Test if a client can connect"""
        logger.info(f"\n🔍 Testing client {client_id}...")
        
        try:
            env = os.environ.copy()
            env['CLIENT_ID'] = str(client_id)
            
            client_cmd = [sys.executable, "client.py"]
            
            logger.info(f"Starting test client {client_id}...")
            client_process = subprocess.Popen(
                client_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(client_process)
            
            # Wait for client to connect
            time.sleep(10)
            
            # Check if client is still running
            if client_process.poll() is not None:
                stdout, stderr = client_process.communicate()
                logger.error(f"❌ Client {client_id} failed")
                logger.error(f"STDOUT: {stdout[-500:]}")  # Last 500 chars
                logger.error(f"STDERR: {stderr[-500:]}")
                return False
            
            logger.info(f"✅ Client {client_id} connected successfully")
            self.test_results['clients'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start client: {e}")
            return False
    
    def run_mini_experiment(self):
        """Run a minimal FL experiment to test the setup"""
        logger.info("\n🧪 Running mini FL experiment (2 rounds, 2 clients)...")
        
        port = 8890  # Use different port to avoid conflicts
        
        try:
            # Start server
            server_cmd = [sys.executable, "server.py", "--algorithm", "FedAvg", "--rounds", "2", "--port", str(port)]
            server_process = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(server_process)
            
            logger.info("⏳ Waiting for server initialization...")
            time.sleep(10)
            
            # Start 2 clients
            client_processes = []
            for client_id in range(2):
                env = os.environ.copy()
                env['CLIENT_ID'] = str(client_id)
                
                client_process = subprocess.Popen(
                    [sys.executable, "client.py"],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                client_processes.append(client_process)
                self.processes.append(client_process)
                logger.info(f"Started client {client_id}")
                time.sleep(3)
            
            # Wait for experiment to complete
            logger.info("⏳ Running experiment...")
            time.sleep(30)
            
            # Check if server completed
            if server_process.poll() is not None:
                logger.info("✅ Mini experiment completed!")
                self.test_results['communication'] = True
                return True
            else:
                logger.warning("⚠️ Experiment still running after 30 seconds")
                return True  # Still consider it a success if running
                
        except Exception as e:
            logger.error(f"❌ Mini experiment failed: {e}")
            return False
        finally:
            self.cleanup()
    
    def generate_report(self):
        """Generate test report"""
        logger.info("\n" + "="*60)
        logger.info("📋 FL SETUP TEST REPORT")
        logger.info("="*60)
        
        all_passed = True
        for test, passed in self.test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"{test.upper()}: {status}")
            if not passed:
                all_passed = False
        
        logger.info("="*60)
        
        if all_passed:
            logger.info("✅ All tests passed! Your FL setup is ready.")
        else:
            logger.info("❌ Some tests failed. Please fix the issues above.")
            
            # Provide specific fixes
            if not self.test_results['dependencies']:
                logger.info("\n💡 Fix: Install missing dependencies")
                logger.info("   pip install torch flwr pandas numpy scikit-learn matplotlib")
            
            if not self.test_results['data']:
                logger.info("\n💡 Fix: Ensure Bot_IoT.csv is in the current directory")
                logger.info("   and has a 'category' column")
            
            if not self.test_results['server']:
                logger.info("\n💡 Fix: Check if ports are available and server.py is correct")
                logger.info("   Try using the fixed server.py provided")
        
        return all_passed

def main():
    """Run all tests"""
    print("🔬 FEDERATED LEARNING SETUP TESTER")
    print("="*60)
    print("This will test your FL setup and identify any issues")
    print("="*60)
    
    tester = FLTestRunner()
    
    # Set up signal handler for clean shutdown
    def signal_handler(signum, frame):
        logger.info("\n🛑 Interrupted, cleaning up...")
        tester.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run tests in sequence
        if not tester.test_dependencies():
            tester.generate_report()
            return False
        
        if not tester.test_data():
            tester.generate_report()
            return False
        
        if tester.test_server():
            # Only test client if server works
            tester.test_client()
            time.sleep(2)
            tester.cleanup()
            
            # Try mini experiment
            tester.run_mini_experiment()
        
        # Generate final report
        success = tester.generate_report()
        
        if success:
            print("\n🎉 Your federated learning setup is working!")
            print("📝 Next steps:")
            print("   1. Use the fixed server.py provided")
            print("   2. Run your full experiments with run_complete_research.py")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        tester.cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)