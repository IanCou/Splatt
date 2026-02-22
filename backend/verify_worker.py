import os
from dotenv import load_dotenv
import paramiko

load_dotenv()

def test_ssh_connection():
    host = os.getenv("WORKER_HOST")
    user = os.getenv("WORKER_USER")
    port = int(os.getenv("WORKER_PORT", "22"))
    key_path = os.getenv("WORKER_KEY_PATH")
    
    if not all([host, user, key_path]):
        print("Error: WORKER_HOST, WORKER_USER, or WORKER_KEY_PATH not set in .env")
        return

    print(f"Testing connection to {host}:{port} as {user} with key {key_path}...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=user, key_filename=key_path)
        print("✅ SSH Connection Successful!")

        
        stdin, stdout, stderr = ssh.exec_command("ls -l /workspace")
        print("Output of 'ls -l /workspace':")
        print(stdout.read().decode())
        
    except Exception as e:
        print(f"❌ SSH Connection Failed: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    test_ssh_connection()
