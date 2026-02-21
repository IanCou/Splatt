import os
import logging
import paramiko

logger = logging.getLogger(__name__)

class ComputeBridge:
    def __init__(self):
        self.host = os.getenv("VASTAI_HOST", "mock.vast.ai")
        self.port = int(os.getenv("VASTAI_PORT", "2222"))
        self.username = os.getenv("VASTAI_USER", "root")
        self.key_filename = os.getenv("VASTAI_SSH_KEY", "~/.ssh/id_rsa")
        
    def train_splat(self, video_path: str, output_name: str) -> bool:
        """
        Connects via SSH to Vast.ai instance and triggers ns-train splatfacto
        (or custom gsplat pipeline).
        """
        if self.host.startswith("mock"):
            logger.info(f"Mocking Vast.ai SSH connection to train {video_path} into {output_name}.splat")
            return True
            
        try:
            # Set up SSH client
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # This will fail unless user properly configures keys and env vars.
            # Catching gracefully for scaffolding demo.
            logger.info(f"Connecting to Vast.ai at {self.host}:{self.port}")
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                key_filename=os.path.expanduser(self.key_filename),
                timeout=10
            )

            # E.g., ns-train splatfacto --data /workspace/data/{video_name}
            # For this scaffolding we'll run a dummy echo command
            command = f"echo 'Training {video_path} using ns-train splatfacto' > /workspace/training.log && sleep 5"
            logger.info(f"Executing remote command: {command}")
            
            stdin, stdout, stderr = client.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status == 0:
                logger.info(f"Remote training job submitted successfully.")
                return True
            else:
                logger.error(f"Remote command failed with status {exit_status}: {stderr.read().decode('utf-8')}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to trigger Vast.ai compute: {e}")
            return False
        finally:
            try:
                client.close()
            except:
                pass

compute_bridge = ComputeBridge()
