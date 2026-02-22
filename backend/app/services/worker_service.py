import os
import re
import paramiko
from pathlib import Path
import logging

logger = logging.getLogger(__name__)




def run_remote_pipeline(task_id: str, local_video_path: str = None, filebin_url: str = None, on_progress=None):
    """
    SSH Orchestration Background Task.

    1. Connect to the remote worker via SSH (with keepalives for long training).
    2. Either upload the video via SFTP (legacy) OR download from Filebin URL (faster).
    3. Run Nerf4Dgsplat.py (already present on the remote) and stream stdout
       to parse training-iteration progress.
    4. Download result PLY files + transforms.json.
    5. Clean up remote temp files.

    Parameters
    ----------
    task_id : str
        Unique task identifier
    local_video_path : str, optional
        Path to local video file (for SFTP upload)
    filebin_url : str, optional
        Filebin URL to download video from (faster alternative)
    on_progress : callable, optional
        Progress callback function
    """

    def report(status: str, progress: int):
        if on_progress:
            on_progress(task_id, status, progress)

    # Read config here (after load_dotenv() has run in main.py)
    host = os.getenv("WORKER_HOST", "YOUR_WORKER_IP")
    user = os.getenv("WORKER_USER", "root")
    port = int(os.getenv("WORKER_PORT", "22"))
    key_path = os.getenv("WORKER_KEY_PATH", "/path/to/your/id_rsa")
    project_dir = os.getenv("WORKER_PROJECT_DIR", "/workspace")

    remote_video_path = f"{project_dir}/{task_id}_video.mp4"
    remote_output_dir = f"{project_dir}/{task_id}_results"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = None

    try:
        # ── 1. Connect ────────────────────────────────────────────────
        report("Connecting to worker...", 10)
        logger.info("Connecting to worker %s:%s ...", host, port)

        ssh.connect(
            host,
            port=port,
            username=user,
            key_filename=key_path,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )

        # Keep the connection alive during long COLMAP / training runs
        transport = ssh.get_transport()
        if transport:
            transport.set_keepalive(60)

        sftp = ssh.open_sftp()
        logger.info("SSH connection established.")

        # ── 2. Get video to worker (either upload or download from Filebin) ──
        if filebin_url:
            # Download from Filebin URL (much faster than SFTP)
            report("Worker downloading video from Filebin...", 15)
            logger.info("Remote downloading from %s → %s", filebin_url, remote_video_path)

            download_cmd = f"curl -s -o {remote_video_path} '{filebin_url}'"
            logger.info("Executing remote download: %s", download_cmd)

            stdin, stdout, stderr = ssh.exec_command(download_cmd, timeout=600)
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                error_msg = stderr.read().decode().strip()
                logger.error("Remote download failed: %s", error_msg)
                report(f"Download failed: {error_msg[:100]}", -1)
                return

            logger.info("Remote download complete.")
            report("Video downloaded on worker", 30)

        elif local_video_path:
            # Legacy: Upload via SFTP (slower)
            report("Uploading video to worker...", 15)
            logger.info("Uploading %s → %s", local_video_path, remote_video_path)

            file_size = os.path.getsize(local_video_path)

            def upload_callback(transferred, total):
                pct = 15 + int((transferred / total) * 15)  # 15-30 %
                report(f"Uploading video… {transferred * 100 // total}%", pct)

            sftp.put(local_video_path, remote_video_path, callback=upload_callback)
            logger.info("Video upload complete.")
        else:
            raise ValueError("Either local_video_path or filebin_url must be provided")

        # ── 3. Run Nerf4Dgsplat.py on the remote ─────────────────────
        #    The script is already deployed at project_dir.
        report("Starting Gaussian Splatting pipeline…", 30)

        remote_cmd = (
            f"cd {project_dir} && "
            f"/opt/miniforge3/envs/nerf/bin/python Nerf4Dgsplat.py "
            f"--video {remote_video_path} "
            f"--output_dir {remote_output_dir} "
            f"--iterations 5000"
        )
        # Note: results will be written to:
        #   {remote_output_dir}/data_prepared/transforms.json
        #   {remote_output_dir}/baked_splat.ply
        #   {remote_output_dir}/baked_splat_sh.ply

        logger.info("Executing remote command: %s", remote_cmd)
        stdin, stdout, stderr = ssh.exec_command(remote_cmd, timeout=None)

        # Stream stdout line-by-line so we can parse iteration progress
        # and keep the connection alive via read activity.
        _stream_training_progress(stdout, report)

        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_msg = stderr.read().decode().strip()
            logger.error("Remote pipeline failed (exit %d): %s", exit_status, error_msg)
            report(f"Error: {error_msg[:100]}", -1)
            return

        logger.info("Remote pipeline finished successfully.")

        # ── 4. Download results ───────────────────────────────────────
        report("Downloading results…", 90)
        result_dir = Path(f"./results/{task_id}")
        result_dir.mkdir(parents=True, exist_ok=True)

        downloads = {
            f"{remote_output_dir}/data_prepared/transforms.json": result_dir / "transforms.json",
            f"{remote_output_dir}/baked_splat.ply": result_dir / "baked_splat.ply",
            f"{remote_output_dir}/baked_splat_sh.ply": result_dir / "baked_splat_sh.ply",
        }

        for remote_path, local_path in downloads.items():
            try:
                sftp.get(remote_path, str(local_path))
                logger.info("Downloaded %s → %s", remote_path, local_path)
            except FileNotFoundError:
                logger.warning("Remote file not found (skipping): %s", remote_path)
                # Paramiko may have created an empty local file before the error —
                # remove it so callers don't see a misleading 0-byte file.
                if local_path.exists() and local_path.stat().st_size == 0:
                    local_path.unlink()
                    logger.debug("Removed empty placeholder: %s", local_path)

        # ── 5. Clean up remote temp files ─────────────────────────────
        report("Cleaning up remote files…", 95)
        _remote_cleanup(ssh, remote_video_path, remote_output_dir)

        report("Completed successfully", 100)
        logger.info("Task %s completed successfully.", task_id)

    except paramiko.AuthenticationException as e:
        logger.error("SSH authentication failed for %s: %s", task_id, e)
        report("SSH authentication failed", -1)
    except paramiko.SSHException as e:
        logger.error("SSH error for %s: %s", task_id, e)
        report(f"SSH error: {str(e)[:80]}", -1)
    except Exception as e:
        logger.error("SSH processing failed for %s: %s", task_id, e)
        report(f"Failed: {str(e)[:80]}", -1)
    finally:
        if sftp:
            sftp.close()
        ssh.close()
        # Cleanup local temp file (only if we uploaded via SFTP)
        if local_video_path and os.path.exists(local_video_path):
            os.remove(local_video_path)
            logger.info("Cleaned up local temp file: %s", local_video_path)


def _stream_training_progress(stdout, report):
    """
    Read remote stdout line-by-line, parsing nerfstudio training output
    to report incremental progress (30 → 85 %).

    Recognises lines like:
        "Step 500/5000 ..."   (nerfstudio trainer)
        "--- Running COLMAP" / "--- Extracting Frames" (Nerf4Dgsplat phases)
    """
    COLMAP_PROG = 40    # after COLMAP finishes → 40 %
    TRAIN_START = 45    # training begins at 45 %
    TRAIN_END = 85      # training ends at 85 %

    # Regex to pick up training steps, handles various nerfstudio formats
    step_re = re.compile(r"[Ss]tep\s+(\d+)[/\s]+(\d+)")

    for raw_line in stdout:
        line = raw_line.strip()
        if not line:
            continue

        logger.debug("remote> %s", line)

        # Phase markers emitted by Nerf4Dgsplat.py
        if "Extracting Frames" in line:
            report("Extracting frames from video…", 32)
        elif "Running COLMAP" in line:
            report("Running COLMAP pose estimation…", 35)
        elif "Training complete" in line or "✅ Training complete" in line:
            report("Training complete, exporting PLY…", 85)
        elif "Pipeline complete" in line or "🎉 Pipeline complete" in line:
            report("Pipeline finished on remote.", 88)
        else:
            # Training iteration progress
            m = step_re.search(line)
            if m:
                step = int(m.group(1))
                total = int(m.group(2))
                frac = step / total if total > 0 else 0
                pct = TRAIN_START + int(frac * (TRAIN_END - TRAIN_START))
                report(f"Training… step {step}/{total}", pct)


def _remote_cleanup(ssh, remote_video_path: str, remote_output_dir: str):
    """Remove temporary files on the remote machine."""
    cmds = [
        f"rm -f {remote_video_path}",
        f"rm -rf {remote_output_dir}",
    ]
    for cmd in cmds:
        try:
            ssh.exec_command(cmd)
            logger.info("Remote cleanup: %s", cmd)
        except Exception as e:
            logger.warning("Remote cleanup failed (%s): %s", cmd, e)