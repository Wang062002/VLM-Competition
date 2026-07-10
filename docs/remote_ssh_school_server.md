# VS Code Remote-SSH Guide for the School Server

This guide assumes you already have a school server username and host address.
Keep the password or private key out of this repository.

## 1. Local Setup

1. Install VS Code.
2. Install the `Remote - SSH` extension. The `Remote Development` extension pack
   is also fine.
3. Open PowerShell locally and test plain SSH first:

```powershell
ssh your_user@school.server.edu
```

If the school gave you a port:

```powershell
ssh -p 2222 your_user@school.server.edu
```

If this does not work, fix SSH before opening VS Code. VS Code Remote-SSH uses
the same underlying connection.

## 2. Add the Host in VS Code

1. Press `Ctrl+Shift+P`.
2. Run `Remote-SSH: Add New SSH Host...`.
3. Paste the same command that worked in PowerShell:

```text
ssh -p 2222 your_user@school.server.edu
```

4. Choose the default SSH config file.
5. Press `Ctrl+Shift+P` again and run `Remote-SSH: Connect to Host...`.
6. Select the new host.
7. If prompted for platform, choose `Linux`.
8. Wait while VS Code installs VS Code Server on the remote machine.

## 3. Verify You Are on the Remote Server

Open `Terminal > New Terminal` in the VS Code window connected to the server:

```bash
hostname
whoami
pwd
nvidia-smi
python3 --version
which python3
df -h
```

Expected result:

- `hostname` is the school server, not your laptop.
- `nvidia-smi` shows one or more NVIDIA GPUs.
- Python is 3.10 or newer, or the server has conda/module tools to load it.
- `df -h` shows a writable location with enough free space for videos and model
  caches.

## 4. Choose Remote Paths

Default paths used by the scripts:

```bash
export WORKSPACE_DIR="$HOME/workspace"
export FOCUS_ROOT_DIR="$HOME/data/focus"
export RUNS_DIR="$HOME/workspace/focus-runs"
```

If the server has a large shared disk, prefer that for `FOCUS_ROOT_DIR`, for
example:

```bash
export FOCUS_ROOT_DIR="/data/$USER/focus"
```

## 5. Install and Run

From the remote VS Code terminal:

```bash
mkdir -p "$HOME/workspace"
cd "$HOME/workspace"
git clone https://github.com/<your-user>/VLM-Competition.git
cd VLM-Competition

export FOCUS_ROOT_DIR="$HOME/data/focus"
bash scripts/setup_school_server.sh
source "$HOME/workspace/.venvs/orena-focus/bin/activate"

python scripts/prepare_heico_data.py --root-dir "$FOCUS_ROOT_DIR" --skip-frames
python scripts/check_focus_dataset.py --root-dir "$FOCUS_ROOT_DIR" --make-video-sample --no-overlay
python scripts/run_segment_baseline.py --root-dir "$FOCUS_ROOT_DIR" --num-eval 3 --no-overlay
```

If the 3-sample smoke test succeeds:

```bash
python scripts/run_segment_baseline.py --root-dir "$FOCUS_ROOT_DIR" --num-eval 100 --no-overlay
```

For the official-style overlay run, generate overlays first and omit
`--no-overlay`:

```bash
python scripts/prepare_heico_data.py --root-dir "$FOCUS_ROOT_DIR" --skip-frames
python scripts/run_segment_baseline.py --root-dir "$FOCUS_ROOT_DIR" --num-eval 100
```

## 6. Common Fixes

- If Hugging Face data access fails, run `huggingface-cli login` on the remote
  server and make sure you have accepted any dataset terms in the browser.
- If CUDA is not available in Python, reinstall PyTorch with the CUDA wheel that
  matches the server driver, then rerun `python -c "import torch; print(torch.cuda.is_available())"`.
- If inference runs out of memory, retry with larger `--video-stride`, smaller
  `--width/--height`, or fewer samples.
- If overlay generation is too slow, use `--no-overlay` for smoke tests and
  come back to overlay once the basic pipeline works.
