# Local To Remote Sync Commands

Use these commands whenever local project files change and need to be copied to
the remote server.

## Full Project Sync

Local PowerShell:

```powershell
cd C:\Users\28101\Documents\VLM-Competition

tar --exclude .git --exclude .codex --exclude .agents --exclude VLM-Competition-sync.tar --exclude VLM-Competition-sync.zip -cf VLM-Competition-sync.tar .

scp C:\Users\28101\Documents\VLM-Competition\VLM-Competition-sync.tar Jiali_Wang@10.176.61.126:/home/Jiali_Wang/workspace/
```

Remote server:

```bash
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate orena-focus

mkdir -p ~/workspace/VLM-Competition
tar -xf ~/workspace/VLM-Competition-sync.tar -C ~/workspace/VLM-Competition

cd ~/workspace/VLM-Competition
ls scripts
```

## Single Script Sync

Local PowerShell example:

```powershell
scp C:\Users\28101\Documents\VLM-Competition\scripts\train_qwen3vl_lora_sft_smoke.py Jiali_Wang@10.176.61.126:/home/Jiali_Wang/workspace/VLM-Competition/scripts/
```

Remote server:

```bash
cd ~/workspace/VLM-Competition
python -m py_compile scripts/train_qwen3vl_lora_sft_smoke.py
```

## Notes

- Prefer `.tar` over PowerShell `Compress-Archive` zip because Linux extraction
  can preserve Windows backslashes as literal filename characters.
- If using `scp`, the user enters the SSH password interactively.
- Do not store passwords in this repository.

