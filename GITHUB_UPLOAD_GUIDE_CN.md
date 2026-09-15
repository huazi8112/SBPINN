# GitHub upload guide

Target repository used in the manuscript:

`https://github.com/huazi8112/SBPINN`

## Recommended update workflow (PowerShell)

```powershell
git clone https://github.com/huazi8112/SBPINN.git
cd SBPINN

# Keep the .git directory, but replace the tracked working-tree files with the
# contents of SBPINN_GitHub_ready.

git status
git add -A
git commit -m "Update code and results for revised SBPINN manuscript"
git push origin main
```

Before pushing, run:

```powershell
python scripts\syntax_check.py
git status
```

Do not upload the original project archive, IDE metadata, archived historical experiments, checkpoints, or dense temporary arrays.
