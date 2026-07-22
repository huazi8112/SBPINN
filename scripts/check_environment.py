from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "torch", "tqdm")


def probe(code: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> None:
    packages = {}
    for name in PACKAGES:
        result = probe(
            f"import {name}; print(getattr({name}, '__version__', 'unknown'))"
        )
        packages[name] = {
            "installed": result["returncode"] == 0,
            "version": result["stdout"],
            "error": result["stderr"],
        }
    torch_probe = probe(
        "import json, torch; print(json.dumps({"
        "'version': torch.__version__,"
        "'cuda_available': torch.cuda.is_available(),"
        "'cuda_version': torch.version.cuda,"
        "'gpu_count': torch.cuda.device_count(),"
        "'gpu_names': [torch.cuda.get_device_name(i) "
        "for i in range(torch.cuda.device_count())]}))"
    )
    torch_info = (
        json.loads(torch_probe["stdout"])
        if torch_probe["returncode"] == 0
        else {"error": torch_probe["stderr"]}
    )
    report = {
        "python": sys.executable,
        "python_version": sys.version,
        "packages": packages,
        "torch": torch_info,
        "cpu_ready": all(
            item["installed"] for item in packages.values()
        ),
        "cuda_ready": bool(torch_info.get("cuda_available", False)),
    }
    path = Path(__file__).resolve().parent / "environment_report.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["cpu_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
