"""Сборка классического PDF из latex/main.tex. Требуется XeLaTeX."""
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "latex"

if __name__ == "__main__":
    shutil.copy2(ROOT / "Рыко_ДЗ1_вариант25.py", PROJECT / "program.py")
    for image in (ROOT / "результаты").glob("*.png"):
        shutil.copy2(image, PROJECT / "images" / image.name)
    with tempfile.TemporaryDirectory(prefix="vychmat25-tex-") as directory:
        # Повторный проход уточняет ссылки и ширину таблиц.
        for _ in range(2):
            subprocess.run([
                "xelatex", "-interaction=nonstopmode", "-halt-on-error",
                "-output-directory=" + directory, "main.tex"
            ], cwd=PROJECT, check=True)
        target = ROOT / "Рыко_ДЗ1_вариант25.pdf"
        shutil.copy2(Path(directory) / "main.pdf", target)
        print(target)
