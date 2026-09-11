import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]


def test_legacy_requirements_cannot_replace_managed_cuda_stack():
    setup = (REPO / "setup_hilti_conda_envs.sh").read_text(encoding="utf-8")
    match = re.search(
        r"^MANAGED_REQUIREMENTS_REGEX='([^']+)'$",
        setup,
        flags=re.MULTILINE,
    )
    assert match is not None
    assert setup.count('grep -vE "$MANAGED_REQUIREMENTS_REGEX"') == 2

    requirements = "\n".join(
        [
            "faiss-gpu",
            "torch==2.5.0",
            "torchvision>=0.20",
            "torchaudio~=2.5",
            "xformers",
            "numpy<2",
        ]
    ) + "\n"
    filtered = subprocess.run(
        ["grep", "-vE", match.group(1)],
        input=requirements,
        text=True,
        capture_output=True,
        check=True,
    )

    assert filtered.stdout == "numpy<2\n"
