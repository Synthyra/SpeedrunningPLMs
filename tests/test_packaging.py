import email.parser
import os
import shutil
import site
import subprocess
import sys
import tarfile
import venv
import zipfile
import pytest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    build_root = tmp_path_factory.mktemp("package-build")
    source = build_root / "source"
    source.mkdir()

    for filename in (
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "prepare.py",
        "train.py",
        "research.py",
        "program.md",
        "experiment.json",
    ):
        shutil.copy2(ROOT / filename, source / filename)
    for directory in ("evaluation", "src", "tests", "targets"):
        shutil.copytree(ROOT / directory, source / directory)

    dist = build_root / "dist"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            str(dist),
        ],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1
    return wheels[0], sdists[0], build_root


def test_wheel_contains_full_package_and_declares_runtime_dependencies(built_distributions: tuple[Path, Path, Path]) -> None:
    wheel, _, _ = built_distributions
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        expected_modules = {
            "speedrunning_plms/__init__.py",
            "speedrunning_plms/data/loaders.py",
            "speedrunning_plms/evaluation/benchmark_assets.py",
            "speedrunning_plms/flex/mods.py",
            "speedrunning_plms/models/plm.py",
            "speedrunning_plms/optim/muon.py",
            "speedrunning_plms/training/cli.py",
            "speedrunning_plms/training/publishing.py",
            "speedrunning_plms/research/benchmark.py",
            "speedrunning_plms/research/engine.py",
            "speedrunning_plms/research/runner.py",
        }
        assert expected_modules <= names
        assert not any(name.startswith("tests/") for name in names)

        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = email.parser.Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        requirements = metadata.get_all("Requires-Dist", [])
        assert metadata["Requires-Python"] == ">=3.10"
        for dependency in ("datasets", "huggingface-hub", "numpy", "torch", "transformers"):
            assert any(requirement.startswith(dependency) for requirement in requirements)
        assert any('extra == "training"' in requirement for requirement in requirements)
        assert any('extra == "evaluation"' in requirement for requirement in requirements)
        assert any('extra == "test"' in requirement for requirement in requirements)


def test_sdist_contains_sources_tests_and_build_metadata(built_distributions: tuple[Path, Path, Path]) -> None:
    _, sdist, _ = built_distributions
    with tarfile.open(sdist, "r:gz") as archive:
        names = {Path(name).as_posix() for name in archive.getnames()}
    prefix = "speedrunning_plms-0.1.0/"
    assert {
        f"{prefix}LICENSE",
        f"{prefix}MANIFEST.in",
        f"{prefix}README.md",
        f"{prefix}pyproject.toml",
        f"{prefix}evaluation/benchmark_manifest.json",
        f"{prefix}src/speedrunning_plms/evaluation/benchmark_assets.py",
        f"{prefix}src/speedrunning_plms/models/plm.py",
        f"{prefix}tests/test_benchmark_manifest.py",
        f"{prefix}tests/test_hf_serialization.py",
        f"{prefix}program.md",
        f"{prefix}experiment.json",
        f"{prefix}prepare.py",
        f"{prefix}research.py",
    } <= names


def test_installed_wheel_imports_and_console_entrypoint(built_distributions: tuple[Path, Path, Path]) -> None:
    wheel, _, build_root = built_distributions
    environment = build_root / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)

    if os.name == "nt":
        python = environment / "Scripts" / "python.exe"
        console = environment / "Scripts" / "speedrun-plm.exe"
    else:
        python = environment / "bin" / "python"
        console = environment / "bin" / "speedrun-plm"

    # Reuse only the test runner's already-installed dependency directories.
    # The child environment still owns the speedrunning_plms wheel, and .pth
    # files from the parent environment are intentionally not reprocessed.
    child_site = Path(
        subprocess.check_output(
            [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
        ).strip()
    )
    dependency_paths = [path for path in site.getsitepackages() if Path(path).is_dir()]
    (child_site / "test-dependencies.pth").write_text(
        "".join(f"{path}\n" for path in dependency_paths),
        encoding="utf-8",
    )

    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    smoke_dir = build_root / "smoke"
    smoke_dir.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    script = """
from importlib.metadata import version
from pathlib import Path

import speedrunning_plms
from speedrunning_plms import PLM, PLMConfig
from speedrunning_plms.data import ChunkPacker
from speedrunning_plms.evaluation import load_benchmark_manifest
from speedrunning_plms.flex import generate_dilated_sliding_window
from speedrunning_plms.optim import Muon
from speedrunning_plms.training.publishing import publish_model_to_hub

assert version("speedrunning-plms") == "0.1.0"
assert "site-packages" in Path(speedrunning_plms.__file__).as_posix()
assert all(item is not None for item in (PLM, PLMConfig, ChunkPacker, Muon))
assert callable(generate_dilated_sliding_window)
assert callable(load_benchmark_manifest)
assert callable(publish_model_to_hub)
"""
    completed = subprocess.run(
        [str(python), "-c", script],
        cwd=smoke_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    completed = subprocess.run(
        [str(console), "--help"],
        cwd=smoke_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "fixed 15% masking" in completed.stdout
    for name, expected in (("speedrun-prepare", "--include-test"), ("speedrun-research", "run")):
        entrypoint = console.with_name(name + (".exe" if os.name == "nt" else ""))
        completed = subprocess.run(
            [str(entrypoint), "--help"], cwd=smoke_dir, env=env,
            capture_output=True, text=True, timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert expected in completed.stdout
