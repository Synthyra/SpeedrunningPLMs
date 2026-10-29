"""Run a command with the workspace credentials loaded into its environment, and never print one.

    python -m foundry.secrets_env COMMAND [ARG ...]
    e.g. py -m foundry.secrets_env .\\ws.cmd run anthropic_adaptyv_competition -- python -m modal run ...

The file is the one `load_secrets_env` finds: the path in `FOUNDRY_SECRETS`, else the nearest `.secrets.env` above this
package. A variable the shell already exports wins over the file. The command's own exit status is returned, so a
caller sees its failure as before; the command runs in this process's environment, which is the only place the values go.
"""

from __future__ import annotations

import subprocess
import sys

from foundry.secrets_env import load_secrets_env


def main(argv: list[str]) -> int:
    """Load the credentials, run `argv` as a command, and return its exit status."""
    if not argv:
        print("usage: python -m foundry.secrets_env COMMAND [ARG ...]", file=sys.stderr)
        return 2
    load_secrets_env()
    return subprocess.run(argv, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
