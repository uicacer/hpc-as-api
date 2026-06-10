"""
hpc_as_api.cli — Lifecycle management CLI for hpc-as-api.

Commands:

  hpc-as-api serve      Start the gateway server directly (foreground)
  hpc-as-api install    Install as a systemd service (Linux) or launchd plist (macOS)
  hpc-as-api start      Start the installed service
  hpc-as-api stop       Stop the running service
  hpc-as-api restart    Restart the service
  hpc-as-api status     Show service status + health check
  hpc-as-api uninstall  Remove the service

When no subcommand is given, ``serve`` is used (backward compatible with v0.1.x).
"""

import argparse
import os
import platform
import subprocess
import sys
import textwrap

SERVICE_NAME = "hpc-as-api"
SYSTEMD_SERVICE_PATH = f"/etc/systemd/system/{SERVICE_NAME}.service"
LAUNCHD_PLIST_PATH = os.path.expanduser(f"~/Library/LaunchAgents/com.uicacer.{SERVICE_NAME}.plist")


def _python_bin() -> str:
    return sys.executable


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def _require_root_or_sudo(action: str):
    if os.geteuid() != 0:
        print(f"[hpc-as-api] {action} requires root. Try: sudo hpc-as-api {action}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def cmd_serve(args):
    """Start the gateway server in the foreground."""
    import uvicorn

    from hpc_as_api.app import LOG_LEVEL, PROXY_HOST, PROXY_PORT, app

    host = getattr(args, "host", None) or PROXY_HOST
    port = getattr(args, "port", None) or PROXY_PORT
    log_level = getattr(args, "log_level", None) or LOG_LEVEL
    print(f"[hpc-as-api] Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def cmd_install(args):
    if _is_linux():
        _install_systemd(args)
    elif _is_macos():
        _install_launchd(args)
    else:
        print(f"[hpc-as-api] Unsupported platform: {platform.system()}")
        print("  Manual start: uvicorn hpc_as_api.app:app --host 0.0.0.0 --port 8001")
        sys.exit(1)


def _install_systemd(args):
    _require_root_or_sudo("install")
    python = _python_bin()
    env_file = getattr(args, "env_file", "") or ""

    env_section = (
        f"EnvironmentFile={env_file}"
        if env_file
        else "# No EnvironmentFile set — add with: EnvironmentFile=/path/to/proxy-env"
    )

    unit = textwrap.dedent(f"""\
        [Unit]
        Description=HPC-as-API Proxy Gateway
        After=network.target

        [Service]
        User={os.getenv("SUDO_USER", "ubuntu")}
        {env_section}
        ExecStart={python} -m uvicorn hpc_as_api.app:app --host 0.0.0.0 --port 8001 --log-level info
        Restart=always
        RestartSec=5
        StandardOutput=journal
        StandardError=journal

        [Install]
        WantedBy=multi-user.target
    """)

    with open(SYSTEMD_SERVICE_PATH, "w") as f:
        f.write(unit)
    print(f"[hpc-as-api] Service file written: {SYSTEMD_SERVICE_PATH}")
    _run(["systemctl", "daemon-reload"])
    _run(["systemctl", "enable", SERVICE_NAME])
    print("[hpc-as-api] Installed. Run: sudo hpc-as-api start")
    if not env_file:
        print("  NOTE: Set EnvironmentFile in the service file before starting:")
        print(f"    sudo nano {SYSTEMD_SERVICE_PATH}")


def _install_launchd(args):
    python = _python_bin()
    env_file = getattr(args, "env_file", "") or ""

    env_dict: dict[str, str] = {}
    if env_file and os.path.isfile(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_dict[k.strip()] = v.strip()

    env_xml = "\n".join(f"            <key>{k}</key><string>{v}</string>" for k, v in env_dict.items())
    env_block = f"<key>EnvironmentVariables</key>\n        <dict>\n{env_xml}\n        </dict>" if env_dict else ""

    plist = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key><string>com.uicacer.{SERVICE_NAME}</string>
            <key>ProgramArguments</key>
            <array>
                <string>{python}</string>
                <string>-m</string><string>uvicorn</string>
                <string>hpc_as_api.app:app</string>
                <string>--host</string><string>0.0.0.0</string>
                <string>--port</string><string>8001</string>
            </array>
            {env_block}
            <key>RunAtLoad</key><true/>
            <key>KeepAlive</key><true/>
            <key>StandardOutPath</key><string>/tmp/hpc-as-api.log</string>
            <key>StandardErrorPath</key><string>/tmp/hpc-as-api.error.log</string>
        </dict>
        </plist>
    """)

    os.makedirs(os.path.dirname(LAUNCHD_PLIST_PATH), exist_ok=True)
    with open(LAUNCHD_PLIST_PATH, "w") as f:
        f.write(plist)
    print(f"[hpc-as-api] LaunchAgent written: {LAUNCHD_PLIST_PATH}")
    _run(["launchctl", "load", LAUNCHD_PLIST_PATH], check=False)
    print("[hpc-as-api] Installed and loaded.")


# ---------------------------------------------------------------------------
# start / stop / restart / status / uninstall
# ---------------------------------------------------------------------------


def cmd_start(args):
    if _is_linux():
        _require_root_or_sudo("start")
        _run(["systemctl", "start", SERVICE_NAME])
        print("[hpc-as-api] Started.")
    elif _is_macos():
        _run(["launchctl", "load", "-w", LAUNCHD_PLIST_PATH], check=False)
    else:
        print("[hpc-as-api] Use: uvicorn hpc_as_api.app:app --host 0.0.0.0 --port 8001")


def cmd_stop(args):
    if _is_linux():
        _require_root_or_sudo("stop")
        _run(["systemctl", "stop", SERVICE_NAME])
        print("[hpc-as-api] Stopped.")
    elif _is_macos():
        _run(["launchctl", "unload", LAUNCHD_PLIST_PATH], check=False)


def cmd_restart(args):
    if _is_linux():
        _require_root_or_sudo("restart")
        _run(["systemctl", "restart", SERVICE_NAME])
        print("[hpc-as-api] Restarted.")
    elif _is_macos():
        _run(["launchctl", "unload", LAUNCHD_PLIST_PATH], check=False)
        _run(["launchctl", "load", LAUNCHD_PLIST_PATH], check=False)


def cmd_status(args):
    print("[hpc-as-api] Service status:")
    if _is_linux():
        _run(["systemctl", "status", SERVICE_NAME, "--no-pager"], check=False)
    elif _is_macos():
        _run(["launchctl", "list", f"com.uicacer.{SERVICE_NAME}"], check=False)

    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    port = getattr(args, "port", 8001) or 8001
    try:
        import json

        import httpx

        resp = httpx.get(f"http://{host}:{port}/health", timeout=5.0)
        print(f"\n[hpc-as-api] Health check → HTTP {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        print(f"\n[hpc-as-api] Health check failed: {e}")
        print("  (Is the service running? Try: sudo hpc-as-api start)")


def cmd_uninstall(args):
    if _is_linux():
        _require_root_or_sudo("uninstall")
        _run(["systemctl", "stop", SERVICE_NAME], check=False)
        _run(["systemctl", "disable", SERVICE_NAME], check=False)
        if os.path.exists(SYSTEMD_SERVICE_PATH):
            os.remove(SYSTEMD_SERVICE_PATH)
            print(f"[hpc-as-api] Removed: {SYSTEMD_SERVICE_PATH}")
        _run(["systemctl", "daemon-reload"])
        print("[hpc-as-api] Service uninstalled.")
    elif _is_macos():
        _run(["launchctl", "unload", LAUNCHD_PLIST_PATH], check=False)
        if os.path.exists(LAUNCHD_PLIST_PATH):
            os.remove(LAUNCHD_PLIST_PATH)
            print(f"[hpc-as-api] Removed: {LAUNCHD_PLIST_PATH}")
        print("[hpc-as-api] Service uninstalled.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    """
    hpc-as-api CLI entry point.

    No subcommand → starts the server (backward compatible with v0.1.x).
    """
    parser = argparse.ArgumentParser(
        prog="hpc-as-api",
        description="HPC-as-API gateway — lifecycle management and server start",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the gateway server (foreground)")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--log-level", default=None, dest="log_level")

    # install
    p_install = subparsers.add_parser("install", help="Install as a systemd/launchd service")
    p_install.add_argument(
        "--env-file",
        default="",
        dest="env_file",
        help="Path to an EnvironmentFile with env vars (recommended for secrets)",
    )

    # start / stop / restart
    subparsers.add_parser("start", help="Start the installed service")
    subparsers.add_parser("stop", help="Stop the running service")
    subparsers.add_parser("restart", help="Restart the service")

    # status
    p_status = subparsers.add_parser("status", help="Show service status and health check")
    p_status.add_argument("--host", default="127.0.0.1")
    p_status.add_argument("--port", type=int, default=8001)

    # uninstall
    subparsers.add_parser("uninstall", help="Remove the installed service")

    args = parser.parse_args()

    dispatch = {
        "serve": cmd_serve,
        "install": cmd_install,
        "start": cmd_start,
        "stop": cmd_stop,
        "restart": cmd_restart,
        "status": cmd_status,
        "uninstall": cmd_uninstall,
        None: cmd_serve,
    }

    handler = dispatch.get(args.command, cmd_serve)
    handler(args)


if __name__ == "__main__":
    main()
