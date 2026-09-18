# Development workflow

GitHub is the source of truth for this project.

## First time on a device

### macOS / Linux

```bash
git clone https://github.com/thomas-t-ma/data-cleaning-assistant.git && cd data-cleaning-assistant && bash sync.sh
```

### Windows PowerShell

```powershell
git clone https://github.com/thomas-t-ma/data-cleaning-assistant.git; cd data-cleaning-assistant; powershell -ExecutionPolicy Bypass -File .\sync.ps1
```

Because the repository is private, GitHub may ask you to authenticate on a new device.

## Normal update

From the project directory:

### macOS / Linux

```bash
bash sync.sh
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\sync.ps1
```

The sync scripts:

1. fetch `origin/main`;
2. rebase local commits and autostash tracked working changes;
3. find a suitable Python command;
4. create `.venv` if needed;
5. reinstall dependencies only when `pyproject.toml` changes;
6. run the test suite.

Use `--no-test` when you only need a quick source/dependency sync.

## Making local changes

Commit useful local work before moving to another device:

```bash
git add -A
git commit -m "Describe the change"
git push
```

The next device can then use the normal sync command.

## Why this replaces ZIP patching

For GitHub-backed projects, Git provides the same file replacement/addition behavior as the old ZIP updater while also preserving history, merging changes made on different devices, and avoiding repeated full-project downloads.

ZIP-based patching remains a fallback for projects that cannot use GitHub or when a repository is temporarily unavailable.
