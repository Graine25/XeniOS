# iOS JIT via Xcode LLDB Broker

XeniOS iOS A64 code cache can request external JIT region preparation through:

- `brk #0xf00d` with `x16=1` (universal prepare)
- `brk #0x69` (legacy prepare)

This repository includes an LLDB stop-hook broker that handles those breakpoints
directly from Xcode's debugger session.

## Script Location

- `tools/ios/xenios_ios_lldb_jit_broker.py`

## One-Time Per Debug Session

Run these in Xcode's debug console (LLDB):

```lldb
command script import /absolute/path/to/tools/ios/xenios_ios_lldb_jit_broker.py
xenios-jit-broker-install
```

When active, the hook will:

1. Catch `SIGTRAP` stops caused by the XeniOS JIT BRK protocol.
2. Handle prepare requests (`x0=address`, `x1=length`) via GDB remote packets
   using a StikDebug-compatible memory marker flow (`M<addr>,1:69` per 16KB
   page; `_M<size>,rx` for allocation requests when `x0 == 0`).
3. Advance `pc` past the `brk` instruction and auto-continue.

## Optional: Auto-Load In LLDB Init

Add to your LLDB init if you want auto-load:

```lldb
command script import /absolute/path/to/tools/ios/xenios_ios_lldb_jit_broker.py
```

Then run `xenios-jit-broker-install` once the target is created.

## Recommended: Xcode Auto-Start

This repo includes an installer that patches `~/.lldbinit-Xcode` with a
managed block:

```bash
tools/ios/install_lldbinit_xenios_jit.sh
```

It adds:

```lldb
command script import /absolute/path/to/tools/ios/xenios_ios_lldb_jit_broker.py
xenios-jit-broker-reset-hooks
xenios-jit-broker-install
```

This installs a single scripted broker hook for the session. On detach command
(`x16=0`), the broker hook disables itself.

The installer also configures LLDB to pass through `SIGUSR2` without stopping
to avoid noisy non-fatal signal stops during gameplay.
It also disables Xcode's built-in "Memory error diagnosis" stop-hook for the
session because it can prevent JIT BRK auto-continue behavior.

If you need to force cleanup of all broker hooks in the current LLDB session:

```lldb
xenios-jit-broker-reset-hooks
```

## Notes

- This is intended for Xcode-attached debug runs on iOS ARM64.
- If your device/runtime disallows the protection transition, the hook leaves
  the stop visible so you can inspect the failure in LLDB output.
