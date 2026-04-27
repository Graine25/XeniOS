#!/usr/bin/env python3
"""LLDB stop-hook broker for XeniOS iOS universal JIT breakpoints.

This script handles the BRK protocol emitted by A64CodeCache on iOS:
  - brk #0xf00d with x16=1: prepare region (x0=addr, x1=len)
  - brk #0xf00d with x16=0: detach broker
  - brk #0x69 (legacy): prepare region (x0=addr, x1=len)

Usage from LLDB / Xcode debug console:
  command script import /absolute/path/to/tools/ios/xenios_ios_lldb_jit_broker.py
  xenios-jit-broker-install
"""

import re

import lldb

SIGTRAP = 5

BRK_MASK = 0xFFE0001F
BRK_OPCODE = 0xD4200000
UNIVERSAL_BRK_IMM = 0xF00D
LEGACY_PREPARE_BRK_IMM = 0x69

CMD_DETACH = 0
CMD_PREPARE_REGION = 1

JIT_PAGE_STRIDE = 0x4000

_installed_hook_id = None
_autostart_hook_id = None
_broker_installed = False
_target_name_patterns = ("xenios", "xenia-app", "xenia")
_auto_detach_after_first_prepare = True


def _stream_write(stream, text):
  if text is None:
    return
  line = text if text.endswith("\n") else (text + "\n")
  if stream is None:
    try:
      print(line, end="")
    except Exception:
      pass
    return
  try:
    stream.Print(line)
  except Exception:
    # Never propagate stream logging failures into LLDB stop-hook execution.
    pass


def _read_reg_u64(frame, name):
  reg = frame.FindRegister(name)
  if not reg.IsValid():
    return None
  return reg.GetValueAsUnsigned(0)


def _write_reg_u64(frame, name, value):
  reg = frame.FindRegister(name)
  if not reg.IsValid():
    return False
  return reg.SetValueFromCString(f"0x{value:x}")


def _read_u32_le(process, address):
  error = lldb.SBError()
  data = process.ReadMemory(address, 4, error)
  if not error.Success() or len(data) != 4:
    return None
  return int.from_bytes(data, byteorder="little", signed=False)


def _parse_packet_response_text(text):
  if not text:
    return None
  for line in text.splitlines():
    line = line.strip()
    if line.startswith("response:"):
      return line.split("response:", 1)[1].strip()
  return None


def _run_command(debugger, command):
  result = lldb.SBCommandReturnObject()
  debugger.GetCommandInterpreter().HandleCommand(command, result)
  return result


def _send_packet(debugger, packet):
  result = _run_command(debugger, f"process plugin packet send {packet}")
  text = (result.GetOutput() or "") + "\n" + (result.GetError() or "")
  if not result.Succeeded():
    return None, f"packet send failed for '{packet}': {text.strip()}"
  response = _parse_packet_response_text(text)
  if response is None:
    return None, f"packet response parse failed for '{packet}': {text.strip()}"
  return response, None


def _stik_prepare_memory_region(debugger, start_address, region_size):
  # StikDebug strategy from handleJITPageWrite:
  # write one byte 0x69 at each 16KB boundary in the requested region.
  if region_size == 0:
    return True, 0, "no-op length=0"

  # Round up so partially covered trailing pages also get marked.
  page_count = (region_size + (JIT_PAGE_STRIDE - 1)) // JIT_PAGE_STRIDE

  current_address = start_address
  for _ in range(page_count):
    response, error = _send_packet(debugger, f"M{current_address:x},1:69")
    if error:
      return False, f"M write failed at 0x{current_address:x}: {error}"
    if response != "OK":
      return False, f"M write failed at 0x{current_address:x}: response='{response}'"
    current_address += JIT_PAGE_STRIDE

  return True, f"wrote {page_count} page markers (stride=0x{JIT_PAGE_STRIDE:x})"


def _prepare_region(debugger, address, length):
  if length == 0:
    return True, address, "no-op length=0"

  target_address = address
  if target_address == 0:
    # StikDebug uses _M<size>,rx to request an RX region from debugserver.
    response, error = _send_packet(debugger, f"_M{length:x},rx")
    if error:
      return False, 0, error
    if not response or response.startswith("E"):
      return False, 0, f"_M allocation failed: response='{response}'"
    try:
      target_address = int(response, 16)
    except Exception:
      return False, 0, f"_M allocation parse failed: response='{response}'"

  ok, note = _stik_prepare_memory_region(debugger, target_address, length)
  if not ok:
    return False, target_address, note
  return True, target_address, note


class XeniosIOSJITStopHook:
  """LLDB scripted stop-hook for XeniOS iOS JIT BRK protocol."""

  def __init__(self, target, extra_args, internal_dict):
    self.target = target
    self.debugger = target.GetDebugger()
    self.detached = False
    self.one_shot_prepare_done = False
    self.handled_breaks = 0
    del extra_args
    del internal_dict

  def handle_stop(self, exe_ctx, stream):
    try:
      if self.detached:
        return True

      thread = exe_ctx.GetThread()
      process = exe_ctx.GetProcess()
      frame = exe_ctx.GetFrame()
      if not thread.IsValid() or not process.IsValid() or not frame.IsValid():
        return True

      stop_reason = thread.GetStopReason()
      if stop_reason not in (
          lldb.eStopReasonSignal,
          lldb.eStopReasonException,
          lldb.eStopReasonBreakpoint,
      ):
        return True
      if stop_reason == lldb.eStopReasonSignal:
        if thread.GetStopReasonDataCount() < 1:
          return True
        if thread.GetStopReasonDataAtIndex(0) != SIGTRAP:
          return True

      pc = _read_reg_u64(frame, "pc")
      if pc is None:
        return True
      instruction = _read_u32_le(process, pc)
      if instruction is None:
        return True
      if (instruction & BRK_MASK) != BRK_OPCODE:
        return True

      brk_imm = (instruction >> 5) & 0xFFFF
      x16 = _read_reg_u64(frame, "x16")
      x0 = _read_reg_u64(frame, "x0")
      x1 = _read_reg_u64(frame, "x1")
      if x16 is None or x0 is None or x1 is None:
        return True

      recognized = False
      command_ok = False
      if brk_imm == UNIVERSAL_BRK_IMM:
        recognized = True
        if x16 == CMD_DETACH:
          self.detached = True
          command_ok = True
          _disable_broker_hook_fast(self.debugger)
          _stream_write(stream, "xenios-jit-broker: detached and broker hook disabled")
        elif x16 == CMD_PREPARE_REGION:
          ok, out_address, note = _prepare_region(self.debugger, x0, x1)
          _write_reg_u64(frame, "x0", out_address)
          command_ok = ok
          if ok:
            _stream_write(
                stream,
                "xenios-jit-broker: prepared universal region "
                f"addr=0x{x0:x} len=0x{x1:x} ({note})",
            )
            if _auto_detach_after_first_prepare:
              self.one_shot_prepare_done = True
              _stream_write(
                  stream,
                  "xenios-jit-broker: one-shot prepare done; awaiting detach command",
              )
          else:
            _stream_write(
                stream,
                "xenios-jit-broker: failed universal prepare "
                f"addr=0x{x0:x} len=0x{x1:x} ({note})",
            )
        else:
          return True
      elif brk_imm == LEGACY_PREPARE_BRK_IMM:
        recognized = True
        ok, out_address, note = _prepare_region(self.debugger, x0, x1)
        _write_reg_u64(frame, "x0", out_address)
        command_ok = ok
        if ok:
          _stream_write(
              stream,
              "xenios-jit-broker: prepared legacy region "
              f"addr=0x{x0:x} len=0x{x1:x} ({note})",
          )
          if _auto_detach_after_first_prepare:
            self.detached = True
            _disable_broker_hook_fast(self.debugger)
            _stream_write(
                stream,
                "xenios-jit-broker: detached after one-shot legacy prepare and broker hook disabled",
            )
        else:
          _stream_write(
              stream,
              "xenios-jit-broker: failed legacy prepare "
              f"addr=0x{x0:x} len=0x{x1:x} ({note})",
          )
      else:
        return True

      if not recognized:
        return True

      # Skip the trapping instruction and auto-continue.
      _write_reg_u64(frame, "pc", pc + 4)
      self.handled_breaks += 1
      if not command_ok:
        _stream_write(
            stream,
            "xenios-jit-broker: continuing after prepare failure; "
            "XeniOS will retry/fallback",
        )
      return False
    except Exception as exc:
      _stream_write(stream, f"xenios-jit-broker: internal stop-hook exception: {exc}")
      return True


def _parse_hook_id_from_add_output(text):
  if not text:
    return None
  for pattern in (
      r"Stop hook #([0-9]+) added\.",
      r"^\s*Hook:\s*([0-9]+)\s*$",
      r"^\s*-\s*Hook\s*#?\s*([0-9]+)(?:\s|\(|$)",
  ):
    match = re.search(pattern, text, re.MULTILINE)
    if match:
      return int(match.group(1))
  return None


def _parse_stop_hook_blocks(text):
  blocks = []
  current_id = None
  current_lines = []
  for raw_line in (text or "").splitlines():
    line = raw_line.rstrip()
    match = re.match(r"^\s*Hook:\s*([0-9]+)\s*$", line)
    if not match:
      match = re.match(r"^\s*-\s*Hook\s*#?\s*([0-9]+)(?:\s|\(|$)", line)
    if match:
      if current_id is not None:
        blocks.append((current_id, "\n".join(current_lines)))
      current_id = int(match.group(1))
      current_lines = [line]
      continue
    if current_id is not None:
      current_lines.append(line)
  if current_id is not None:
    blocks.append((current_id, "\n".join(current_lines)))
  return blocks


def _find_stop_hook_ids_containing(debugger, needle):
  hook_list = _run_command(debugger, "target stop-hook list")
  if not hook_list.Succeeded():
    return []
  matches = []
  lowered_needle = needle.lower()
  for hook_id, hook_block in _parse_stop_hook_blocks(hook_list.GetOutput() or ""):
    if lowered_needle in hook_block.lower():
      matches.append(hook_id)
  return matches


def _delete_stop_hooks(debugger, hook_ids):
  seen = set()
  for hook_id in hook_ids:
    if hook_id is None:
      continue
    if hook_id in seen:
      continue
    seen.add(hook_id)
    _run_command(debugger, f"target stop-hook delete {hook_id}")


def _disable_builtin_memory_diagnosis_hooks(debugger):
  hook_list = _run_command(debugger, "target stop-hook list")
  if not hook_list.Succeeded():
    return 0
  disabled_count = 0
  for hook_id, hook_block in _parse_stop_hook_blocks(hook_list.GetOutput() or ""):
    if "Memory error diagnosis (built-in)" not in hook_block:
      continue
    _run_command(debugger, f"target stop-hook disable {hook_id}")
    disabled_count += 1
  return disabled_count


def _remove_autostart_hooks(debugger):
  global _autostart_hook_id
  hook_ids = []
  if _autostart_hook_id is not None:
    hook_ids.append(_autostart_hook_id)
  hook_ids.extend(_find_stop_hook_ids_containing(debugger, "xenios-jit-broker-autostart"))
  _delete_stop_hooks(debugger, hook_ids)
  _autostart_hook_id = None


def _remove_broker_hooks(debugger):
  global _installed_hook_id
  global _broker_installed
  hook_ids = []
  if _installed_hook_id is not None:
    hook_ids.append(_installed_hook_id)
  hook_ids.extend(
      _find_stop_hook_ids_containing(
          debugger, "xenios_ios_lldb_jit_broker.XeniosIOSJITStopHook"
      )
  )
  _delete_stop_hooks(debugger, hook_ids)
  _installed_hook_id = None
  _broker_installed = False


def _clear_runtime_broker_state():
  global _broker_installed
  _broker_installed = False


def _disable_hook_by_id(debugger, hook_id):
  if hook_id is None:
    return
  _run_command(debugger, f"target stop-hook disable {hook_id}")


def _disable_broker_hook_fast(debugger):
  global _installed_hook_id
  _clear_runtime_broker_state()
  _disable_hook_by_id(debugger, _installed_hook_id)


def _disable_autostart_hook_fast(debugger):
  global _autostart_hook_id
  _disable_hook_by_id(debugger, _autostart_hook_id)
  _autostart_hook_id = None


def _get_target_executable(debugger):
  target = debugger.GetSelectedTarget()
  if not target or not target.IsValid():
    return ""
  executable = target.GetExecutable()
  if not executable or not executable.IsValid():
    return ""
  return executable.GetFilename() or ""


def _is_xenios_target(debugger):
  executable_name = _get_target_executable(debugger).lower()
  if not executable_name:
    return False
  return any(pattern in executable_name for pattern in _target_name_patterns)


def _is_jit_brk_stop(exe_ctx):
  thread = exe_ctx.GetThread()
  process = exe_ctx.GetProcess()
  frame = exe_ctx.GetFrame()
  if not thread.IsValid() or not process.IsValid() or not frame.IsValid():
    return False

  stop_reason = thread.GetStopReason()
  if stop_reason not in (
      lldb.eStopReasonSignal,
      lldb.eStopReasonException,
      lldb.eStopReasonBreakpoint,
  ):
    return False
  if stop_reason == lldb.eStopReasonSignal:
    if thread.GetStopReasonDataCount() < 1:
      return False
    if thread.GetStopReasonDataAtIndex(0) != SIGTRAP:
      return False

  pc = _read_reg_u64(frame, "pc")
  if pc is None:
    return False
  instruction = _read_u32_le(process, pc)
  if instruction is None:
    return False
  if (instruction & BRK_MASK) != BRK_OPCODE:
    return False

  brk_imm = (instruction >> 5) & 0xFFFF
  return brk_imm in (UNIVERSAL_BRK_IMM, LEGACY_PREPARE_BRK_IMM)


def install_xenios_jit_broker(debugger, command, exe_ctx, result, internal_dict):
  global _installed_hook_id
  global _broker_installed
  del command
  del exe_ctx
  del internal_dict

  # Avoid recursive stop-hook list/delete command execution from within a stop
  # callback path (can destabilize LLDB RPC on newer Xcode). Delete only the
  # previously tracked broker hook id if we have one.
  _broker_installed = False
  if _installed_hook_id is not None:
    _run_command(debugger, f"target stop-hook delete {_installed_hook_id}")
    _installed_hook_id = None

  # Use LLDB's canonical argument order to avoid parser/version quirks.
  sigtrap = _run_command(debugger, "process handle -p false -s true -n true SIGTRAP")
  if not sigtrap.Succeeded():
    result.AppendWarning(
        "xenios-jit-broker: failed to update SIGTRAP handling: %s"
        % (sigtrap.GetError() or "")
    )
  sigusr2 = _run_command(debugger, "process handle -p true -s false -n false SIGUSR2")
  if not sigusr2.Succeeded():
    result.AppendWarning(
        "xenios-jit-broker: failed to update SIGUSR2 handling: %s"
        % (sigusr2.GetError() or "")
    )
  disabled_builtin_hooks = _disable_builtin_memory_diagnosis_hooks(debugger)
  if disabled_builtin_hooks > 0:
    result.AppendMessage(
        f"xenios-jit-broker: disabled {disabled_builtin_hooks} built-in memory diagnosis stop-hook(s)"
    )

  hook = _run_command(
      debugger,
      "target stop-hook add -P xenios_ios_lldb_jit_broker.XeniosIOSJITStopHook -I false",
  )
  if not hook.Succeeded():
    result.SetError(
        "xenios-jit-broker: failed to install stop-hook: %s" % (hook.GetError() or "")
    )
    return

  output = hook.GetOutput() or ""
  _installed_hook_id = _parse_hook_id_from_add_output(output)
  _broker_installed = True
  result.AppendMessage("xenios-jit-broker: installed")
  if output.strip():
    result.AppendMessage(output.strip())


def install_autostart_xenios_jit_broker(debugger, command, exe_ctx, result, internal_dict):
  global _autostart_hook_id
  del command
  del exe_ctx
  del internal_dict

  _remove_autostart_hooks(debugger)
  hook = _run_command(
      debugger,
      'target stop-hook add -o "xenios-jit-broker-autostart" -I false',
  )
  if not hook.Succeeded():
    result.SetError(
        "xenios-jit-broker: failed to install autostart hook: %s"
        % (hook.GetError() or "")
    )
    return

  output = hook.GetOutput() or ""
  _autostart_hook_id = _parse_hook_id_from_add_output(output)
  result.AppendMessage("xenios-jit-broker: autostart hook installed")
  if output.strip():
    result.AppendMessage(output.strip())


def reset_xenios_jit_broker_hooks(debugger, command, exe_ctx, result, internal_dict):
  del command
  del exe_ctx
  del internal_dict
  _remove_broker_hooks(debugger)
  _remove_autostart_hooks(debugger)
  result.AppendMessage("xenios-jit-broker: removed all broker/autostart hooks")


def autostart_xenios_jit_broker(debugger, command, exe_ctx, result, internal_dict):
  del command
  del internal_dict
  global _broker_installed

  if not _is_xenios_target(debugger):
    return

  # Avoid expensive / noisy stop-hook list queries on every stop.
  if _broker_installed:
    return

  target = debugger.GetSelectedTarget()
  process = target.GetProcess() if target and target.IsValid() else None
  del process

  # Lazy install: only arm the broker when we've actually stopped on one of
  # XeniOS JIT BRK instructions.
  if not exe_ctx or not _is_jit_brk_stop(exe_ctx):
    return
  install_xenios_jit_broker(debugger, "", None, result, None)
  _disable_autostart_hook_fast(debugger)
  result.AppendMessage("xenios-jit-broker: lazy autostart armed")


def __lldb_init_module(debugger, internal_dict):
  del internal_dict
  debugger.HandleCommand(
      "command script add -o -f "
      "xenios_ios_lldb_jit_broker.install_xenios_jit_broker "
      "xenios-jit-broker-install"
  )
  debugger.HandleCommand(
      "command script add -o -f "
      "xenios_ios_lldb_jit_broker.autostart_xenios_jit_broker "
      "xenios-jit-broker-autostart"
  )
  debugger.HandleCommand(
      "command script add -o -f "
      "xenios_ios_lldb_jit_broker.install_autostart_xenios_jit_broker "
      "xenios-jit-broker-autostart-install"
  )
  debugger.HandleCommand(
      "command script add -o -f "
      "xenios_ios_lldb_jit_broker.reset_xenios_jit_broker_hooks "
      "xenios-jit-broker-reset-hooks"
  )
  print("xenios-ios-lldb-jit-broker loaded. Run: xenios-jit-broker-install")
