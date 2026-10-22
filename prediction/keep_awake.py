"""Ask Windows not to go to sleep while the calling process runs (the screen may still turn off).

Calls SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) on the main thread; the request ends when the
process exits. It only blocks idle sleep: closing the lid still follows the power plan's lid action. Does nothing on
other systems and never raises, so it cannot affect a training run.
"""
import sys


def keep_awake():
    if sys.platform != 'win32':
        return False
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001))
    except Exception:
        return False
