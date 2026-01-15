import ctypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scripts.win_capture import capture_client, find_window_by_title_substring, get_client_rect_on_screen


user32 = ctypes.WinDLL("user32", use_last_error=True)


@dataclass
class WindowTarget:
    title_substring: str
    hwnd: int


class WindowsInput:
    def __init__(self, *, title_substring: str):
        self.title_substring = (title_substring or "").strip()
        self._target: Optional[WindowTarget] = None

    def _resolve(self) -> WindowTarget:
        if self._target is not None:
            return self._target
        hwnd = find_window_by_title_substring(self.title_substring)
        if hwnd is None:
            raise RuntimeError(f"window not found: {self.title_substring}")
        self._target = WindowTarget(title_substring=self.title_substring, hwnd=hwnd)
        return self._target

    def _focus(self) -> int:
        t = self._resolve()
        try:
            user32.SetForegroundWindow(ctypes.c_void_p(t.hwnd))
        except Exception:
            pass
        return t.hwnd

    def screenshot_client(self, out_path: str) -> None:
        t = self._resolve()
        img = capture_client(t.hwnd)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

    def _client_to_screen(self, x: int, y: int) -> tuple[int, int]:
        t = self._resolve()
        r = get_client_rect_on_screen(t.hwnd)
        return (int(r.left + x), int(r.top + y))

    def click_client(self, x: int, y: int) -> None:
        self._focus()
        sx, sy = self._client_to_screen(int(x), int(y))
        user32.SetCursorPos(int(sx), int(sy))
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        user32.mouse_event(0x0004, 0, 0, 0, 0)

    def swipe_client(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        self._focus()
        sx1, sy1 = self._client_to_screen(int(x1), int(y1))
        sx2, sy2 = self._client_to_screen(int(x2), int(y2))

        user32.SetCursorPos(int(sx1), int(sy1))
        user32.mouse_event(0x0002, 0, 0, 0, 0)

        steps = 20
        dur = max(1, int(duration_ms))
        for i in range(1, steps + 1):
            t = i / steps
            sx = int(sx1 + (sx2 - sx1) * t)
            sy = int(sy1 + (sy2 - sy1) * t)
            user32.SetCursorPos(int(sx), int(sy))
            time.sleep(dur / 1000.0 / steps)

        user32.mouse_event(0x0004, 0, 0, 0, 0)

    def press_escape(self) -> None:
        self._focus()
        VK_ESCAPE = 0x1B
        KEYEVENTF_KEYUP = 0x0002
        user32.keybd_event(VK_ESCAPE, 0, 0, 0)
        user32.keybd_event(VK_ESCAPE, 0, KEYEVENTF_KEYUP, 0)
