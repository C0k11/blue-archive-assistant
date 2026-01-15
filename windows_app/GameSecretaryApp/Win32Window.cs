using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows;
using System.Windows.Interop;

namespace GameSecretaryApp;

public static class Win32Window
{
    public struct Rect
    {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;

        public int Width => Right - Left;
        public int Height => Bottom - Top;
    }

    private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")] private static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
    [DllImport("user32.dll")] private static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
    [DllImport("user32.dll")] private static extern int GetWindowTextLength(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool GetClientRect(IntPtr hWnd, out Rect lpRect);
    [DllImport("user32.dll")] private static extern bool ClientToScreen(IntPtr hWnd, ref PointApi lpPoint);

    [DllImport("user32.dll")] private static extern int GetWindowLong(IntPtr hWnd, int nIndex);
    [DllImport("user32.dll")] private static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);

    [DllImport("user32.dll")] private static extern bool RegisterHotKey(IntPtr hWnd, int id, uint fsModifiers, uint vk);
    [DllImport("user32.dll")] private static extern bool UnregisterHotKey(IntPtr hWnd, int id);

    private const int GWL_EXSTYLE = -20;
    private const int WS_EX_TRANSPARENT = 0x20;
    private const int WS_EX_LAYERED = 0x80000;
    private const int WS_EX_TOOLWINDOW = 0x80;
    private const int WS_EX_NOACTIVATE = 0x08000000;

    public const int WM_HOTKEY = 0x0312;
    public const uint MOD_ALT = 0x0001;
    public const uint MOD_CONTROL = 0x0002;
    public const uint MOD_SHIFT = 0x0004;
    public const uint MOD_WIN = 0x0008;

    [StructLayout(LayoutKind.Sequential)]
    private struct PointApi
    {
        public int X;
        public int Y;
    }

    public static void MakeClickThrough(Window w)
    {
        var hwnd = new WindowInteropHelper(w).Handle;
        if (hwnd == IntPtr.Zero) return;
        var ex = GetWindowLong(hwnd, GWL_EXSTYLE);
        ex |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE;
        SetWindowLong(hwnd, GWL_EXSTYLE, ex);
    }

    public static bool TryRegisterHotKey(IntPtr hwnd, int id, uint modifiers, uint vk)
    {
        if (hwnd == IntPtr.Zero) return false;
        try
        {
            return RegisterHotKey(hwnd, id, modifiers, vk);
        }
        catch
        {
            return false;
        }
    }

    public static void TryUnregisterHotKey(IntPtr hwnd, int id)
    {
        if (hwnd == IntPtr.Zero) return;
        try
        {
            UnregisterHotKey(hwnd, id);
        }
        catch
        {
        }
    }

    public static bool TryGetClientRectOnScreen(string titleSubstring, out Rect rect)
    {
        rect = default;
        var target = FindWindowByTitleSubstring(titleSubstring);
        if (target == IntPtr.Zero) return false;

        if (!GetClientRect(target, out var cr)) return false;
        var p1 = new PointApi { X = cr.Left, Y = cr.Top };
        var p2 = new PointApi { X = cr.Right, Y = cr.Bottom };
        if (!ClientToScreen(target, ref p1)) return false;
        if (!ClientToScreen(target, ref p2)) return false;
        rect = new Rect { Left = p1.X, Top = p1.Y, Right = p2.X, Bottom = p2.Y };
        return rect.Width > 0 && rect.Height > 0;
    }

    public static IntPtr FindWindowByTitleSubstring(string titleSubstring)
    {
        IntPtr found = IntPtr.Zero;
        var needle = (titleSubstring ?? "").Trim();
        if (needle.Length == 0) return IntPtr.Zero;

        EnumWindows((h, _) =>
        {
            try
            {
                if (!IsWindowVisible(h)) return true;
                var len = GetWindowTextLength(h);
                if (len <= 0) return true;
                var sb = new StringBuilder(len + 1);
                GetWindowText(h, sb, sb.Capacity);
                var title = sb.ToString();
                if (title.IndexOf(needle, StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    found = h;
                    return false;
                }
            }
            catch
            {
            }
            return true;
        }, IntPtr.Zero);

        return found;
    }
}
