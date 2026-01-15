using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Interop;
using Microsoft.Web.WebView2.Core;

namespace GameSecretaryApp;

public partial class MainWindow : Window
{
    private const int HotkeyIdToggleOverlay = 0xA115;
    private OverlayWindow? _overlay;
    private IntPtr _hwnd;
    private HwndSource? _source;

    public MainWindow()
    {
        InitializeComponent();
        SourceInitialized += OnSourceInitialized;
        Loaded += OnLoaded;
        Closing += OnClosing;
    }

    private void OnSourceInitialized(object? sender, EventArgs e)
    {
        try
        {
            _hwnd = new WindowInteropHelper(this).Handle;
            _source = HwndSource.FromHwnd(_hwnd);
            _source?.AddHook(WndProc);

            var vk = (uint)KeyInterop.VirtualKeyFromKey(Key.O);
            Win32Window.TryRegisterHotKey(
                _hwnd,
                HotkeyIdToggleOverlay,
                Win32Window.MOD_CONTROL | Win32Window.MOD_SHIFT,
                vk
            );
        }
        catch
        {
        }
    }

    private IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled)
    {
        if (msg == Win32Window.WM_HOTKEY)
        {
            try
            {
                var id = wParam.ToInt32();
                if (id == HotkeyIdToggleOverlay)
                {
                    ToggleOverlay();
                    handled = true;
                }
            }
            catch
            {
            }
        }
        return IntPtr.Zero;
    }

    private void ToggleOverlay()
    {
        try
        {
            if (_overlay == null)
            {
                _overlay = new OverlayWindow
                {
                    TargetWindowTitleSubstring = "Blue Archive",
                };
                _overlay.Hide();
            }

            if (_overlay.IsVisible)
            {
                _overlay.Hide();
            }
            else
            {
                _overlay.Show();
            }
        }
        catch
        {
        }
    }

    private async void OnLoaded(object sender, RoutedEventArgs e)
    {
        Title = "AI Game Secretary (Starting backend...)";

        var userData = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GameSecretaryApp",
            "WebView2"
        );
        Directory.CreateDirectory(userData);
        var env = await CoreWebView2Environment.CreateAsync(null, userData);
        await WebView.EnsureCoreWebView2Async(env);

        try
        {
            await BackendManager.Instance.StartAsync();
        }
        catch (Exception ex)
        {
            Title = "AI Game Secretary (Backend failed)";
            MessageBox.Show(ex.Message, "Backend start failed", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        Title = "AI Game Secretary (Warming up model...)";
        try
        {
            await BackendManager.Instance.WarmupLocalVlmAsync(timeoutSec: 1800);
        }
        catch
        {
        }

        Title = "AI Game Secretary";
        WebView.CoreWebView2.Navigate(BackendManager.Instance.DashboardUrl);
    }

    private void OnClosing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        try
        {
            BackendManager.Instance.Stop();
        }
        catch
        {
        }

        try { Win32Window.TryUnregisterHotKey(_hwnd, HotkeyIdToggleOverlay); } catch { }
        try { _source?.RemoveHook(WndProc); } catch { }
        try { _overlay?.Close(); } catch { }
    }
}
