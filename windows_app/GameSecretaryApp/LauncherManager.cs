using System;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Threading;

namespace GameSecretaryApp;

public sealed class LauncherManager
{
    public static LauncherManager Instance { get; } = new LauncherManager();

    private Process? _proc;
    private Process? _llmProc;

    private LauncherManager() { }

    public string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (var i = 0; i < 12 && dir is not null; i++)
        {
            var main = Path.Combine(dir.FullName, "main.py");
            if (File.Exists(main))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }
        return null;
    }

    public void Start(LaunchConfig cfg)
    {
        throw new InvalidOperationException(
            "LauncherManager is deprecated. Use BackendManager (FastAPI backend) and start the VLM policy agent via dashboard.html (POST /api/v1/start). " +
            "The legacy main.py launcher requires ADB and is not used in the Windows workflow."
        );

        if (_proc is { HasExited: false })
        {
            if (!cfg.ForceRestart)
            {
                return;
            }
            Stop();
        }

        if (cfg.ForceRestart && _proc is null)
        {
            Stop();
        }

        var root = FindRepoRoot();
        if (string.IsNullOrWhiteSpace(root))
        {
            throw new InvalidOperationException("Cannot locate repo root (main.py not found). Run the app from within the repo, or package the backend with the app.");
        }

        NormalizeConfig(cfg);

        if (!cfg.NoLlm && cfg.StartLocalLlm)
        {
            EnsureLocalLlm(cfg, root);
        }

        Directory.CreateDirectory(Path.Combine(root, "logs"));
        var outLog = Path.Combine(root, "logs", "agent.out.log");
        var errLog = Path.Combine(root, "logs", "agent.err.log");

        var exe = "py";
        var args = $"main.py --llm-base-url {Quote(cfg.LlmBaseUrl)} --llm-model {Quote(cfg.ModelName)}";
        if (!string.IsNullOrWhiteSpace(cfg.AdbSerial))
        {
            // Pass via env
        }
        if (cfg.Steps > 0)
        {
            args += $" --steps {cfg.Steps}";
        }
        if (cfg.OdQueries is { Length: > 0 })
        {
            foreach (var q in cfg.OdQueries)
            {
                args += $" --od {Quote(q)}";
            }
        }

        var psi = new ProcessStartInfo
        {
            FileName = exe,
            Arguments = args,
            WorkingDirectory = root,
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        try
        {
            psi.Environment["LLM_BASE_URL"] = cfg.LlmBaseUrl;
            psi.Environment["LLM_MODEL"] = cfg.ModelName;
            if (!string.IsNullOrWhiteSpace(cfg.OllamaModelsDir))
            {
                psi.Environment["OLLAMA_MODELS"] = cfg.OllamaModelsDir;
            }
            if (!string.IsNullOrWhiteSpace(cfg.AdbSerial))
            {
                psi.Environment["ADB_SERIAL"] = cfg.AdbSerial;
            }
        }
        catch
        {
        }

        var p = new Process { StartInfo = psi, EnableRaisingEvents = true };
        p.OutputDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(outLog, ev.Data); };
        p.ErrorDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(errLog, ev.Data); };

        if (!p.Start())
        {
            throw new InvalidOperationException("Failed to start agent process");
        }
        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        _proc = p;
    }

    public void Stop()
    {
        try
        {
            if (_proc is null) return;
            if (!_proc.HasExited)
            {
                try { _proc.Kill(entireProcessTree: true); } catch { }
            }
        }
        finally
        {
            try { _proc?.Dispose(); } catch { }
            _proc = null;
        }

        try
        {
            if (_llmProc is { HasExited: false })
            {
                try { _llmProc.Kill(entireProcessTree: true); } catch { }
            }
        }
        finally
        {
            try { _llmProc?.Dispose(); } catch { }
            _llmProc = null;
        }
    }

    private static void NormalizeConfig(LaunchConfig cfg)
    {
        if (!string.IsNullOrWhiteSpace(cfg.LlmBaseUrl))
        {
            try
            {
                var uri = new Uri(cfg.LlmBaseUrl.Trim());
                cfg.LlmHost = uri.Host;
                cfg.LlmPort = uri.Port;
            }
            catch
            {
            }
        }

        if (cfg.LlmPort <= 0)
        {
            cfg.LlmPort = 11434;
        }

        if (!string.IsNullOrWhiteSpace(cfg.LlmHost) && cfg.LlmPort > 0)
        {
            cfg.LlmBaseUrl = $"http://{cfg.LlmHost}:{cfg.LlmPort}";
        }
    }

    private void EnsureLocalLlm(LaunchConfig cfg, string repoRoot)
    {
        if (TcpListening(cfg.LlmHost, cfg.LlmPort, timeoutMs: 400))
        {
            return;
        }

        var logDir = Path.Combine(repoRoot, "logs");
        Directory.CreateDirectory(logDir);
        var outLog = Path.Combine(logDir, "ollama.out.log");
        var errLog = Path.Combine(logDir, "ollama.err.log");

        var psi = new ProcessStartInfo
        {
            FileName = "ollama",
            Arguments = "serve",
            WorkingDirectory = repoRoot,
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        try
        {
            if (!string.IsNullOrWhiteSpace(cfg.OllamaModelsDir))
            {
                psi.Environment["OLLAMA_MODELS"] = cfg.OllamaModelsDir;
            }
        }
        catch
        {
        }

        var p = new Process { StartInfo = psi, EnableRaisingEvents = true };
        p.OutputDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(outLog, ev.Data); };
        p.ErrorDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(errLog, ev.Data); };

        try
        {
            if (!p.Start())
            {
                throw new InvalidOperationException("Failed to start Ollama");
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                "Failed to start Ollama. Make sure Ollama is installed and `ollama` is available in PATH.",
                ex
            );
        }

        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        _llmProc = p;

        var ok = WaitTcpPort(cfg.LlmHost, cfg.LlmPort, timeoutSec: 30, proc: p);
        if (!ok)
        {
            bool exited;
            int exitCode = -1;
            try
            {
                exited = p.HasExited;
                if (exited)
                {
                    try { exitCode = p.ExitCode; } catch { }
                }
            }
            catch
            {
                exited = false;
            }

            if (exited)
            {
                throw new InvalidOperationException(
                    $"Ollama process exited before opening {cfg.LlmHost}:{cfg.LlmPort} (exit={exitCode}). " +
                    $"Check logs\\ollama.err.log for details."
                );
            }

            throw new TimeoutException($"Ollama did not open {cfg.LlmHost}:{cfg.LlmPort} within 30s (see logs\\ollama.*.log)");
        }

        if (cfg.AutoPullModel)
        {
            EnsureOllamaModel(cfg, repoRoot);
        }
    }

    private static void EnsureOllamaModel(LaunchConfig cfg, string repoRoot)
    {
        if (string.IsNullOrWhiteSpace(cfg.ModelName))
        {
            return;
        }

        var logDir = Path.Combine(repoRoot, "logs");
        Directory.CreateDirectory(logDir);
        var outLog = Path.Combine(logDir, "ollama_pull.out.log");
        var errLog = Path.Combine(logDir, "ollama_pull.err.log");

        var psi = new ProcessStartInfo
        {
            FileName = "ollama",
            Arguments = $"pull {Quote(cfg.ModelName)}",
            WorkingDirectory = repoRoot,
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        try
        {
            if (!string.IsNullOrWhiteSpace(cfg.OllamaModelsDir))
            {
                psi.Environment["OLLAMA_MODELS"] = cfg.OllamaModelsDir;
            }
        }
        catch
        {
        }

        var p = new Process { StartInfo = psi, EnableRaisingEvents = true };
        p.OutputDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(outLog, ev.Data); };
        p.ErrorDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(errLog, ev.Data); };

        if (!p.Start())
        {
            throw new InvalidOperationException("Failed to start `ollama pull`");
        }
        p.BeginOutputReadLine();
        p.BeginErrorReadLine();

        if (!p.WaitForExit(1000 * 60 * 60))
        {
            try { p.Kill(entireProcessTree: true); } catch { }
            throw new TimeoutException("ollama pull timed out (see logs\\ollama_pull.*.log)");
        }

        if (p.ExitCode != 0)
        {
            throw new InvalidOperationException($"ollama pull failed (exit={p.ExitCode}). See logs\\ollama_pull.err.log");
        }
    }

    private static bool TcpListening(string host, int port, int timeoutMs)
    {
        try
        {
            using var client = new TcpClient();
            var task = client.ConnectAsync(host, port);
            return task.Wait(timeoutMs) && client.Connected;
        }
        catch
        {
            return false;
        }
    }

    private static bool WaitTcpPort(string host, int port, int timeoutSec, Process? proc)
    {
        var t0 = DateTime.UtcNow;
        while ((DateTime.UtcNow - t0).TotalSeconds < timeoutSec)
        {
            try
            {
                if (proc is { HasExited: true })
                {
                    return false;
                }
            }
            catch
            {
            }

            if (TcpListening(host, port, timeoutMs: 400))
            {
                return true;
            }
            Thread.Sleep(350);
        }
        return false;
    }

    private static void AppendLineSafe(string path, string line)
    {
        try
        {
            if (line != null)
            {
                line = line.Replace("\0", "");
            }
            File.AppendAllText(path, (line ?? "") + Environment.NewLine);
        }
        catch
        {
        }
    }

    private static string Quote(string s)
    {
        if (string.IsNullOrWhiteSpace(s)) return "\"\"";
        if (s.Contains(' ')) return "\"" + s + "\"";
        return s;
    }
}
