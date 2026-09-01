const { app, BrowserWindow, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const net = require('net');
const http = require('http');
const readline = require('readline');

let pythonProcess = null;

// Packaged: the PyInstaller sidecar shipped in resources/server, no Python on the machine needed.
// Development: the checkout's own interpreter through uv.
function serverCommand() {
  if (app.isPackaged) {
    const exe = process.platform === 'win32' ? 'omw-server.exe' : 'omw-server';
    return { cmd: path.join(process.resourcesPath, 'server', exe), args: [], cwd: process.resourcesPath };
  }
  const root = path.join(__dirname, '..');
  return { cmd: 'uv', args: ['run', 'python', path.join(root, 'server.py')], cwd: root };
}

function refreshData(port) {
  const req = http.request({ host: '127.0.0.1', port, path: '/api/refresh', method: 'POST' });
  req.on('error', (err) => console.error('[refresh]', err.message));
  req.end();
}

function startServer() {
  return new Promise((resolve, reject) => {
    const { cmd, args, cwd } = serverCommand();
    // An installed app must not write downloads next to its own binary.
    const env = { ...process.env, OMW_CACHE_DIR: path.join(app.getPath('userData'), 'cache') };
    const proc = spawn(cmd, args, { cwd, env });
    pythonProcess = proc;

    let settled = false;
    const settle = (fn) => { if (!settled) { settled = true; fn(); } };

    readline.createInterface({ input: proc.stdout }).once('line', (line) => {
      const port = parseInt(line.trim(), 10);
      if (Number.isInteger(port) && port > 0) settle(() => resolve(port));
      else settle(() => reject(new Error(`Unexpected port value: ${line}`)));
    });

    proc.stderr.on('data', (d) => console.error('[server]', d.toString()));
    proc.on('error', (err) => settle(() => reject(err)));
    proc.on('close', (code) => settle(() => reject(new Error(`Server exited (code ${code}) before starting`))));
  });
}

function waitForServer(port, timeout = 30000) {
  const deadline = Date.now() + timeout;
  return new Promise((resolve, reject) => {
    function attempt() {
      const socket = net.createConnection(port, '127.0.0.1');
      socket.once('connect', () => { socket.destroy(); resolve(); });
      socket.once('error', () => {
        socket.destroy();
        if (Date.now() > deadline) return reject(new Error('Server did not start in time'));
        setTimeout(attempt, 200);
      });
    }
    attempt();
  });
}

// A scoped handler, not globalShortcut: a weather viewer has no business holding
// Ctrl+Shift+R hostage system-wide. preventDefault also stops the default menu's force-reload,
// which would reload the page without refreshing the data behind it.
function registerRefreshShortcut(win, port) {
  win.webContents.on('before-input-event', (event, input) => {
    const chord = input.type === 'keyDown' && input.shift && (input.control || input.meta);
    if (!chord || input.key.toLowerCase() !== 'r') return;
    event.preventDefault();
    refreshData(port);
    win.webContents.reload();
  });
}

app.whenReady().then(async () => {
  const port = await startServer();
  await waitForServer(port);
  const win = new BrowserWindow({
    width: 1280,
    height: 900,
    title: 'Outside My Window',
    webPreferences: { preload: path.join(__dirname, 'preload.js'), nodeIntegration: false, contextIsolation: true },
  });
  win.loadURL(`http://127.0.0.1:${port}`);
  registerRefreshShortcut(win, port);
}).catch((err) => {
  // startServer and waitForServer produce specific messages; without this they reject into
  // nowhere and the app hangs with no window and no explanation.
  dialog.showErrorBox('Outside My Window', `The data server did not start.

${err.message}`);
  app.quit();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
  if (pythonProcess) pythonProcess.kill();
});

