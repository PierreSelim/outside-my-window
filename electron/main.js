const { app, BrowserWindow, globalShortcut } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const net = require('net');
const http = require('http');
const readline = require('readline');

let pythonProcess = null;

function refreshData(port) {
  const req = http.request({ host: '127.0.0.1', port, path: '/api/refresh', method: 'POST' });
  req.on('error', (err) => console.error('[refresh]', err.message));
  req.end();
}

function startServer() {
  return new Promise((resolve, reject) => {
    const proc = spawn('uv', ['run', 'python', path.join(__dirname, '..', 'server.py')], {
      cwd: path.join(__dirname, '..'),
    });
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

app.whenReady().then(async () => {
  const port = await startServer();
  await waitForServer(port);
  const win = new BrowserWindow({ width: 1280, height: 900 });
  win.loadURL(`http://127.0.0.1:${port}`);

  globalShortcut.register('CommandOrControl+Shift+R', () => {
    refreshData(port);
    win.webContents.reload();
  });
});

app.on('will-quit', () => {
  if (pythonProcess) pythonProcess.kill();
  globalShortcut.unregisterAll();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
