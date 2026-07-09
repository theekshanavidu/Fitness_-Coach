import { app, BrowserWindow } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

// Load environment variables in the main process
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 850,
    minWidth: 1000,
    minHeight: 700,
    title: "AuraFit - AI Fitness & Wellness Tracker",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  // Open devTools to troubleshoot if there are any rendering issues
  win.webContents.openDevTools();

  // Remove the default Electron menu bar for a cleaner look
  win.setMenuBarVisibility(false);

  // Load the compiled index.html from Vite's build output
  win.loadFile(path.join(__dirname, 'dist/index.html'));
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
