const fs = require('fs');
const path = require('path');

const srcDir = path.join(__dirname, 'dist');
const destDir = path.join(__dirname, 'android', 'app', 'src', 'main', 'assets');
const logoSrc = path.join(__dirname, 'src', 'images (1).jpg');
const mipmapDir = path.join(__dirname, 'android', 'app', 'src', 'main', 'res', 'mipmap');
const mipmapRoundDir = path.join(__dirname, 'android', 'app', 'src', 'main', 'res', 'mipmap-round');

function copyRecursiveSync(src, dest) {
  const exists = fs.existsSync(src);
  const stats = exists && fs.statSync(src);
  const isDirectory = exists && stats.isDirectory();
  if (isDirectory) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    fs.readdirSync(src).forEach((childItemName) => {
      copyRecursiveSync(path.join(src, childItemName), path.join(dest, childItemName));
    });
  } else {
    fs.copyFileSync(src, dest);
  }
}

// 1. Sync assets
if (fs.existsSync(destDir)) {
  fs.rmSync(destDir, { recursive: true, force: true });
}
fs.mkdirSync(destDir, { recursive: true });

if (fs.existsSync(srcDir)) {
  copyRecursiveSync(srcDir, destDir);
  console.log('✓ Compiled Vite assets copied to Android assets folder.');
} else {
  console.error('dist folder not found! Run npm run build first.');
}

// 2. Sync Launcher Icon
if (fs.existsSync(logoSrc)) {
  if (!fs.existsSync(mipmapDir)) fs.mkdirSync(mipmapDir, { recursive: true });
  if (!fs.existsSync(mipmapRoundDir)) fs.mkdirSync(mipmapRoundDir, { recursive: true });
  
  fs.copyFileSync(logoSrc, path.join(mipmapDir, 'ic_launcher.jpg'));
  fs.copyFileSync(logoSrc, path.join(mipmapRoundDir, 'ic_launcher_round.jpg'));
  console.log('✓ Launcher icons copied to Android res/mipmap folders.');
} else {
  console.warn('App logo source file not found at src/images (1).jpg');
}
