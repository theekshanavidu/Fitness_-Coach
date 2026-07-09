import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

const releaseDir = path.resolve('release');
const destAppDir = path.join(releaseDir, 'AuraFit-win32-x64');

try {
  console.log('1. Building web bundle using Vite...');
  execSync('npm run build', { stdio: 'inherit' });

  console.log('2. Preparing release directory...');
  if (fs.existsSync(releaseDir)) {
    fs.rmSync(releaseDir, { recursive: true, force: true });
  }
  fs.mkdirSync(destAppDir, { recursive: true });

  console.log('3. Copying prebuilt Electron binaries...');
  const electronDist = path.resolve('node_modules/electron/dist');
  fs.cpSync(electronDist, destAppDir, { recursive: true });

  console.log('4. Renaming executable to AuraFit.exe...');
  fs.renameSync(
    path.join(destAppDir, 'electron.exe'),
    path.join(destAppDir, 'AuraFit.exe')
  );

  console.log('5. Cleaning default Electron app resources...');
  const defaultAsar = path.join(destAppDir, 'resources/default_app.asar');
  if (fs.existsSync(defaultAsar)) {
    fs.unlinkSync(defaultAsar);
  }

  console.log('6. Creating application source bundle directories...');
  const appSourceDest = path.join(destAppDir, 'resources/app');
  fs.mkdirSync(appSourceDest, { recursive: true });

  console.log('7. Copying compiled Vite web assets...');
  fs.cpSync(path.resolve('dist'), path.join(appSourceDest, 'dist'), { recursive: true });

  console.log('8. Copying main process entry files...');
  fs.copyFileSync(path.resolve('main.js'), path.join(appSourceDest, 'main.js'));
  fs.copyFileSync(path.resolve('package.json'), path.join(appSourceDest, 'package.json'));

  // Copy .env key if present for runtime security reference
  if (fs.existsSync('.env')) {
    fs.copyFileSync('.env', path.join(appSourceDest, '.env'));
  }

  console.log('\n✨ Standalone Portable Windows PC Application successfully created!');
  console.log('📂 Location: release/AuraFit-win32-x64/');
  console.log('🚀 Run file: release/AuraFit-win32-x64/AuraFit.exe');
} catch (error) {
  console.error('Packaging failed:', error);
  process.exit(1);
}
