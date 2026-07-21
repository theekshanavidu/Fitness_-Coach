import { auth, db } from "./firebase-config.js";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  sendPasswordResetEmail,
  GoogleAuthProvider,
  signInWithPopup
} from "firebase/auth";
import {
  doc,
  setDoc,
  getDoc,
  updateDoc,
  collection,
  getDocs,
  query,
  orderBy,
  limit,
  addDoc,
  serverTimestamp,
  deleteDoc,
  onSnapshot
} from "firebase/firestore";
import { GoogleGenAI } from "@google/genai";
import { Chart } from "chart.js/auto";

// =========================================================================
// GEMINI AI INIT — Robust Multi-Key & Model Fallback Chain
// =========================================================================
const KEYS = [
  process.env.GEMINI_API_KEY_A,
  process.env.GEMINI_API_KEY_B,
  process.env.GEMINI_API_KEY_C,
  process.env.GEMINI_API_KEY_D,
  process.env.GEMINI_API_KEY_E
].filter(Boolean);

let activeKeyIndex = 0;

// Fallback model list as specified by the user
const MODEL_CHAIN = [
  "gemini-3.5-flash",
  "gemini-3.5-pro",
  "gemini-1.5-pro",   // fallback in case 3.5 pro is not available in free tier
  "gemini-2.5-flash",
  "gemini-1.5-flash"
];

// Helper: Try each model in the chain sequentially on the current key.
// If all models fail, shift to the next key and start trying models from index 0.
async function callWithFallback(apiFn) {
  if (KEYS.length === 0) {
    throw new Error("No Gemini API keys configured. Check your env settings.");
  }

  let lastError = null;

  for (let k = 0; k < KEYS.length; k++) {
    const keyIndex = (activeKeyIndex + k) % KEYS.length;
    const currentKey = KEYS[keyIndex];
    const client = new GoogleGenAI({ apiKey: currentKey });

    console.log(`[AuraFit AI] Trying Key Index ${keyIndex} (${currentKey.substring(0, 8)}...)`);

    for (const modelName of MODEL_CHAIN) {
      try {
        const result = await apiFn(client, modelName);
        activeKeyIndex = keyIndex; // update active key on success
        return result;
      } catch (err) {
        console.warn(`[AuraFit AI] Call failed for Key Index ${keyIndex} and Model ${modelName}:`, err.message || err);
        lastError = err;
      }
    }
    console.warn(`[AuraFit AI] All models exhausted for Key Index ${keyIndex}. Rotating key...`);
  }
  throw lastError;
}

// Generate a single response (non-streaming) with fallback
async function aiGenerate(prompt, systemInstruction) {
  return await callWithFallback(async (client, modelName) => {
    const config = {};
    if (systemInstruction) config.systemInstruction = systemInstruction;
    const response = await client.models.generateContent({
      model: modelName,
      contents: prompt,
      config,
    });
    return response.text;
  });
}

// Create a stateful multi-turn chat (returns the chat session)
async function aiChat(systemInstruction, history = []) {
  return await callWithFallback(async (client, modelName) => {
    const config = {};
    if (systemInstruction) config.systemInstruction = systemInstruction;
    const session = client.chats.create({ model: modelName, history, config });
    // Attach details to session for mid-chat recovery
    session.modelName = modelName;
    session.clientInstance = client;
    return session;
  });
}

/** Detects if the prompt is asking to draw/generate an image */
function isImageGenerationPrompt(text) {
  if (!text) return false;
  const t = text.toLowerCase().trim();
  if (t.startsWith("/draw") || t.startsWith("/generate") || t.startsWith("/image")) return true;
  
  const keywords = [
    "draw a", "generate an image", "create a picture", "create an image",
    "sketch a", "paint a", "photo of", "රූපයක් අඳින්න", "පින්තූරයක් අඳින්න", 
    "පින්තූරයක් හදන්න", "රූපයක් හදන්න", "image එකක් හදන්න", "drawing of"
  ];
  return keywords.some(kw => t.includes(kw));
}

/** Converts imageBytes (Uint8Array or base64 string) to a safe base64 data URL */
function imageBytesToDataUrl(imageBytes, mimeType = "image/jpeg") {
  if (typeof imageBytes === "string") {
    // Already a base64 string
    return `data:${mimeType};base64,${imageBytes}`;
  }
  // Uint8Array — convert to base64
  let binary = "";
  const bytes = new Uint8Array(imageBytes);
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return `data:${mimeType};base64,${btoa(binary)}`;
}

// Image generation model chain — tries in order until one works
const IMAGE_GEN_MODELS = [
  "gemini-2.5-flash-image",
  "gemini-2.0-flash-exp",
  "gemini-2.0-flash",
];

/** Generates an image using Pollinations.ai (free, no API key needed) */
async function generateAiImage(promptText) {
  // Strip trigger prefix if present
  let cleanPrompt = promptText;
  if (promptText.toLowerCase().startsWith("/draw"))          cleanPrompt = promptText.slice(5).trim();
  else if (promptText.toLowerCase().startsWith("/generate")) cleanPrompt = promptText.slice(9).trim();
  else if (promptText.toLowerCase().startsWith("/image"))    cleanPrompt = promptText.slice(6).trim();
  if (!cleanPrompt) cleanPrompt = promptText;

  console.log(`[AuraFit ImgGen] 🎨 Generating via Pollinations.ai: "${cleanPrompt}"`);

  // Pollinations.ai — completely free, no API key required
  const seed = Math.floor(Math.random() * 999999);
  const url = `https://image.pollinations.ai/prompt/${encodeURIComponent(cleanPrompt)}?width=768&height=768&nologo=true&seed=${seed}`;

  console.log(`[AuraFit ImgGen] 🌐 URL: ${url}`);

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Pollinations fetch failed: ${res.status} ${res.statusText}`);

  const blob = await res.blob();
  const mimeType = blob.type || "image/jpeg";

  // Convert blob → base64 data URL for rendering + Firestore storage
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = () => { 
      console.log(`[AuraFit ImgGen] ✅ Image ready! size=${blob.size} bytes`);
      resolve(reader.result); 
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}




// Helper to construct structured health context & logs instruction
function getSystemInstruction(isSi, age, h, w, wa, ch, bmi, logs) {
  let progressStr = "No historical log data available yet.";
  if (logs && logs.length > 0) {
    progressStr = logs.map(l => {
      const d = l.timestamp?.seconds ? new Date(l.timestamp.seconds * 1000) : new Date(l.timestamp || l.id);
      const dateStr = d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
      return `Date: ${dateStr}, Weight: ${l.weight}kg, Waist: ${l.waist}in, Chest: ${l.chest}in`;
    }).join(" | ");

    const firstLog = logs[0];
    const lastLog = logs[logs.length - 1];
    if (logs.length > 1) {
      const weightDiff = (lastLog.weight - firstLog.weight).toFixed(1);
      const waistDiff = (lastLog.waist - firstLog.waist).toFixed(1);
      const chestDiff = (lastLog.chest - firstLog.chest).toFixed(1);
      
      progressStr += `\nOverall Changes (First log to Latest log): `;
      progressStr += `Weight: ${weightDiff > 0 ? '+' : ''}${weightDiff}kg, `;
      progressStr += `Waist: ${waistDiff > 0 ? '+' : ''}${waistDiff}in, `;
      progressStr += `Chest: ${chestDiff > 0 ? '+' : ''}${chestDiff}in.`;
    }
  }

  return `You are a professional fitness coach & health advisor.
User Profile:
- Name: ${userProfile?.name || "User"}
- Gender: ${userProfile?.gender || "not specified"}
- Age: ${age} Years
- Height: ${h} cm
- Latest Weight: ${w} kg
- Latest Waist Size: ${wa} inches
- Latest Chest Size: ${ch} inches
- Latest BMI: ${bmi}

User Historical Progress & Logs:
${progressStr}

Instructions:
1. Review all of the user's profile details, latest metrics, and historical progress.
2. When the user asks about their progress, weight loss, gains, waist reduction, or queries their history, refer directly to these stats and details.
3. Encourage the user based on their actual trends (e.g. if they lost weight or reduced waist size, praise them; if they gained, offer constructive advice).
4. Respond EXCLUSIVELY in ${isSi ? "Sinhala (සිංහල)" : "English"}. Never mix languages or respond in English if the language is Sinhala.
5. Be professional, concise, encouraging, and use clean Markdown.`;
}

// =========================================================================
// SESSION CACHE  (clears on tab close; persists across page navigations)
// =========================================================================
const CACHE_KEYS = {
  PROFILE:  "af_profile",
  DAILY:    "af_daily",
  HEIGHT:   "af_height",
  ALL_LOGS: "af_all_logs"
};

function cacheSet(key, data) {
  try { sessionStorage.setItem(key, JSON.stringify(data)); } catch (_) {}
}
function cacheGet(key) {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch (_) { return null; }
}
function cacheClear() {
  Object.values(CACHE_KEYS).forEach(k => sessionStorage.removeItem(k));
}

// =========================================================================
// STATE
// =========================================================================
let currentUserId  = null;
let userProfile    = null;
let currentChatSession = null;
let currentChatId = null;
let unsubscribeChats = null;
let currentChart   = null;
let aiNutritionAdviceTimeout = null;
let currentWeight  = null;
let currentHeight  = null;
let currentAge     = null;
let currentWaist   = null;
let currentChest   = null;
let currentBmi     = null;
let currentChatLogs = [];

// =========================================================================
// HELPERS
// =========================================================================
function calculateAge(dob) {
  if (!dob) return 0;
  const today = new Date();
  const birth = new Date(dob);
  let age = today.getFullYear() - birth.getFullYear();
  const m = today.getMonth() - birth.getMonth();
  if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--;
  return age >= 0 ? age : 0;
}

function formatMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g,    "<em>$1</em>")
    .replace(/^### (.*?)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*?)$/gm,  "<h2>$1</h2>")
    .replace(/^# (.*?)$/gm,   "<h1>$1</h1>")
    .replace(/^\s*[-*]\s+(.*?)$/gm, "<li>$1</li>")
    .replace(/\n/g, "<br>");
}

async function compressImage(file, maxWidth = 600, maxHeight = 600, quality = 0.7) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        let width = img.width;
        let height = img.height;
        
        if (width > height) {
          if (width > maxWidth) {
            height = Math.round((height * maxWidth) / width);
            width = maxWidth;
          }
        } else {
          if (height > maxHeight) {
            width = Math.round((width * maxHeight) / height);
            height = maxHeight;
          }
        }
        
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, width, height);
        
        const compressedBase64 = canvas.toDataURL("image/jpeg", quality);
        resolve(compressedBase64);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

function todayStr() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")}`;
}
function monthStr() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}`;
}

// =========================================================================
// CACHED FIRESTORE FETCHERS
// =========================================================================
async function fetchLatestDaily(uid) {
  const cached = cacheGet(CACHE_KEYS.DAILY);
  if (cached) return cached;
  const q = query(collection(db, "users", uid, "daily_logs"), orderBy("timestamp","desc"), limit(1));
  const snap = await getDocs(q);
  const data = snap.empty ? null : snap.docs[0].data();
  if (data) cacheSet(CACHE_KEYS.DAILY, data);
  return data;
}

async function fetchLatestHeight(uid) {
  const cached = cacheGet(CACHE_KEYS.HEIGHT);
  if (cached) return cached;
  const q = query(collection(db, "users", uid, "monthly_logs"), orderBy("timestamp","desc"), limit(1));
  const snap = await getDocs(q);
  const data = snap.empty ? null : snap.docs[0].data();
  if (data) cacheSet(CACHE_KEYS.HEIGHT, data);
  return data;
}

async function fetchAllLogs(uid) {
  const cached = cacheGet(CACHE_KEYS.ALL_LOGS);
  if (cached) return cached;
  const q = query(collection(db, "users", uid, "daily_logs"), orderBy("timestamp","asc"));
  const snap = await getDocs(q);
  const rows = [];
  snap.forEach(d => rows.push({ id: d.id, ...d.data() }));
  cacheSet(CACHE_KEYS.ALL_LOGS, rows);
  return rows;
}

// =========================================================================
// LANGUAGE TOGGLE  (header button present on every inner page)
// =========================================================================
function initLangToggle() {
  const btn = document.getElementById("btn-lang-toggle");
  if (!btn) return;

  function refreshBtn() {
    const isSi = userProfile && userProfile.language === "sinhala";
    btn.textContent = isSi ? "🌐 SI" : "🌐 EN";
    btn.classList.toggle("sinhala-active", isSi);
    btn.title = isSi ? "Switch to English" : "භාෂාව සිංහලට මාරු කරන්න";
  }

  refreshBtn(); // set initial state when profile is loaded

  btn.addEventListener("click", async () => {
    if (!currentUserId || !userProfile) return;
    const newLang = userProfile.language === "sinhala" ? "english" : "sinhala";

    // Optimistic update
    userProfile.language = newLang;
    refreshBtn();

    try {
      // Persist to Firestore (only update language field)
      await updateDoc(doc(db, "users", currentUserId), { language: newLang });
      // Update cache
      const cached = cacheGet(CACHE_KEYS.PROFILE);
      if (cached) { cached.language = newLang; cacheSet(CACHE_KEYS.PROFILE, cached); }

      // Update the profile select dropdown if on profile page
      const langSelect = document.getElementById("update-language");
      if (langSelect) langSelect.value = newLang;
      const dispLang = document.getElementById("profile-display-language");
      if (dispLang) dispLang.innerText = newLang;

      // Reload AI content for pages that show AI output
      const path = window.location.pathname;
      if (path.endsWith("ai-chat.html")) {
        loadAiPageData();
      } else if (path.endsWith("nutrition.html") && currentWeight && currentHeight) {
        const factor = window.getSelectedActivityFactor ? window.getSelectedActivityFactor() : 1.2;
        const goal   = window.getSelectedGoal          ? window.getSelectedGoal()           : "maintain";
        calculateNutrition(null, factor, goal);
      }
    } catch (err) {
      console.error("Language toggle save error:", err);
    }
  });
}

// =========================================================================
// MOBILE DRAWER SIDEBAR HANDLERS
// =========================================================================
function initMobileSidebar() {
  const toggleBtn = document.getElementById("btn-sidebar-toggle");
  const closeBtn  = document.getElementById("btn-sidebar-close");
  const overlay   = document.getElementById("sidebar-overlay");
  const sidebar   = document.querySelector(".sidebar");

  if (!sidebar) return;

  function openSidebar() {
    sidebar.classList.add("active");
    if (overlay) overlay.classList.add("active");
  }

  function closeSidebar() {
    sidebar.classList.remove("active");
    if (overlay) overlay.classList.remove("active");
  }

  if (toggleBtn) toggleBtn.addEventListener("click", openSidebar);
  if (closeBtn)  closeBtn.addEventListener("click", closeSidebar);
  if (overlay)   overlay.addEventListener("click", closeSidebar);

  // Close sidebar when clicking any navigation link
  const navLinks = sidebar.querySelectorAll(".nav-link");
  navLinks.forEach(link => {
    link.addEventListener("click", closeSidebar);
  });
}

// =========================================================================
// AUTH STATE + ROUTING
// =========================================================================
onAuthStateChanged(auth, async (user) => {
  const path = window.location.pathname;
  const isAuth = path.endsWith("index.html") || path === "/" || path.endsWith("/");

  if (user) {
    currentUserId = user.uid;

    // --- Load profile (cache first) ---
    let profileData = cacheGet(CACHE_KEYS.PROFILE);
    if (!profileData) {
      try {
        const snap = await getDoc(doc(db, "users", currentUserId));
        if (snap.exists()) {
          profileData = snap.data();
          cacheSet(CACHE_KEYS.PROFILE, profileData);
        }
      } catch (e) { console.error("Profile fetch error:", e); }
    }

    if (profileData) {
      userProfile = profileData;
      if (!userProfile.gender)   userProfile.gender   = "male";
      if (!userProfile.language) userProfile.language = "english";
      userProfile.age = calculateAge(userProfile.dob);
      updateUIWithProfile();
      initLangToggle();
      initMobileSidebar();
    }

    if (isAuth) {
      window.location.href = "dashboard.html";
    } else {
      if      (path.endsWith("dashboard.html")) loadDashboardData();
      else if (path.endsWith("ai-chat.html"))   loadAiPageData();
      else if (path.endsWith("nutrition.html")) loadNutritionPageData();
      else if (path.endsWith("profile.html"))   loadProfilePageData();
    }
  } else {
    currentUserId = null;
    userProfile   = null;
    cacheClear();
    if (!isAuth) window.location.href = "index.html";
  }
});

function updateUIWithProfile() {
  if (!userProfile) return;
  const el = (id) => document.getElementById(id);
  if (el("sidebar-user-name"))    el("sidebar-user-name").innerText    = userProfile.name;
  if (el("sidebar-user-age"))     el("sidebar-user-age").innerText     = `Age: ${userProfile.age} Years`;
  if (el("greeting-name"))        el("greeting-name").innerText        = userProfile.name;
  if (el("greeting-name-ai"))     el("greeting-name-ai").innerText     = `${userProfile.name} (Age: ${userProfile.age})`;
  if (el("greeting-name-nutrition")) el("greeting-name-nutrition").innerText = `${userProfile.name} (Age: ${userProfile.age})`;
}

// =========================================================================
// REGISTER — age preview
// =========================================================================
const registerDob = document.getElementById("register-dob");
if (registerDob) {
  registerDob.addEventListener("input", () => {
    const dob = registerDob.value;
    const container = document.getElementById("age-preview-container");
    const span      = document.getElementById("calculated-age");
    if (dob) { span.innerText = calculateAge(dob); container.style.display = "block"; }
    else      { container.style.display = "none"; }
  });
}

// =========================================================================
// AUTH ACTIONS
// =========================================================================
const btnLogin = document.getElementById("btn-login-submit");
if (btnLogin) {
  btnLogin.addEventListener("click", async () => {
    const email = document.getElementById("login-email").value.trim();
    const pass  = document.getElementById("login-password").value;
    if (!email || !pass) { alert("Please fill in email and password."); return; }
    try {
      btnLogin.disabled = true; btnLogin.innerText = "Logging in…";
      await signInWithEmailAndPassword(auth, email, pass);
    } catch (err) {
      alert("Incorrect credentials. Please try again.");
      btnLogin.disabled = false; btnLogin.innerText = "Log In";
    }
  });
}

const btnForgot = document.getElementById("btn-forgot");
if (btnForgot) {
  btnForgot.addEventListener("click", async () => {
    const email = document.getElementById("login-email").value.trim();
    if (!email) { alert("Enter your email above first."); return; }
    try {
      btnForgot.innerText = "Sending…";
      await sendPasswordResetEmail(auth, email);
      alert(`Password reset link sent to ${email}. Check your inbox.`);
    } catch (err) {
      alert("Failed to send reset link. Check the email address.");
    } finally { btnForgot.innerText = "Forgot Password?"; }
  });
}

const btnGoogle = document.getElementById("btn-google-signin");
if (btnGoogle) {
  btnGoogle.addEventListener("click", async () => {
    try {
      btnGoogle.disabled = true; btnGoogle.innerText = "Connecting…";
      const res = await signInWithPopup(auth, new GoogleAuthProvider());
      const snap = await getDoc(doc(db, "users", res.user.uid));
      if (!snap.exists()) {
        const dob = "1995-01-01";
        await setDoc(doc(db, "users", res.user.uid), {
          name: res.user.displayName || "Google User",
          email: res.user.email, dob,
          age: calculateAge(dob), gender: "male", language: "english",
          createdAt: new Date()
        });
      }
    } catch (err) {
      alert("Google Sign‑In failed: " + err.message);
      btnGoogle.disabled = false; btnGoogle.innerText = "Continue with Google";
    }
  });
}

const btnRegister = document.getElementById("btn-register-submit");
if (btnRegister) {
  btnRegister.addEventListener("click", async () => {
    const name     = document.getElementById("register-name").value.trim();
    const email    = document.getElementById("register-email").value.trim();
    const pass     = document.getElementById("register-password").value;
    const dob      = document.getElementById("register-dob").value;
    const gender   = document.getElementById("register-gender").value;
    const language = document.getElementById("register-language").value;
    if (!name || !email || !pass || !dob) { alert("Please fill all fields."); return; }
    try {
      btnRegister.disabled = true; btnRegister.innerText = "Registering…";
      const res = await createUserWithEmailAndPassword(auth, email, pass);
      await setDoc(doc(db, "users", res.user.uid), {
        name, email, dob, age: calculateAge(dob), gender, language, createdAt: new Date()
      });
    } catch (err) {
      alert("Registration failed: " + err.message);
      btnRegister.disabled = false; btnRegister.innerText = "Sign Up";
    }
  });
}

const btnLogout = document.getElementById("btn-logout");
if (btnLogout) {
  btnLogout.addEventListener("click", async (e) => {
    e.preventDefault();
    cacheClear();
    await signOut(auth);
    window.location.href = "index.html";
  });
}

// =========================================================================
// DASHBOARD
// =========================================================================
const formDaily = document.getElementById("form-daily");
if (formDaily) {
  formDaily.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!currentUserId) return;
    const weight = document.getElementById("daily-weight").value;
    const waist  = document.getElementById("daily-waist").value;
    const chest  = document.getElementById("daily-chest").value;
    const date   = document.getElementById("daily-date").value;
    if (!date) { alert("Please choose a log date."); return; }
    const btn = document.getElementById("btn-daily-submit");
    btn.disabled = true; btn.innerText = "Saving…";
    try {
      await setDoc(doc(db, "users", currentUserId, "daily_logs", date), {
        weight: parseFloat(weight), waist: parseFloat(waist), chest: parseFloat(chest),
        timestamp: new Date(date)
      });
      // Invalidate caches so fresh data loads
      sessionStorage.removeItem(CACHE_KEYS.DAILY);
      sessionStorage.removeItem(CACHE_KEYS.ALL_LOGS);
      formDaily.reset();
      document.getElementById("daily-date").value = todayStr();
      loadDashboardData();
    } catch (err) { alert("Save failed. Try again."); }
    finally { btn.disabled = false; btn.innerText = "Save Daily Log"; }
  });
}

const formMonthly = document.getElementById("form-monthly");
if (formMonthly) {
  formMonthly.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!currentUserId) return;
    const height = document.getElementById("monthly-height").value;
    const month  = document.getElementById("monthly-month").value;
    if (!month) { alert("Please choose a month."); return; }
    const btn = document.getElementById("btn-monthly-submit");
    btn.disabled = true; btn.innerText = "Saving…";
    try {
      await setDoc(doc(db, "users", currentUserId, "monthly_logs", month), {
        height: parseFloat(height), timestamp: new Date(month + "-02")
      });
      sessionStorage.removeItem(CACHE_KEYS.HEIGHT);
      formMonthly.reset();
      document.getElementById("monthly-month").value = monthStr();
      loadDashboardData();
    } catch (err) { alert("Save failed. Try again."); }
    finally { btn.disabled = false; btn.innerText = "Save Height"; }
  });
}

async function loadDashboardData() {
  if (!currentUserId) return;

  // Pre-fill date inputs
  const di = document.getElementById("daily-date");
  if (di && !di.value) di.value = todayStr();
  const mi = document.getElementById("monthly-month");
  if (mi && !mi.value) mi.value = monthStr();

  try {
    const daily  = await fetchLatestDaily(currentUserId);
    const height = await fetchLatestHeight(currentUserId);

    const w = daily?.weight ?? null;
    const wa = daily?.waist ?? null;
    const ch = daily?.chest ?? null;
    const h  = height?.height ?? null;

    const el = (id) => document.getElementById(id);
    el("stat-weight").innerHTML = w  ? `${w} <span>kg</span>`  : `-- <span>kg</span>`;
    el("stat-waist").innerHTML  = wa ? `${wa} <span>in</span>` : `-- <span>in</span>`;
    el("stat-chest").innerHTML  = ch ? `${ch} <span>in</span>` : `-- <span>in</span>`;
    el("stat-height").innerHTML = h  ? `${h} <span>cm</span>`  : `-- <span>cm</span>`;

    if (w && h) {
      const bmi = w / ((h/100) ** 2);
      el("stat-bmi").innerText = bmi.toFixed(1);
      const s = el("stat-bmi-status");
      if      (bmi < 18.5) { s.innerText = "Underweight"; s.style.color = "var(--accent-cyan)"; }
      else if (bmi < 25)   { s.innerText = "Normal Weight"; s.style.color = "var(--accent-emerald)"; }
      else if (bmi < 30)   { s.innerText = "Overweight";  s.style.color = "#f97316"; }
      else                 { s.innerText = "Obese";       s.style.color = "var(--accent-rose)"; }
    } else {
      el("stat-bmi").innerText = "--";
      el("stat-bmi-status").innerText = "Weight & Height needed";
    }

    await renderProgressChart();
  } catch (err) { console.error("Dashboard load error:", err); }
}

async function renderProgressChart() {
  const canvas = document.getElementById("progressChart");
  if (!canvas) return;
  try {
    const rows = await fetchAllLogs(currentUserId);
    if (currentChart) { currentChart.destroy(); currentChart = null; }
    currentChart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels: rows.map(r => r.id),
        datasets: [
          { label:"Weight (kg)",    data: rows.map(r=>r.weight), borderColor:"#f43f5e", backgroundColor:"rgba(244,63,94,0.1)",  fill:true, tension:0.35, borderWidth:3, pointRadius:4, pointBackgroundColor:"#f43f5e" },
          { label:"Waist (inches)", data: rows.map(r=>r.waist),  borderColor:"#10b981", backgroundColor:"rgba(16,185,129,0.1)", fill:true, tension:0.35, borderWidth:3, pointRadius:4, pointBackgroundColor:"#10b981" },
          { label:"Chest (inches)", data: rows.map(r=>r.chest),  borderColor:"#06b6d4", backgroundColor:"rgba(6,182,212,0.1)",  fill:true, tension:0.35, borderWidth:3, pointRadius:4, pointBackgroundColor:"#06b6d4" }
        ]
      },
      options: {
        responsive:true, maintainAspectRatio:false,
        plugins: {
          legend: { position:"top", labels:{ color:"#cbd5e1", font:{ family:"Outfit", size:11, weight:"bold" } } },
          tooltip:{ mode:"index", intersect:false, backgroundColor:"rgba(12,16,27,0.9)", titleColor:"#06b6d4", bodyColor:"#f8fafc", borderColor:"rgba(255,255,255,0.08)", borderWidth:1 }
        },
        scales: {
          x:{ grid:{ color:"rgba(255,255,255,0.03)" }, ticks:{ color:"#64748b", font:{ family:"Plus Jakarta Sans", size:10 } } },
          y:{ grid:{ color:"rgba(255,255,255,0.03)" }, ticks:{ color:"#64748b", font:{ family:"Plus Jakarta Sans", size:10 } } }
        }
      }
    });
  } catch (err) { console.error("Chart error:", err); }
}

// =========================================================================
// AI CHAT PAGE
// =========================================================================
// =========================================================================
// AI CHAT PAGE (Multi-Session with Recent Chats Sidebar)
// =========================================================================

/** Copies the content of an AI chat bubble or panel to clipboard */
function copyChatMessage(btn) {
  const container = btn.closest(".chat-bubble") || btn.closest(".tune-results-box") || btn.closest(".glass-panel") || btn.parentElement.parentElement;
  if (!container) return;

  const bodyEl = container.querySelector(".msg-body") || container;
  let textToCopy = "";

  if (bodyEl) {
    const clone = bodyEl.cloneNode(true);
    const actions = clone.querySelector(".chat-bubble-actions");
    if (actions) actions.remove();
    textToCopy = clone.innerText.trim();
  } else {
    textToCopy = container.innerText.trim();
  }

  const isSi = userProfile?.language === "sinhala";

  const successHandler = () => {
    const originalContent = btn.dataset.origHtml || btn.innerHTML;
    btn.dataset.origHtml = originalContent;
    btn.classList.add("copied");
    btn.innerHTML = `✅ ${isSi ? "කොපි විය!" : "Copied!"}`;
    setTimeout(() => {
      btn.innerHTML = originalContent;
      btn.classList.remove("copied");
    }, 2000);
  };

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(textToCopy)
      .then(successHandler)
      .catch(() => {
        fallbackCopyText(textToCopy);
        successHandler();
      });
  } else {
    fallbackCopyText(textToCopy);
    successHandler();
  }
}

function fallbackCopyText(text) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand("copy"); } catch (e) {}
  document.body.removeChild(ta);
}

window.copyChatMessage = copyChatMessage;

/** Animates text appearing character-by-character / chunk-by-chunk like live AI message typing */
function typeWriterEffect(targetElement, fullText, isSi, onComplete) {
  if (!targetElement) return;
  targetElement.classList.remove("loading");
  
  targetElement.innerHTML = `<div class="msg-body"></div>`;
  const msgBody = targetElement.querySelector(".msg-body");
  
  let index = 0;
  const totalLength = fullText.length;
  // Calculate dynamic chunk size to ensure smooth typing animation (~2-3 seconds total duration)
  const chunkSize = Math.max(3, Math.ceil(totalLength / 80));

  const timer = setInterval(() => {
    index += chunkSize;
    if (index >= totalLength) {
      index = totalLength;
      clearInterval(timer);
      if (msgBody) msgBody.innerHTML = formatMarkdown(fullText);
      if (onComplete) onComplete();
    } else {
      const currentPartial = fullText.substring(0, index);
      if (msgBody) msgBody.innerHTML = formatMarkdown(currentPartial) + `<span class="typing-cursor">▌</span>`;
    }
    targetElement.scrollTop = targetElement.scrollHeight;
  }, 18);
}

/** Starts a blank new chat session */
async function createNewChat() {
  currentChatId = null;
  const chatEl = document.getElementById("chat-messages");
  if (chatEl) {
    const isSi = userProfile?.language === "sinhala";
    const welcome = isSi
      ? `ආයුබෝවන්! මම ඔබේ AuraFit AI සෞඛ්‍ය උපදේශකයායි.\nදත්ත ලැබුණා (වයස:${currentAge}, බර:${currentWeight}kg, උස:${currentHeight}cm, BMI:${currentBmi}).\nඔබට ගැළපෙන **Meal Plan** හෝ **Workout Plan** සකසා දිය හැකි. අද කුමක් අවශ්‍ය ද?`
      : `Hello! I am your AuraFit AI coach.\nProfile received — Age:${currentAge}, Weight:${currentWeight}kg, Height:${currentHeight}cm, BMI:${currentBmi}.\nHow can I help you today?`;
    const copyLabel = isSi ? "කොපි කරන්න" : "Copy";
    chatEl.innerHTML = `<div class="chat-bubble ai chat-bubble-markdown"><div class="msg-body">${formatMarkdown(welcome)}</div><div class="chat-bubble-actions"><button type="button" class="chat-copy-btn" onclick="copyChatMessage(this)">📋 ${copyLabel}</button></div></div>`;
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  // Reset chat header title
  const titleDisplay = document.getElementById("chat-title-display");
  if (titleDisplay) {
    titleDisplay.innerText = userProfile?.language === "sinhala" ? "💬 අලුත් Chat එකක්" : "💬 New Chat";
  }

  // Remove active class highlighting in the sidebar list
  document.querySelectorAll(".recent-chat-item").forEach(item => item.classList.remove("active"));

  // Build a clean, history-free Gemini chat session
  const isSi = userProfile?.language === "sinhala";
  const sysInstr = getSystemInstruction(isSi, currentAge, currentHeight, currentWeight, currentWaist, currentChest, currentBmi, currentChatLogs);
  currentChatSession = await aiChat(sysInstr, []);
}

/** RESTORES/LOADS a past chat conversation from Firestore */
async function loadChat(chatId) {
  if (!currentUserId) return;
  currentChatId = chatId;

  // Add visual active highlight in sidebar list
  document.querySelectorAll(".recent-chat-item").forEach(item => {
    if (item.getAttribute("data-chat-id") === chatId) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });

  const chatEl = document.getElementById("chat-messages");
  if (chatEl) {
    chatEl.innerHTML = `<div style="text-align:center; padding:2rem; color:var(--text-muted); font-size:0.9rem;">${
      userProfile?.language === "sinhala" ? "පෙර සංවාදය පූරණය වෙමින් පවතී…" : "Loading past conversation…"
    }</div>`;
  }

  // Close mobile sidebar drawer once loaded
  const sidebarEl = document.getElementById("chats-sidebar");
  if (sidebarEl) sidebarEl.classList.remove("open");

  // Get conversation title
  try {
    const chatDoc = await getDoc(doc(db, "users", currentUserId, "chats", chatId));
    if (chatDoc.exists()) {
      const titleDisplay = document.getElementById("chat-title-display");
      if (titleDisplay) {
        titleDisplay.innerText = chatDoc.data().title || "Chat";
      }
    }
  } catch (err) {
    console.error("Error fetching chat title:", err);
  }

  // Fetch messages nested under users/{userId}/chats/{chatId}/messages
  let dbHistory = [];
  try {
    const q = query(
      collection(db, "users", currentUserId, "chats", chatId, "messages"),
      orderBy("timestamp", "asc")
    );
    const snap = await getDocs(q);
    snap.forEach(docSnap => {
      dbHistory.push({ id: docSnap.id, ...docSnap.data() });
    });
  } catch (e) {
    console.error("Error loading chat history from Firestore:", e);
  }

  if (chatEl) {
    if (dbHistory.length === 0) {
      const welcome = userProfile?.language === "sinhala"
        ? `ආයුබෝවන්! මම ඔබේ AuraFit AI සෞඛ්‍ය උපදේශකයායි.\nදත්ත ලැබුණා (වයස:${currentAge}, බර:${currentWeight}kg, උස:${currentHeight}cm, BMI:${currentBmi}).\nඔබට ගැළපෙන **Meal Plan** හෝ **Workout Plan** සකසා දිය හැකි. අද කුමක් අවශ්‍ය ද?`
        : `Hello! I am your AuraFit AI coach.\nProfile received — Age:${currentAge}, Weight:${currentWeight}kg, Height:${currentHeight}cm, BMI:${currentBmi}.\nHow can I help you today?`;
      const copyLabel = userProfile?.language === "sinhala" ? "කොපි කරන්න" : "Copy";
      chatEl.innerHTML = `<div class="chat-bubble ai chat-bubble-markdown"><div class="msg-body">${formatMarkdown(welcome)}</div><div class="chat-bubble-actions"><button type="button" class="chat-copy-btn" onclick="copyChatMessage(this)">📋 ${copyLabel}</button></div></div>`;
    } else {
      const isSi = userProfile?.language === "sinhala";
      const copyLabel = isSi ? "කොපි කරන්න" : "Copy";
      chatEl.innerHTML = dbHistory.map(msg => {
        const imgs = msg.images || (msg.image ? [msg.image] : []);
        const imgMarkup = imgs.map(src => renderAttachmentMarkup(src, msg.sender)).join("");
        if (msg.sender === "user") {
          const textMarkup = msg.text ? `<div>${msg.text}</div>` : "";
          return `<div class="chat-bubble user">${textMarkup}${imgMarkup}</div>`;
        } else {
          const textMarkup = msg.text ? formatMarkdown(msg.text) : "";
          return `<div class="chat-bubble ai chat-bubble-markdown"><div class="msg-body">${textMarkup}${imgMarkup}</div><div class="chat-bubble-actions"><button type="button" class="chat-copy-btn" onclick="copyChatMessage(this)">📋 ${copyLabel}</button></div></div>`;
        }
      }).join("");
      chatEl.scrollTop = chatEl.scrollHeight;
    }
  }

  // Setup Gemini chat history block
  const isSi = userProfile?.language === "sinhala";
  const sysInstr = getSystemInstruction(isSi, currentAge, currentHeight, currentWeight, currentWaist, currentChest, currentBmi, currentChatLogs);
  
  const geminiHistory = dbHistory.map(msg => {
    const parts = [];
    if (msg.text) parts.push({ text: msg.text });
    const imgs = msg.images || (msg.image ? [msg.image] : []);
    for (const src of imgs) {
      const mimeType = src.match(/data:(.*?);base64/)?.[1] || "image/jpeg";
      const data = src.split(",")[1] || src;
      parts.push({ inlineData: { data, mimeType } });
    }
    return {
      role: msg.sender === "user" ? "user" : "model",
      parts: parts
    };
  });

  currentChatSession = await aiChat(sysInstr, geminiHistory);
}

/** Subscribes to live updates of the last 20 chats */
function bindRecentChatsListener() {
  if (unsubscribeChats) unsubscribeChats();

  const listEl = document.getElementById("recent-chats-list");
  if (!listEl) return;

  const q = query(
    collection(db, "users", currentUserId, "chats"),
    orderBy("updatedAt", "desc"),
    limit(20)
  );

  unsubscribeChats = onSnapshot(q, (snapshot) => {
    if (snapshot.empty) {
      listEl.innerHTML = `<li class="recent-chat-placeholder">${
        userProfile?.language === "sinhala" ? "පරණ chats කිසිවක් නැත" : "No recent chats"
      }</li>`;
      return;
    }

    listEl.innerHTML = snapshot.docs.map(docSnap => {
      const data = docSnap.data();
      const id = docSnap.id;
      const title = data.title || (userProfile?.language === "sinhala" ? "අලුත් Chat එකක්" : "New Chat");
      const activeClass = id === currentChatId ? "active" : "";
      return `
        <li class="recent-chat-item ${activeClass}" data-chat-id="${id}">
          <span class="recent-chat-link">${title}</span>
          <button class="recent-chat-delete-btn" data-chat-id="${id}" title="Delete Chat">🗑️</button>
        </li>
      `;
    }).join("");

    // Bind chat click
    listEl.querySelectorAll(".recent-chat-item").forEach(item => {
      item.addEventListener("click", (e) => {
        if (e.target.classList.contains("recent-chat-delete-btn")) return;
        const cid = item.getAttribute("data-chat-id");
        loadChat(cid);
      });
    });

    // Bind delete buttons
    listEl.querySelectorAll(".recent-chat-delete-btn").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const cid = btn.getAttribute("data-chat-id");
        const confirmMsg = userProfile?.language === "sinhala"
          ? "මෙම chat එක ස්ථිරවම මකා දැමීමට අවශ්‍යද?"
          : "Are you sure you want to delete this chat?";
        if (confirm(confirmMsg)) {
          try {
            await deleteDoc(doc(db, "users", currentUserId, "chats", cid));
            if (currentChatId === cid) {
              createNewChat();
            }
          } catch (err) {
            console.error("Error deleting chat document:", err);
          }
        }
      });
    });
  }, (error) => {
    console.error("Recent chats list sync error:", error);
  });
}

async function loadAiPageData() {
  if (!currentUserId) return;

  const isSi = userProfile?.language === "sinhala";

  try {
    const daily  = await fetchLatestDaily(currentUserId);
    const height = await fetchLatestHeight(currentUserId);
    currentChatLogs = await fetchAllLogs(currentUserId);

    const w  = daily?.weight  ?? "--";
    const wa = daily?.waist   ?? "--";
    const ch = daily?.chest   ?? "--";
    const h  = height?.height ?? "--";
    const age = userProfile?.age ?? "--";
    const bmi = (w !== "--" && h !== "--") ? (w / ((h/100)**2)).toFixed(1) : "--";

    // Set state variables for session tracking (e.g. mid-chat fallbacks)
    currentWeight = w;
    currentHeight = h;
    currentAge    = age;
    currentWaist  = wa;
    currentChest  = ch;
    currentBmi    = bmi;

    // Bind Mobile sidebar toggle listener
    const btnToggleChats = document.getElementById("btn-toggle-chats-sidebar");
    const sidebarChats = document.getElementById("chats-sidebar");
    if (btnToggleChats && sidebarChats) {
      btnToggleChats.addEventListener("click", (e) => {
        e.stopPropagation();
        sidebarChats.classList.toggle("open");
      });
      document.addEventListener("click", (e) => {
        if (sidebarChats.classList.contains("open") && !sidebarChats.contains(e.target) && e.target !== btnToggleChats) {
          sidebarChats.classList.remove("open");
        }
      });
    }

    // Bind New Chat button
    const btnNewChat = document.getElementById("btn-new-chat");
    if (btnNewChat) {
      btnNewChat.addEventListener("click", () => {
        createNewChat();
      });
    }

    // Load recent chats sidebar entries
    bindRecentChatsListener();

    // Start with a blank New Chat session by default
    await createNewChat();

    // AI Suggestions Board - load once daily per user (resets at midnight 12:00 AM)
    loadSuggestionsBoard(age, h, w, wa, ch, bmi, isSi);
  } catch (err) { console.error("AI page init error:", err); }
}

async function loadSuggestionsBoard(age, height, weight, waist, chest, bmi, isSi) {
  const board = document.getElementById("suggestions-board");
  if (!board) return;

  if (weight === "--" || height === "--") {
    board.innerHTML = isSi
      ? `<div class="suggestion-item"><div class="suggestion-icon">⚠️</div><div class="suggestion-content"><div class="suggestion-title">දත්ත නොමැත</div><div class="suggestion-desc">Dashboard එකෙන් උස හා බර ඇතුළත් කරන්න.</div></div></div>`
      : `<div class="suggestion-item"><div class="suggestion-icon">⚠️</div><div class="suggestion-content"><div class="suggestion-title">Missing Data</div><div class="suggestion-desc">Log your weight and height on the Dashboard first.</div></div></div>`;
    return;
  }

  // Get local date string YYYY-MM-DD to verify if 12:00 AM midnight has passed
  const now = new Date();
  const dateKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  const cacheKey = `aura_daily_suggestions_${currentUserId || 'default'}`;

  const renderItems = (items) => {
    board.innerHTML = items.map(item => {
      const [icon, title, desc] = item.split("|");
      return `
        <div class="suggestion-item">
          <div class="suggestion-icon">${(icon || '⚡').trim()}</div>
          <div class="suggestion-content">
            <div class="suggestion-title">${(title || '').trim()}</div>
            <div class="suggestion-desc">${(desc || '').trim()}</div>
          </div>
        </div>
      `;
    }).join("");
  };

  // 1. Check if cached suggestions for today already exist
  try {
    const cachedDataStr = localStorage.getItem(cacheKey);
    if (cachedDataStr) {
      const cachedData = JSON.parse(cachedDataStr);
      if (cachedData && cachedData.date === dateKey && Array.isArray(cachedData.items) && cachedData.items.length >= 3 && cachedData.isSi === isSi) {
        renderItems(cachedData.items);
        return; // Use cached suggestions for the entire day!
      }
    }
  } catch (e) {
    console.warn("Error reading suggestions cache:", e);
  }

  board.innerHTML = `<div class="suggestion-item" style="opacity:0.5"><div class="suggestion-icon">⏳</div><div class="suggestion-content"><div class="suggestion-title">${isSi ? "යෝජනා ලබා ගනිමින්…" : "Loading suggestions…"}</div></div></div>`;

  // Default suggestions to fall back on or backfill if AI fails
  const defaultSi = [
    "🏃‍♂️|ක්‍රියාශීලී වන්න|දිනපතා විනාඩි 30ක් වේගයෙන් ඇවිදින්න සිරුරේ ක්‍රියාශීලී බව වැඩි කර ගැනීමට.",
    "💧|ප්‍රමාණවත් ජලය පානය|දිනකට ජලය ලීටර් 3ක්වත් පානය කර ශරීරයේ උෂ්ණත්වය පාලනය කරන්න.",
    "🥗|සමබර පෝෂණය|ඔබේ ප්‍රධාන ආහාර වේල්වලට ප්‍රෝටීන් සහ එළවළු වැඩිපුර එකතු කරගන්න."
  ];
  const defaultEn = [
    "🏃‍♂️|Stay Active|Walk briskly for at least 30 minutes daily to boost metabolism.",
    "💧|Hydration|Drink at least 3 liters of water throughout the day to stay fit.",
    "🥗|Balanced Nutrition|Add lean proteins and fresh greens to your main meals."
  ];
  const defaults = isSi ? defaultSi : defaultEn;

  try {
    const prompt = `User stats: Age:${age}, Height:${height}cm, Weight:${weight}kg, Waist:${waist}in, Chest:${chest}in, BMI:${bmi}.
Generate exactly 3 personalized health/fitness suggestions.
Format EACH suggestion on a single line starting directly with Emoji as: Emoji|Title|Description
Do NOT include any number prefixes (like 1., 2., 3.) or markdown formatting.
Write ALL content 100% in ${isSi ? "Sinhala" : "English"}.
Output ONLY the 3 lines. No extra text.`;

    const resultText = await aiGenerate(prompt);
    
    // Clean up code block fences and number prefixes
    const cleanedText = resultText.replace(/```[a-z]*\n?/gi, "").replace(/```/g, "").trim();
    const parsedLines = cleanedText.split("\n")
      .map(line => line.replace(/^\d+[\.\-\s]*/, "").trim()) // remove leading numbering
      .filter(line => line.includes("|"));

    const finalItems = [];
    parsedLines.forEach(line => {
      const parts = line.split("|");
      if (parts.length >= 3 && parts[0].trim() && parts[1].trim() && parts[2].trim()) {
        finalItems.push(line);
      }
    });

    // Backfill with defaults to guarantee exactly 3 suggestions
    for (let i = finalItems.length; i < 3; i++) {
      finalItems.push(defaults[i]);
    }

    // Save to local cache for today until midnight
    try {
      localStorage.setItem(cacheKey, JSON.stringify({
        date: dateKey,
        items: finalItems,
        isSi: isSi
      }));
    } catch (e) {}

    renderItems(finalItems);

  } catch (err) {
    console.error("Suggestions generation error (loading defaults):", err);
    renderItems(defaults);
    try {
      localStorage.setItem(cacheKey, JSON.stringify({
        date: dateKey,
        items: defaults,
        isSi: isSi
      }));
    } catch (e) {}
  }
}

// Attachment handling state — supports multiple images and PDFs
let attachedImages = []; // array of base64 strings (both images and pdfs)

const btnAttach = document.getElementById("btn-chat-attach");
const fileInput = document.getElementById("chat-image-input");
const previewContainer = document.getElementById("chat-image-preview-container");

/** Renders attachment card UI for a base64 string */
function renderAttachmentMarkup(src, sender = "user") {
  if (src.startsWith("data:application/pdf")) {
    return `
      <a href="${src}" download="document.pdf" class="chat-pdf-attachment-card" style="display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,0.06);border:1px solid var(--border);padding:8px 12px;border-radius:10px;text-decoration:none;color:var(--text-primary);margin-top:6px;margin-bottom:6px;">
        <span style="font-size:1.5rem;color:#ef4444;">📄</span>
        <div style="text-align:left;">
          <div style="font-size:0.8rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:140px;color:var(--text-primary);">document.pdf</div>
          <div style="font-size:0.65rem;color:var(--text-muted);">Download / බාගත කරන්න</div>
        </div>
      </a>
    `;
  }

  // If the image was sent by the AI model, render a downloadable card overlay
  if (sender === "model") {
    return `
      <div class="generated-image-container" style="position: relative; max-width: 100%; border-radius: 12px; overflow: hidden; border: 1px solid var(--border); margin-top: 6px; margin-bottom: 6px;">
        <img src="${src}" alt="Generated Image" style="width: 100%; height: auto; display: block; max-width: 100% !important; max-height: none !important;">
        <a href="${src}" download="generated-image.jpg" class="download-badge-btn" style="position: absolute; bottom: 8px; right: 8px; background: rgba(12,16,27,0.85); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); border-radius: 6px; padding: 6px 10px; font-size: 0.7rem; text-decoration: none; display: flex; align-items: center; gap: 4px; font-weight: bold; cursor: pointer; transition: all 0.2s;">
          📥 Download / බාගත කරන්න
        </a>
      </div>
    `;
  }
  return `<img src="${src}" alt="Attached Image">`;
}

/** Renders all current thumbnails into the preview container */
function renderThumbnails() {
  if (!previewContainer) return;
  if (attachedImages.length === 0) {
    previewContainer.style.display = "none";
    previewContainer.innerHTML = "";
    if (btnAttach) btnAttach.classList.remove("has-file");
    return;
  }
  previewContainer.style.display = "flex";
  previewContainer.innerHTML = attachedImages.map((b64, idx) => {
    const isPdf = b64.startsWith("data:application/pdf");
    const previewContent = isPdf
      ? `<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;height:100%;background:rgba(239,68,68,0.15);color:#ef4444;font-size:0.7rem;font-weight:bold;gap:2px;">
           <span style="font-size:1.2rem;">📄</span>PDF
         </div>`
      : `<img src="${b64}" alt="Attachment ${idx + 1}">`;

    return `
      <div class="chat-thumb-card">
        ${previewContent}
        <button type="button" class="chat-thumb-remove" data-idx="${idx}" title="Remove">✕</button>
      </div>
    `;
  }).join("");

  // Bind remove buttons
  previewContainer.querySelectorAll(".chat-thumb-remove").forEach(btn => {
    btn.addEventListener("click", () => {
      const i = parseInt(btn.dataset.idx, 10);
      attachedImages.splice(i, 1);
      renderThumbnails();
      if (attachedImages.length === 0 && btnAttach) btnAttach.classList.remove("has-file");
    });
  });
  if (btnAttach) btnAttach.classList.add("has-file");
}

/** Process and push a selected File into attachedImages[], then re-render */
async function addAttachmentFile(file) {
  if (!file) return;

  if (file.type.startsWith("image/")) {
    try {
      const b64 = await compressImage(file, 600, 600, 0.7);
      attachedImages.push(b64);
      renderThumbnails();
    } catch (err) {
      console.error("Image compress error:", err);
    }
  } else if (file.type === "application/pdf") {
    // 2MB size limit to avoid memory pressure / Firestore Document size limit
    if (file.size > 2 * 1024 * 1024) {
      alert(userProfile?.language === "sinhala"
        ? "PDF ගොනුවේ ප්‍රමාණය 2MB ට වඩා අඩු විය යුතුය."
        : "PDF file size should be under 2MB."
      );
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) {
        attachedImages.push(e.target.result);
        renderThumbnails();
      }
    };
    reader.readAsDataURL(file);
  }
}

// File picker — multiple files (images & pdfs)
if (btnAttach && fileInput) {
  btnAttach.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", async () => {
    const files = Array.from(fileInput.files || []);
    for (const f of files) await addAttachmentFile(f);
    fileInput.value = ""; // reset
  });
}

// Clipboard paste — Ctrl+V on the text input
const chatMsgInput = document.getElementById("chat-user-message");
if (chatMsgInput) {
  chatMsgInput.addEventListener("paste", async (e) => {
    const items = Array.from(e.clipboardData?.items || []);
    const imageItem = items.find(it => it.type.startsWith("image/"));
    if (imageItem) {
      e.preventDefault();
      chatMsgInput.classList.add("paste-active");
      setTimeout(() => chatMsgInput.classList.remove("paste-active"), 600);
      const file = imageItem.getAsFile();
      await addAttachmentFile(file);
    }
  });
}


// AI Chat — send message (Streaming, Multimodal, Firestore logs)
const chatInputForm = document.getElementById("chat-input-form");
if (chatInputForm) {
  chatInputForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const msgInput = document.getElementById("chat-user-message");
    const userMsg  = msgInput.value.trim();
    
    // Enforce message constraint — need at least text or one image
    if (!userMsg && attachedImages.length === 0) return;

    const chatEl   = document.getElementById("chat-messages");
    const typingEl = document.getElementById("typing-indicator");

    // ── IMAGE GENERATION INTERCEPT (runs before chat-session check) ──────────
    if (userMsg && isImageGenerationPrompt(userMsg)) {
      // Show user bubble
      chatEl.innerHTML += `<div class="chat-bubble user"><div>${userMsg}</div></div>`;
      msgInput.value = "";
      chatEl.scrollTop = chatEl.scrollHeight;
      typingEl.style.display = "flex";
      chatEl.scrollTop = chatEl.scrollHeight;

      // Ensure chat doc exists in Firestore
      if (!currentChatId) {
        try {
          const initialTitle = userMsg.substring(0, 30) + (userMsg.length > 30 ? "..." : "");
          const newChatRef = await addDoc(collection(db, "users", currentUserId, "chats"), {
            title: initialTitle,
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp()
          });
          currentChatId = newChatRef.id;
          const titleDisplay = document.getElementById("chat-title-display");
          if (titleDisplay) titleDisplay.innerText = initialTitle;
        } catch (cErr) {
          console.error("Error creating chat doc for image gen:", cErr);
        }
      } else {
        try {
          await updateDoc(doc(db, "users", currentUserId, "chats", currentChatId), { updatedAt: serverTimestamp() });
        } catch (uErr) { /* ignore */ }
      }

      // Save user message to Firestore
      try {
        await addDoc(collection(db, "users", currentUserId, "chats", currentChatId, "messages"), {
          sender: "user", text: userMsg, timestamp: serverTimestamp()
        });
      } catch (fsErr) { console.error("Error saving user image-gen msg:", fsErr); }

      try {
        const dataUrl = await generateAiImage(userMsg);
        typingEl.style.display = "none";

        // Render the AI bubble with the generated image & download button
        const imgCaption = userProfile?.language === "sinhala"
          ? "🎨 ඔබගේ රූපය සාර්ථකව නිර්මාණය විය!"
          : "🎨 Your image has been generated!";
        const imgBubble = document.createElement("div");
        imgBubble.className = "chat-bubble ai";
        imgBubble.innerHTML = `<div style="margin-bottom:8px;font-weight:500;">${imgCaption}</div>${renderAttachmentMarkup(dataUrl, "model")}`;
        chatEl.appendChild(imgBubble);
        chatEl.scrollTop = chatEl.scrollHeight;

        // Save AI image response to Firestore
        try {
          await addDoc(collection(db, "users", currentUserId, "chats", currentChatId, "messages"), {
            sender: "model",
            text: imgCaption,
            images: [dataUrl],
            timestamp: serverTimestamp()
          });
        } catch (fsErr) { console.error("Error saving AI image to Firestore:", fsErr); }

      } catch (imgErr) {
        typingEl.style.display = "none";
        const errMsg = userProfile?.language === "sinhala"
          ? "⚠️ රූපය නිර්මාණය කිරීමේ දෝෂයක් ඇතිවිය. නැවත උත්සාහ කරන්න."
          : "⚠️ Image generation failed. Please try again.";
        chatEl.innerHTML += `<div class="chat-bubble ai" style="color:var(--accent-rose);">${errMsg}</div>`;
        console.error("Image generation error:", imgErr);
      }
      chatEl.scrollTop = chatEl.scrollHeight;
      return; // ← skip normal chat flow
    }
    // ── END IMAGE GENERATION INTERCEPT ──────────────────────────────────────

    // Display user message in UI immediately
    let userBubbleHtml = `<div class="chat-bubble user">`;
    if (userMsg) {
      userBubbleHtml += `<div>${userMsg}</div>`;
    }
    for (const b64 of attachedImages) {
      userBubbleHtml += renderAttachmentMarkup(b64);
    }
    userBubbleHtml += `</div>`;
    chatEl.innerHTML += userBubbleHtml;

    // Snapshot images & clear all inputs
    const activeImages = [...attachedImages];
    attachedImages = [];
    msgInput.value = "";
    if (fileInput) fileInput.value = "";
    renderThumbnails(); // hides the preview bar

    chatEl.scrollTop = chatEl.scrollHeight;
    typingEl.style.display = "flex";
    chatEl.scrollTop = chatEl.scrollHeight;

    // 1. Create chat document in Firestore if new session
    if (!currentChatId) {
      try {
        const initialTitle = userMsg 
          ? (userMsg.substring(0, 30) + (userMsg.length > 30 ? "..." : "")) 
          : (userProfile?.language === "sinhala" ? "රූපමය කතාබහ" : "Image Chat");
        
        const newChatRef = await addDoc(collection(db, "users", currentUserId, "chats"), {
          title: initialTitle,
          createdAt: serverTimestamp(),
          updatedAt: serverTimestamp()
        });
        currentChatId = newChatRef.id;

        const titleDisplay = document.getElementById("chat-title-display");
        if (titleDisplay) {
          titleDisplay.innerText = initialTitle;
        }
      } catch (cErr) {
        console.error("Error creating new chat session document:", cErr);
      }
    } else {
      try {
        await updateDoc(doc(db, "users", currentUserId, "chats", currentChatId), {
          updatedAt: serverTimestamp()
        });
      } catch (uErr) {
        console.warn("Failed to update chat timestamp:", uErr);
      }
    }

    // 2. Write user message to Firestore
    try {
      await addDoc(collection(db, "users", currentUserId, "chats", currentChatId, "messages"), {
        sender: "user",
        text: userMsg || "",
        images: activeImages.length > 0 ? activeImages : null,
        timestamp: serverTimestamp()
      });
    } catch (fsErr) {
      console.error("Error logging user message to Firestore:", fsErr);
    }

    // Build multi-part payload: text string + image inlineData parts
    const parts = [];
    if (userMsg) parts.push(userMsg);
    for (const b64 of activeImages) {
      const mimeType = b64.match(/data:(.*?);base64/)?.[1] || "image/jpeg";
      const data = b64.split(",")[1] || b64;
      parts.push({ inlineData: { data, mimeType } });
    }
    const payload = parts.length === 1 && typeof parts[0] === "string" ? parts[0] : parts;

    let streamSuccess = false;
    let fullReplyText = "";

    try {
      const responseStream = await currentChatSession.sendMessageStream({ message: payload });
      typingEl.style.display = "none";

      const aiBubble = document.createElement("div");
      aiBubble.className = "chat-bubble ai chat-bubble-markdown";
      const msgBody = document.createElement("div");
      msgBody.className = "msg-body";
      aiBubble.appendChild(msgBody);
      chatEl.appendChild(aiBubble);
      chatEl.scrollTop = chatEl.scrollHeight;

      for await (const chunk of responseStream) {
        fullReplyText += chunk.text;
        msgBody.innerHTML = formatMarkdown(fullReplyText);
        chatEl.scrollTop = chatEl.scrollHeight;
      }

      const isSi = userProfile?.language === "sinhala";
      const copyLabel = isSi ? "කොපි කරන්න" : "Copy";
      const actionsDiv = document.createElement("div");
      actionsDiv.className = "chat-bubble-actions";
      actionsDiv.innerHTML = `<button type="button" class="chat-copy-btn" onclick="copyChatMessage(this)">📋 ${copyLabel}</button>`;
      aiBubble.appendChild(actionsDiv);

      streamSuccess = true;
    } catch (err) {
      console.error("Primary chat stream failed, executing fallback...", err);
      
      let history = [];
      try {
        history = await currentChatSession.getHistory();
      } catch (hErr) {
        console.warn("Failed to get chat history, proceeding with empty history:", hErr);
      }

      const isSi = userProfile?.language === "sinhala";
      const sysInstr = getSystemInstruction(isSi, currentAge, currentHeight, currentWeight, currentWaist, currentChest, currentBmi, currentChatLogs);
      const config = { systemInstruction: sysInstr };

      for (let k = 0; k < KEYS.length; k++) {
        const keyIndex = (activeKeyIndex + k) % KEYS.length;
        const currentKey = KEYS[keyIndex];
        const client = new GoogleGenAI({ apiKey: currentKey });

        const startModelIndex = (keyIndex === activeKeyIndex)
          ? MODEL_CHAIN.indexOf(currentChatSession.modelName || "gemini-3.5-flash") + 1
          : 0;

        for (let i = startModelIndex; i < MODEL_CHAIN.length; i++) {
          const nextModel = MODEL_CHAIN[i];
          try {
            console.warn(`[AuraFit AI Stream Fallback] Trying Key Index ${keyIndex} with Model ${nextModel}...`);
            const newSession = client.chats.create({
              model: nextModel,
              history: history,
              config
            });
            newSession.modelName = nextModel;
            newSession.clientInstance = client;

            const resStream = await newSession.sendMessageStream({ message: payload });
            currentChatSession = newSession;
            activeKeyIndex = keyIndex;

            typingEl.style.display = "none";

            const aiBubble = document.createElement("div");
            aiBubble.className = "chat-bubble ai chat-bubble-markdown";
            const msgBody = document.createElement("div");
            msgBody.className = "msg-body";
            aiBubble.appendChild(msgBody);
            chatEl.appendChild(aiBubble);
            chatEl.scrollTop = chatEl.scrollHeight;

            fullReplyText = "";
            for await (const chunk of resStream) {
              fullReplyText += chunk.text;
              msgBody.innerHTML = formatMarkdown(fullReplyText);
              chatEl.scrollTop = chatEl.scrollHeight;
            }

            const copyLabel = isSi ? "කොපි කරන්න" : "Copy";
            const actionsDiv = document.createElement("div");
            actionsDiv.className = "chat-bubble-actions";
            actionsDiv.innerHTML = `<button type="button" class="chat-copy-btn" onclick="copyChatMessage(this)">📋 ${copyLabel}</button>`;
            aiBubble.appendChild(actionsDiv);

            streamSuccess = true;
            break;
          } catch (fallbackErr) {
            console.error(`[AuraFit AI Stream Fallback] Failed for Key Index ${keyIndex} and Model ${nextModel}:`, fallbackErr);
          }
        }
        if (streamSuccess) break;
      }
    }

    if (streamSuccess) {
      // 3. Write AI response to Firestore
      try {
        await addDoc(collection(db, "users", currentUserId, "chats", currentChatId, "messages"), {
          sender: "model",
          text: fullReplyText,
          timestamp: serverTimestamp()
        });
      } catch (fsErr) {
        console.error("Error writing AI response to Firestore:", fsErr);
      }
    } else {
      typingEl.style.display = "none";
      chatEl.innerHTML += `<div class="chat-bubble ai" style="color:var(--accent-rose);">⚠️ ${userProfile?.language==="sinhala"?"ප්‍රතිචාරය ලබා ගැනීමේ දෝෂයක් ඇතිවිය. නැවත උත්සාහ කරන්න.":"Response failed. Please try again."}</div>`;
      chatEl.scrollTop = chatEl.scrollHeight;
    }
    chatEl.scrollTop = chatEl.scrollHeight;
  });
}

// Plan review
const btnReviewPlan = document.getElementById("btn-review-plan");
if (btnReviewPlan) {
  btnReviewPlan.addEventListener("click", async () => {
    const planText   = document.getElementById("plan-text-input").value.trim();
    const resultsBox = document.getElementById("plan-review-results");
    if (!planText) { alert("Please paste your plan first."); return; }

    resultsBox.innerHTML = "Analyzing plan…";
    resultsBox.classList.add("loading");

    try {
      const daily  = await fetchLatestDaily(currentUserId);
      const height = await fetchLatestHeight(currentUserId);
      const isSi   = userProfile?.language === "sinhala";
      const age    = userProfile?.age ?? "--";

      const w  = daily?.weight  ?? "--";
      const wa = daily?.waist   ?? "--";
      const ch = daily?.chest   ?? "--";
      const h  = height?.height ?? "--";

      const activeTab    = window.getActivePlanTab ? window.getActivePlanTab() : "meal";
      const planTypeLabel = activeTab === "meal" ? "Meal Plan" : "Workout Plan";

      const prompt = `User: Age:${age}, Height:${h}cm, Weight:${w}kg, Waist:${wa}in, Chest:${ch}in.
Their ${planTypeLabel}:
---
${planText}
---
Does this suit their profile? Critique and give 3-4 specific improvement suggestions.
Respond 100% in ${isSi ? "Sinhala (සිංහල)" : "English"}.`;

      const resultText = await aiGenerate(prompt);
      typeWriterEffect(resultsBox, resultText, isSi, () => {
        const copyLabel = isSi ? "කොපි කරන්න" : "Copy";
        const actionsDiv = document.createElement("div");
        actionsDiv.className = "chat-bubble-actions";
        actionsDiv.style.marginTop = "1rem";
        actionsDiv.innerHTML = `<button type="button" class="chat-copy-btn" onclick="copyChatMessage(this)">📋 ${copyLabel}</button>`;
        resultsBox.appendChild(actionsDiv);
      });
    } catch (err) {
      console.error("Plan review error:", err);
      resultsBox.classList.remove("loading");
      resultsBox.innerHTML = `<span style="color:var(--accent-rose);">⚠️ Review failed: ${err.message}</span>`;
    }
  });
}

// =========================================================================
// NUTRITION CALCULATOR
// =========================================================================
async function loadNutritionPageData() {
  if (!currentUserId || !userProfile) return;
  const warnEl  = document.getElementById("calculator-warning");
  const calcEl  = document.getElementById("calculator-container");

  try {
    const daily  = await fetchLatestDaily(currentUserId);
    const height = await fetchLatestHeight(currentUserId);

    if (!daily || !height) {
      if (warnEl) warnEl.style.display = "block";
      if (calcEl) calcEl.style.display = "none";
      return;
    }

    if (warnEl) warnEl.style.display = "none";
    if (calcEl) calcEl.style.display = "grid";

    currentWeight = daily.weight;
    currentHeight = height.height;
    currentAge    = userProfile.age;

    document.getElementById("calc-pill-weight").innerText = currentWeight;
    document.getElementById("calc-pill-height").innerText = currentHeight;
    document.getElementById("calc-pill-age").innerText    = currentAge;

    // Check if daily cache exists for today (resets after 12:00 AM midnight)
    const now = new Date();
    const todayDateKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    const cacheKey = `aura_daily_nutrition_${currentUserId}`;
    let loadedFromCache = false;

    try {
      const cachedStr = localStorage.getItem(cacheKey);
      if (cachedStr) {
        const cached = JSON.parse(cachedStr);
        if (cached && cached.date === todayDateKey && cached.activityName && cached.goal) {
          // Restore selected active option cards UI
          if (window.setSelectedOptions) {
            window.setSelectedOptions(cached.activityName, cached.activityFactor, cached.goal);
          }
          // Calculate macros with cached options
          calculateNutrition(null, cached.activityFactor, cached.goal, true);
          // Render cached advice directly without API call!
          const box = document.getElementById("ai-nutrition-advice");
          if (box && cached.resultText) {
            const isSi = userProfile?.language === "sinhala";
            const copyLabel = isSi ? "කොපි කරන්න" : "Copy";
            box.classList.remove("loading");
            box.innerHTML = `
              <div class="msg-body">${formatMarkdown(cached.resultText)}</div>
              <div class="chat-bubble-actions" style="margin-top: 1rem;">
                <button type="button" class="chat-copy-btn" onclick="copyChatMessage(this)">📋 ${copyLabel}</button>
              </div>
            `;
          }
          loadedFromCache = true;
        }
      }
    } catch (e) {
      console.warn("Failed to load daily nutrition cache:", e);
    }

    if (!loadedFromCache) {
      // Unselected default state: clear active option cards and prompt selection
      if (window.resetSelectedOptions) window.resetSelectedOptions();
      calculateNutrition(null, null, null);
    }
  } catch (err) { console.error("Nutrition page error:", err); }
}

function calculateNutrition(_ignored, activityFactor, goal, skipAiCall = false) {
  const box = document.getElementById("ai-nutrition-advice");

  if (!activityFactor || !goal) {
    document.getElementById("val-tdee").innerHTML    = `-- <span>kcal</span>`;
    document.getElementById("val-bmr-lbl").innerText = `BMR: -- kcal`;
    document.getElementById("val-protein").innerHTML  = `-- <span>g</span>`;
    document.getElementById("val-protein-lbl").innerText = `Based on goal`;

    const bt = document.getElementById("bar-tdee");
    const bp = document.getElementById("bar-protein");
    if (bt) bt.style.width = `0%`;
    if (bp) bp.style.width = `0%`;

    if (box) {
      const isSi = userProfile?.language === "sinhala";
      box.classList.remove("loading");
      box.innerHTML = isSi
        ? `<div style="padding:1rem;color:var(--accent-cyan);font-weight:500;">👉 කරුණාකර පෝෂණ උපදෙස් ලබා ගැනීමට ඔබේ ක්‍රියාශීලී මට්ටම (Activity Level) සහ ඉලක්කය (Fitness Goal) තෝරන්න.</div>`
        : `<div style="padding:1rem;color:var(--accent-cyan);font-weight:500;">👉 Please select both your Activity Level and Fitness Goal to calculate macros and generate AI advice.</div>`;
    }
    return;
  }

  if (!currentWeight || !currentHeight || !currentAge || !userProfile) return;

  const gender = userProfile.gender || "male";
  let bmr = gender === "male"
    ? 88.362 + (13.397 * currentWeight) + (4.799 * currentHeight) - (5.677 * currentAge)
    : 447.593 + (9.247 * currentWeight) + (3.098 * currentHeight) - (4.330 * currentAge);

  let tdee = bmr * activityFactor;
  let cals = goal === "loss" ? tdee - 500 : goal === "gain" ? tdee + 300 : tdee;
  if (cals < 1200) cals = 1200;

  const pFactor = goal === "loss" ? 1.8 : goal === "gain" ? 2.2 : 1.6;
  const protein = currentWeight * pFactor;

  document.getElementById("val-tdee").innerHTML    = `${cals.toFixed(0)} <span>kcal</span>`;
  document.getElementById("val-bmr-lbl").innerText = `BMR: ${bmr.toFixed(0)} kcal  |  Maintenance: ${tdee.toFixed(0)} kcal`;
  document.getElementById("val-protein").innerHTML  = `${protein.toFixed(0)} <span>g</span>`;
  document.getElementById("val-protein-lbl").innerText = `${pFactor.toFixed(1)} g/kg body weight`;

  const bt = document.getElementById("bar-tdee");
  const bp = document.getElementById("bar-protein");
  if (bt) bt.style.width = `${Math.min((cals/3500)*100, 100)}%`;
  if (bp) bp.style.width = `${Math.min((protein/200)*100, 100)}%`;

  if (!skipAiCall) {
    const actName = window.getSelectedActivityName ? window.getSelectedActivityName() : "sedentary";
    triggerNutritionAiAdvice(currentWeight, currentHeight, currentAge, bmr, cals, protein, gender, actName, goal, activityFactor);
  }
}
window.calculateNutrition = calculateNutrition;

function triggerNutritionAiAdvice(weight, height, age, bmr, tdee, protein, gender, activity, goal, activityFactor) {
  const box = document.getElementById("ai-nutrition-advice");
  if (!box) return;
  const isSi = userProfile?.language === "sinhala";

  // Check if cache exists for today for this exact activity + goal
  const now = new Date();
  const todayDateKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  const cacheKey = `aura_daily_nutrition_${currentUserId}`;

  try {
    const cachedStr = localStorage.getItem(cacheKey);
    if (cachedStr) {
      const cached = JSON.parse(cachedStr);
      if (cached && cached.date === todayDateKey && cached.activityName === activity && cached.goal === goal && cached.resultText) {
        const copyLabel = isSi ? "කොපි කරන්න" : "Copy";
        box.classList.remove("loading");
        box.innerHTML = `
          <div class="msg-body">${formatMarkdown(cached.resultText)}</div>
          <div class="chat-bubble-actions" style="margin-top: 1rem;">
            <button type="button" class="chat-copy-btn" onclick="copyChatMessage(this)">📋 ${copyLabel}</button>
          </div>
        `;
        return; // Render cached advice instantly without calling API!
      }
    }
  } catch (e) {}

  box.innerHTML = isSi ? "⏳ පෝෂණ උපදෙස් සකසමින්..." : "⏳ Generating AI Nutrition advice…";
  box.classList.add("loading");
  if (aiNutritionAdviceTimeout) clearTimeout(aiNutritionAdviceTimeout);

  aiNutritionAdviceTimeout = setTimeout(async () => {
    try {
      const bmi = (weight / ((height / 100) ** 2)).toFixed(1);
      const waistInfo = (currentWaist && currentWaist !== "--") ? `| Waist: ${currentWaist} inches` : "";
      const chestInfo = (currentChest && currentChest !== "--") ? `| Chest: ${currentChest} inches` : "";

      const prompt = `You are a world-class sports nutritionist and physical health analyst.
Analyze ALL of the user's physical parameters thoroughly before formulating personalized advice:
- Age: ${age} years | Gender: ${gender}
- Height: ${height} cm | Weight: ${weight} kg | Body Mass Index (BMI): ${bmi} ${waistInfo} ${chestInfo}
- Activity Level: ${activity} | Fitness & Health Goal: ${goal}
- Basal Metabolic Rate (BMR): ${bmr.toFixed(0)} kcal
- Target Daily Caloric Intake: ${tdee.toFixed(0)} kcal
- Target Daily Protein Intake: ${protein.toFixed(0)} g (${(protein / weight).toFixed(1)} g/kg)

Required Output Structure:
1. 📊 **Physical Metric Analysis**: Analyze their BMI (${bmi}), weight/height ratio, age, and caloric balance (${bmr.toFixed(0)} BMR vs ${tdee.toFixed(0)} TDEE target).
2. 🎯 **Custom Strategy**: Specific macro guidance & health tips tailored for their profile and goal (${goal}).
3. 🥗 **Daily Meal Plan**: Detailed breakdown (Breakfast, Lunch, Evening Snack, Dinner) engineered to hit ${tdee.toFixed(0)} kcal & ${protein.toFixed(0)}g protein.

Respond EXCLUSIVELY in ${isSi ? "Sinhala (සිංහල)" : "English"}.
Use clean Markdown formatting with clear headers and emojis.`;

      const resultText = await aiGenerate(prompt);

      // Save advice to local storage for today until 12:00 AM midnight
      try {
        localStorage.setItem(cacheKey, JSON.stringify({
          date: todayDateKey,
          activityName: activity,
          activityFactor: activityFactor || (window.getSelectedActivityFactor ? window.getSelectedActivityFactor() : 1.2),
          goal: goal,
          resultText: resultText,
          isSi: isSi
        }));
      } catch (e) {}

      // Animate live typing effect as if AI is typing the message in real time
      typeWriterEffect(box, resultText, isSi, () => {
        const copyLabel = isSi ? "කොපි කරන්න" : "Copy";
        const actionsDiv = document.createElement("div");
        actionsDiv.className = "chat-bubble-actions";
        actionsDiv.style.marginTop = "1rem";
        actionsDiv.innerHTML = `<button type="button" class="chat-copy-btn" onclick="copyChatMessage(this)">📋 ${copyLabel}</button>`;
        box.appendChild(actionsDiv);
      });
    } catch (err) {
      console.error("Nutrition AI error:", err);
      box.classList.remove("loading");
      box.innerHTML = `<span style="color:var(--accent-rose);">⚠️ ${isSi ? "AI උපදෙස් ලබා ගැනීම අසාර්ථක විය. API key / ජාල සම්බන්ධතාවය පරීක්ෂා කරන්න." : "Failed to load AI advice. Check your API key and network connection."}</span>`;
    }
  }, 1000);
}

// =========================================================================
// PROFILE PAGE
// =========================================================================
async function loadProfilePageData() {
  if (!currentUserId || !userProfile) return;

  const el = (id) => document.getElementById(id);
  el("profile-display-name").innerText     = userProfile.name;
  el("profile-display-email").innerText    = userProfile.email;
  el("profile-display-dob").innerText      = userProfile.dob;
  el("profile-display-age").innerText      = `${userProfile.age} Years`;
  el("profile-display-gender").innerText   = userProfile.gender;
  el("profile-display-language").innerText = userProfile.language;

  let joined = "Unknown";
  if (userProfile.createdAt) {
    const d = userProfile.createdAt.seconds
      ? new Date(userProfile.createdAt.seconds * 1000)
      : new Date(userProfile.createdAt);
    joined = d.toLocaleDateString("en-US", { year:"numeric", month:"long", day:"numeric" });
  }
  el("profile-display-joined").innerText = joined;

  el("update-name").value     = userProfile.name;
  el("update-dob").value      = userProfile.dob;
  el("update-gender").value   = userProfile.gender;
  el("update-language").value = userProfile.language;
}

const formProfile = document.getElementById("form-update-profile");
if (formProfile) {
  formProfile.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!currentUserId || !userProfile) return;

    const name     = document.getElementById("update-name").value.trim();
    const dob      = document.getElementById("update-dob").value;
    const gender   = document.getElementById("update-gender").value;
    const language = document.getElementById("update-language").value;
    const age      = calculateAge(dob);

    const btn = document.getElementById("btn-update-profile-submit");
    btn.disabled = true; btn.innerText = "Saving…";
    try {
      await setDoc(doc(db, "users", currentUserId), { ...userProfile, name, dob, age, gender, language });

      // Update in-memory state & cache
      Object.assign(userProfile, { name, dob, age, gender, language });
      cacheSet(CACHE_KEYS.PROFILE, userProfile);

      alert("Profile updated successfully!");
      updateUIWithProfile();
      initLangToggle();       // refresh toggle button appearance
      loadProfilePageData();
    } catch (err) {
      alert("Save failed. Try again.");
      console.error(err);
    } finally {
      btn.disabled = false; btn.innerText = "Save Profile Changes";
    }
  });
}
