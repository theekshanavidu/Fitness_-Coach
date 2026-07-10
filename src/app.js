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
  limit
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
  process.env.GEMINI_API_KEY_D
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
async function aiChat(systemInstruction) {
  return await callWithFallback(async (client, modelName) => {
    const config = {};
    if (systemInstruction) config.systemInstruction = systemInstruction;
    const session = client.chats.create({ model: modelName, config });
    // Attach details to session for mid-chat recovery
    session.modelName = modelName;
    session.clientInstance = client;
    return session;
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

    // Welcome bubble
    const chatEl = document.getElementById("chat-messages");
    if (chatEl) {
      const welcome = isSi
        ? `ආයුබෝවන්! මම ඔබේ AuraFit AI සෞඛ්‍ය උපදේශකයායි.\nදත්ත ලැබුණා (වයස:${age}, බර:${w}kg, උස:${h}cm, BMI:${bmi}).\nඔබට ගැළපෙන **Meal Plan** හෝ **Workout Plan** සකසා දිය හැකි. අද කුමක් අවශ්‍ය ද?`
        : `Hello! I am your AuraFit AI coach.\nProfile received — Age:${age}, Weight:${w}kg, Height:${h}cm, BMI:${bmi}.\nHow can I help you today?`;
      chatEl.innerHTML = `<div class="chat-bubble ai">${formatMarkdown(welcome)}</div>`;
    }

    // Build rich context system instruction
    const sysInstr = getSystemInstruction(isSi, age, h, w, wa, ch, bmi, currentChatLogs);

    // Start a NEW chat session each time
    currentChatSession = await aiChat(sysInstr);

    // AI Suggestions Board
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

  board.innerHTML = `<div class="suggestion-item" style="opacity:0.5"><div class="suggestion-icon">⏳</div><div class="suggestion-content"><div class="suggestion-title">${isSi?"යෝජනා ලබා ගනිමින්…":"Loading suggestions…"}</div></div></div>`;

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

    board.innerHTML = finalItems.map(item => {
      const [icon, title, desc] = item.split("|");
      return `
        <div class="suggestion-item">
          <div class="suggestion-icon">${icon.trim()}</div>
          <div class="suggestion-content">
            <div class="suggestion-title">${title.trim()}</div>
            <div class="suggestion-desc">${desc.trim()}</div>
          </div>
        </div>
      `;
    }).join("");

  } catch (err) {
    console.error("Suggestions generation error (loading defaults):", err);
    // Display 3 default suggestions directly
    board.innerHTML = defaults.map(item => {
      const [icon, title, desc] = item.split("|");
      return `
        <div class="suggestion-item">
          <div class="suggestion-icon">${icon.trim()}</div>
          <div class="suggestion-content">
            <div class="suggestion-title">${title.trim()}</div>
            <div class="suggestion-desc">${desc.trim()}</div>
          </div>
        </div>
      `;
    }).join("");
  }
}

// AI Chat — send message
const chatInputForm = document.getElementById("chat-input-form");
if (chatInputForm) {
  chatInputForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!currentChatSession) return;
    const msgInput = document.getElementById("chat-user-message");
    const userMsg  = msgInput.value.trim();
    if (!userMsg) return;

    const chatEl   = document.getElementById("chat-messages");
    const typingEl = document.getElementById("typing-indicator");

    chatEl.innerHTML += `<div class="chat-bubble user">${userMsg}</div>`;
    msgInput.value = "";
    chatEl.scrollTop = chatEl.scrollHeight;
    typingEl.style.display = "flex";
    chatEl.scrollTop = chatEl.scrollHeight;

    try {
      const res  = await currentChatSession.sendMessage({ message: userMsg });
      const text = res.text;
      typingEl.style.display = "none";
      chatEl.innerHTML += `<div class="chat-bubble ai chat-bubble-markdown">${formatMarkdown(text)}</div>`;
    } catch (err) {
      console.error("Chat error:", err);
      
      let success = false;
      let history = [];
      try {
        history = await currentChatSession.getHistory();
      } catch (hErr) {
        console.warn("Failed to get chat history, proceeding with empty history:", hErr);
      }

      const isSi = userProfile?.language === "sinhala";
      const sysInstr = getSystemInstruction(isSi, currentAge, currentHeight, currentWeight, currentWaist, currentChest, currentBmi, currentChatLogs);
      const config = { systemInstruction: sysInstr };

      // Try remaining models on the current working key, then rotate key and try from first model
      for (let k = 0; k < KEYS.length; k++) {
        const keyIndex = (activeKeyIndex + k) % KEYS.length;
        const currentKey = KEYS[keyIndex];
        const client = new GoogleGenAI({ apiKey: currentKey });

        // If it's the current key, start from the next model. Otherwise, start from model index 0
        const startModelIndex = (keyIndex === activeKeyIndex)
          ? MODEL_CHAIN.indexOf(currentChatSession.modelName || "gemini-3.5-flash") + 1
          : 0;

        for (let i = startModelIndex; i < MODEL_CHAIN.length; i++) {
          const nextModel = MODEL_CHAIN[i];
          try {
            console.warn(`[AuraFit AI Chat Fallback] Trying Key Index ${keyIndex} with Model ${nextModel}...`);
            const newSession = client.chats.create({
              model: nextModel,
              history: history,
              config
            });
            newSession.modelName = nextModel;
            newSession.clientInstance = client;

            const res2 = await newSession.sendMessage({ message: userMsg });
            currentChatSession = newSession;
            activeKeyIndex = keyIndex; // update working key index

            typingEl.style.display = "none";
            chatEl.innerHTML += `<div class="chat-bubble ai chat-bubble-markdown">${formatMarkdown(res2.text)}</div>`;
            success = true;
            break;
          } catch (fallbackErr) {
            console.error(`[AuraFit AI Chat Fallback] Failed for Key Index ${keyIndex} and Model ${nextModel}:`, fallbackErr);
          }
        }
        if (success) break;
      }

      if (success) {
        chatEl.scrollTop = chatEl.scrollHeight;
        return;
      }

      typingEl.style.display = "none";
      chatEl.innerHTML += `<div class="chat-bubble ai" style="color:var(--accent-rose);">⚠️ ${userProfile?.language==="sinhala"?"ප්‍රතිචාරය ලබා ගැනීමේ දෝෂයක් ඇතිවිය. නැවත උත්සාහ කරන්න.":"Response failed. Please try again."}</div>`;
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
      resultsBox.classList.remove("loading");
      resultsBox.innerHTML = formatMarkdown(resultText);
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

    const factor = window.getSelectedActivityFactor ? window.getSelectedActivityFactor() : 1.2;
    const goal   = window.getSelectedGoal           ? window.getSelectedGoal()           : "maintain";
    calculateNutrition(null, factor, goal);
  } catch (err) { console.error("Nutrition page error:", err); }
}

function calculateNutrition(_ignored, activityFactor, goal) {
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

  const actName = window.getSelectedActivityName ? window.getSelectedActivityName() : "sedentary";
  triggerNutritionAiAdvice(currentWeight, currentHeight, currentAge, bmr, cals, protein, gender, actName, goal);
}
window.calculateNutrition = calculateNutrition;

function triggerNutritionAiAdvice(weight, height, age, bmr, tdee, protein, gender, activity, goal) {
  const box = document.getElementById("ai-nutrition-advice");
  if (!box) return;
  const isSi = userProfile?.language === "sinhala";
  box.innerHTML = isSi ? "⏳ පෝෂණ උපදෙස් සකසමින්..." : "⏳ Generating AI Nutrition advice…";
  box.classList.add("loading");
  if (aiNutritionAdviceTimeout) clearTimeout(aiNutritionAdviceTimeout);

  aiNutritionAdviceTimeout = setTimeout(async () => {
    try {
      const prompt = `You are a professional sports dietitian.
User profile:
- Age: ${age} | Gender: ${gender} | Height: ${height}cm | Weight: ${weight}kg
- Activity: ${activity} | Goal: ${goal}
- BMR: ${bmr.toFixed(0)} kcal | Daily calorie target: ${tdee.toFixed(0)} kcal | Protein: ${protein.toFixed(0)}g

Provide a personalized daily meal structure (Breakfast, Lunch, Dinner, Snack) with brief macro tips.
Respond EXCLUSIVELY in ${isSi ? "Sinhala (සිංහල)" : "English"}.
Use clean Markdown. Be concise (~150 words).`;

      const resultText = await aiGenerate(prompt);
      box.classList.remove("loading");
      box.innerHTML = formatMarkdown(resultText);
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
