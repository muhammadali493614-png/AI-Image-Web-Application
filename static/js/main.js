/**
 * SafeVision AI - Frontend Controller Script
 * Handles: SPA tab switching, session/login/register/forgot-password flow,
 * drag & drop uploads, dashboard stats, detection history (AJAX),
 * live PPE status polling (incl. recognized person names), recording
 * controls, target rules, profile updates (incl. profile photo upload for
 * face recognition), mobile hamburger navigation, and the Admin panel
 * (role management).
 */

// ==========================================
// 0. CSRF HELPER
// ==========================================
function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.content : "";
}

// Wraps fetch() so every state-changing request automatically carries the
// CSRF token header. GET requests don't need it, but sending it is harmless.
async function apiFetch(url, options = {}) {
    const headers = Object.assign({}, options.headers || {}, { "X-CSRFToken": getCsrfToken() });
    return fetch(url, Object.assign({ credentials: "same-origin" }, options, { headers }));
}

// ==========================================
// 0b. MOBILE SIDEBAR (hamburger menu)
// ==========================================
const MOBILE_BREAKPOINT = 768;
const sidebarEl = document.getElementById("sidebar");
const sidebarOverlayEl = document.getElementById("sidebarOverlay");

window.openSidebar = function () {
    if (!sidebarEl) return;
    sidebarEl.classList.add("open");
    if (sidebarOverlayEl) sidebarOverlayEl.classList.add("visible");
    document.body.style.overflow = "hidden"; // avoid background scroll while menu is open
};

window.closeSidebar = function () {
    if (!sidebarEl) return;
    sidebarEl.classList.remove("open");
    if (sidebarOverlayEl) sidebarOverlayEl.classList.remove("visible");
    document.body.style.overflow = "";
};

window.toggleSidebar = function () {
    if (!sidebarEl) return;
    if (sidebarEl.classList.contains("open")) {
        closeSidebar();
    } else {
        openSidebar();
    }
};

// If the window is resized past the mobile breakpoint, make sure the sidebar
// isn't left stuck in its "open" (translated) state or blocking scroll.
window.addEventListener("resize", () => {
    if (window.innerWidth > MOBILE_BREAKPOINT) {
        closeSidebar();
    }
});

// ==========================================
// 1. TAB SWITCHING
// ==========================================
// Shared by the initial page-load logic (below) and every nav click, so
// there's exactly one place that decides which .page is visible and which
// nav link is highlighted.
function activatePage(pageId, navLinkEl) {
    document.querySelectorAll(".page").forEach(p => (p.style.display = "none"));
    document.querySelectorAll(".sidebar nav a").forEach(l => l.classList.remove("active"));

    const targetPage = document.getElementById(pageId);
    if (targetPage) targetPage.style.display = "block";

    const link = navLinkEl || document.querySelector(`.sidebar nav a[data-page="${pageId}"]`);
    if (link) link.classList.add("active");

    return targetPage;
}

window.show = function (pageId, element) {
    if (window.event) window.event.preventDefault();

    activatePage(pageId, element || (window.event && window.event.currentTarget));

    // Load fresh data whenever a specific tab is opened
    if (pageId === "dashboard") loadDashboardStats();
    if (pageId === "history") loadDetectionHistory();
    if (pageId === "admin") loadAdminUsers();

    // On mobile, picking a nav item should close the slide-in menu so the
    // user immediately sees the page they tapped through to.
    if (window.innerWidth <= MOBILE_BREAKPOINT) closeSidebar();
};

// ==========================================
// 2. SESSION / LOGIN / REGISTER / LOGOUT / FORGOT PASSWORD
// ==========================================
const loginModal = document.getElementById("loginModal");
const loginForm = document.getElementById("loginForm");
const loginError = document.getElementById("loginError");
const headerUserName = document.getElementById("header-user-name");
const headerUserRole = document.getElementById("header-user-role");
const headerUserAvatar = document.getElementById("header-user-avatar");

const registerForm = document.getElementById("registerForm");
const registerError = document.getElementById("registerError");
const registerSuccess = document.getElementById("registerSuccess");
const showRegisterText = document.getElementById("showRegisterText");
const showLoginText = document.getElementById("showLoginText");

const headerGuestActions = document.getElementById("header-guest-actions");
const headerUserActions = document.getElementById("header-user-actions");

const forgotPasswordModal = document.getElementById("forgotPasswordModal");
const forgotPasswordForm = document.getElementById("forgotPasswordForm");
const forgotPasswordMessage = document.getElementById("forgotPasswordMessage");

const adminNavLink = document.getElementById("admin-nav-link");
const targetsAdminNote = document.getElementById("targets-admin-note");
const saveTargetsBtn = document.getElementById("save-targets-btn");

// Small helper: shows a spinner + disables a submit button while a request
// is in flight, then restores its original label afterwards.
function setButtonBusy(button, busy, busyLabel) {
    if (!button) return;
    if (busy) {
        button.dataset.originalLabel = button.dataset.originalLabel || button.innerHTML;
        button.disabled = true;
        button.innerHTML = `<span class="btn-spinner"></span>${busyLabel || "Please wait..."}`;
    } else {
        button.disabled = false;
        if (button.dataset.originalLabel) button.innerHTML = button.dataset.originalLabel;
    }
}

function hideLoginModal() {
    if (loginModal) loginModal.style.display = "none";
}

function openAuthModal(mode) {
    if (!loginModal) return;

    const wantRegister = mode === "register";
    if (loginForm) loginForm.style.display = wantRegister ? "none" : "block";
    if (registerForm) registerForm.style.display = wantRegister ? "block" : "none";
    if (showRegisterText) showRegisterText.style.display = wantRegister ? "none" : "block";
    if (showLoginText) showLoginText.style.display = wantRegister ? "block" : "none";

    if (loginError) loginError.style.display = "none";
    if (registerError) registerError.style.display = "none";
    if (registerSuccess) registerSuccess.style.display = "none";

    loginModal.style.display = "flex";
}

window.showLoginModal = function () {
    openAuthModal("login");
};

window.showSignupModal = function () {
    openAuthModal("register");
};

window.toggleAuthMode = function (e) {
    if (e) e.preventDefault();
    const showingLogin = loginForm.style.display !== "none";
    openAuthModal(showingLogin ? "register" : "login");
};

// --- Forgot password modal ---
window.showForgotPasswordModal = function (e) {
    if (e) e.preventDefault();
    if (!forgotPasswordModal) return;
    if (loginModal) loginModal.style.display = "none";
    if (forgotPasswordMessage) forgotPasswordMessage.style.display = "none";
    if (forgotPasswordForm) forgotPasswordForm.reset();
    forgotPasswordModal.style.display = "flex";
};

window.hideForgotPasswordModal = function (e) {
    if (e) e.preventDefault();
    if (forgotPasswordModal) forgotPasswordModal.style.display = "none";
    if (loginModal) loginModal.style.display = "flex";
};

if (forgotPasswordForm) {
    forgotPasswordForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const email = document.getElementById("forgotEmail").value.trim();
        const submitBtn = forgotPasswordForm.querySelector('button[type="submit"]');

        setButtonBusy(submitBtn, true, "Sending...");
        try {
            const res = await apiFetch("/forgot_password", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email }),
            });
            const data = await res.json();

            if (forgotPasswordMessage) {
                forgotPasswordMessage.style.color = res.ok ? "#22c55e" : "#ef4444";
                forgotPasswordMessage.innerText = data.message || "Something went wrong.";
                forgotPasswordMessage.style.display = "block";
            }
            if (res.ok) forgotPasswordForm.reset();
        } catch (err) {
            console.error("Forgot password request failed:", err);
            if (forgotPasswordMessage) {
                forgotPasswordMessage.style.color = "#ef4444";
                forgotPasswordMessage.innerText = "Unable to reach server. Please try again.";
                forgotPasswordMessage.style.display = "block";
            }
        } finally {
            setButtonBusy(submitBtn, false);
        }
    });
}

function updateHeaderAuthState(loggedIn) {
    if (headerUserActions) headerUserActions.style.display = loggedIn ? "flex" : "none";
}

// Shows/hides the small round avatar next to "Welcome, <name>" in the
// header. photoUrl is the server-relative /profile_photos/... path (or
// null/undefined if the user hasn't uploaded one yet).
function applyHeaderAvatar(photoUrl) {
    if (!headerUserAvatar) return;
    if (photoUrl) {
        headerUserAvatar.src = photoUrl;
        headerUserAvatar.style.display = "inline-block";
    } else {
        headerUserAvatar.removeAttribute("src");
        headerUserAvatar.style.display = "none";
    }
}

function applyAdminUI(isAdmin) {
    if (adminNavLink) adminNavLink.style.display = isAdmin ? "block" : "none";

    document.querySelectorAll("#targetsForm .toggle-switch").forEach(cb => (cb.disabled = !isAdmin));
    if (saveTargetsBtn) {
        saveTargetsBtn.disabled = !isAdmin;
        saveTargetsBtn.style.opacity = isAdmin ? "1" : ".5";
        saveTargetsBtn.style.cursor = isAdmin ? "pointer" : "not-allowed";
    }
    if (targetsAdminNote) targetsAdminNote.style.display = isAdmin ? "none" : "block";
}

async function checkSession() {
    try {
        const res = await fetch("/api/current_user", { credentials: "same-origin" });
        const data = await res.json();

        if (data.logged_in) {
            hideLoginModal();
            updateHeaderAuthState(true);
            if (headerUserName) headerUserName.innerText = data.full_name || data.username;
            if (headerUserRole) headerUserRole.innerText = data.role ? `(${data.role})` : "";
            applyHeaderAvatar(data.profile_photo);
            const profName = document.getElementById("prof-name");
            const profEmail = document.getElementById("prof-email");
            if (profName) profName.value = data.full_name || "";
            if (profEmail) profEmail.value = data.email || "";
            applyProfilePhotoPreview(data.profile_photo);
            applyAdminUI(!!data.is_admin);
            loadDashboardStats();
        } else {
            // Not logged in — just show the header's Login/Sign Up buttons.
            // We deliberately do NOT auto-open the login modal on every plain
            // page load; it only opens when the person clicks Login/Sign Up,
            // OR when the server redirected them back here after they tried
            // an action that requires login (see __AUTH_REQUIRED__ below).
            updateHeaderAuthState(false);
            applyAdminUI(false);

            if (window.__AUTH_REQUIRED__) {
                window.__AUTH_REQUIRED__ = false; // only trigger once per load
                showLoginModal();
                if (loginError) {
                    loginError.innerText = "Please log in to do that.";
                    loginError.style.display = "block";
                }
            }
        }
    } catch (err) {
        console.error("Session check failed:", err);
        updateHeaderAuthState(false);
    }
}

if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const username = document.getElementById("loginUser").value.trim();
        const password = document.getElementById("loginPass").value;
        const submitBtn = loginForm.querySelector('button[type="submit"]');

        if (loginError) loginError.style.display = "none";
        setButtonBusy(submitBtn, true, "Signing in...");

        try {
            const res = await apiFetch("/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password }),
            });
            const data = await res.json();

            if (res.ok && data.status === "success") {
                hideLoginModal();
                updateHeaderAuthState(true);
                if (headerUserName) headerUserName.innerText = data.full_name || username;
                if (headerUserRole) headerUserRole.innerText = data.role ? `(${data.role})` : "";
                applyHeaderAvatar(data.profile_photo);
                applyProfilePhotoPreview(data.profile_photo);
                applyAdminUI(!!data.is_admin);
                loginForm.reset();
                loadDashboardStats();
            } else {
                if (loginError) {
                    loginError.innerText = data.message || "Invalid username or password.";
                    loginError.style.display = "block";
                }
            }
        } catch (err) {
            console.error("Login request failed:", err);
            if (loginError) {
                loginError.innerText = "Unable to reach server. Please try again.";
                loginError.style.display = "block";
            }
        } finally {
            setButtonBusy(submitBtn, false);
        }
    });
}

if (registerForm) {
    registerForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (registerError) registerError.style.display = "none";
        if (registerSuccess) registerSuccess.style.display = "none";
        const submitBtn = registerForm.querySelector('button[type="submit"]');

        const payload = {
            full_name: document.getElementById("regFullName").value.trim(),
            username: document.getElementById("regUsername").value.trim(),
            email: document.getElementById("regEmail").value.trim(),
            password: document.getElementById("regPassword").value,
        };

        setButtonBusy(submitBtn, true, "Creating account...");
        try {
            const res = await apiFetch("/register", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await res.json();

            if (res.ok && data.status === "success") {
                if (registerSuccess) {
                    registerSuccess.innerText = data.message + " After logging in, visit your Profile tab to upload a photo so you're recognized by name on the Live Camera Stream.";
                    registerSuccess.style.display = "block";
                }
                registerForm.reset();
                setTimeout(() => {
                    toggleAuthMode();
                    const loginUserField = document.getElementById("loginUser");
                    if (loginUserField) loginUserField.value = payload.username;
                }, 1200);
            } else {
                if (registerError) {
                    registerError.innerText = data.message || "Registration failed.";
                    registerError.style.display = "block";
                }
            }
        } catch (err) {
            console.error("Registration request failed:", err);
            if (registerError) {
                registerError.innerText = "Unable to reach server. Please try again.";
                registerError.style.display = "block";
            }
        } finally {
            setButtonBusy(submitBtn, false);
        }
    });
}

window.logoutUser = async function () {
    try {
        await apiFetch("/logout", { method: "POST" });
    } catch (err) {
        console.error("Logout request failed:", err);
    } finally {
        updateHeaderAuthState(false);
        applyAdminUI(false);
        applyHeaderAvatar(null);
        showLoginModal();
        if (headerUserName) headerUserName.innerText = "Guest";
        if (headerUserRole) headerUserRole.innerText = "";
    }
};

// ==========================================
// 3. DASHBOARD STATS
// ==========================================
async function loadDashboardStats() {
    try {
        const res = await fetch("/api/dashboard_stats", { credentials: "same-origin" });
        if (!res.ok) return;
        const data = await res.json();

        setText("total-scanned", data.total_scanned);
        setText("compliant-workers", data.compliant_workers);
        setText("safety-violations", data.safety_violations);
        // model_accuracy is a static validation metric (mAP50), not a live
        // number — it shows "N/A" until utils/model_validation.py has been
        // run at least once to generate model_accuracy.json.
        setText("model-accuracy", data.model_accuracy === "N/A" ? "N/A" : `${data.model_accuracy}%`);
    } catch (err) {
        console.error("Dashboard stats fetch failed:", err);
    }

    loadAnalytics();
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerText = value;
}

// ==========================================
// 3a. ANALYTICS CHARTS (Dashboard — Chart.js)
// ==========================================
let trendChartInstance = null;
let complianceChartInstance = null;
let sourceChartInstance = null;

async function loadAnalytics() {
    if (typeof Chart === "undefined") return;

    try {
        const res = await fetch("/api/analytics", { credentials: "same-origin" });
        if (!res.ok) return;
        const data = await res.json();

        renderTrendChart(data.trend);
        renderComplianceChart(data.compliance_ratio);
        renderSourceChart(data.source_breakdown);
    } catch (err) {
        console.error("Analytics fetch failed:", err);
    }
}

function renderTrendChart(trend) {
    const canvas = document.getElementById("chart-trend");
    if (!canvas || !Array.isArray(trend)) return;

    const labels = trend.map(t => t.date.slice(5));
    const totals = trend.map(t => t.total);
    const violations = trend.map(t => t.violations);

    if (trendChartInstance) trendChartInstance.destroy();
    trendChartInstance = new Chart(canvas, {
        type: "line",
        data: {
            labels,
            datasets: [
                { label: "Total Scans", data: totals, borderColor: "#0284c7", backgroundColor: "rgba(2,132,199,.15)", tension: .3, fill: true },
                { label: "Violations", data: violations, borderColor: "#ef4444", backgroundColor: "rgba(239,68,68,.15)", tension: .3, fill: true },
            ],
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: "#e5e7eb" } } },
            scales: {
                x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,.05)" } },
                y: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,.05)" }, beginAtZero: true },
            },
        },
    });
}

function renderComplianceChart(ratio) {
    const canvas = document.getElementById("chart-compliance");
    if (!canvas || !ratio) return;

    if (complianceChartInstance) complianceChartInstance.destroy();
    complianceChartInstance = new Chart(canvas, {
        type: "doughnut",
        data: {
            labels: ["Compliant", "Violations"],
            datasets: [{ data: [ratio.compliant, ratio.violations], backgroundColor: ["#22c55e", "#ef4444"], borderWidth: 0 }],
        },
        options: {
            responsive: true,
            plugins: { legend: { position: "bottom", labels: { color: "#e5e7eb" } } },
        },
    });
}

function renderSourceChart(breakdown) {
    const canvas = document.getElementById("chart-source");
    if (!canvas || !breakdown) return;

    const labels = Object.keys(breakdown);
    const values = Object.values(breakdown);

    if (sourceChartInstance) sourceChartInstance.destroy();
    sourceChartInstance = new Chart(canvas, {
        type: "bar",
        data: {
            labels,
            datasets: [{ label: "Detections", data: values, backgroundColor: "#0284c7", borderRadius: 6 }],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,.05)" }, beginAtZero: true },
                y: { ticks: { color: "#9ca3af" }, grid: { display: false } },
            },
        },
    });
}

// ==========================================
// 3b. DETECTION HISTORY (AJAX — no full page reload)
// ==========================================
async function loadDetectionHistory() {
    const detectionBody = document.getElementById("detection-logs-body");
    const loginBody = document.getElementById("login-logs-body");

    try {
        const [detRes, loginRes] = await Promise.all([
            fetch("/api/detection_logs", { credentials: "same-origin" }),
            fetch("/api/login_logs", { credentials: "same-origin" }),
        ]);
        const detections = detRes.ok ? await detRes.json() : [];
        const logins = loginRes.ok ? await loginRes.json() : [];

        if (detectionBody) {
            detectionBody.innerHTML = detections.length
                ? detections.map(log => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,.05)">
                        <td>#${log.id}</td>
                        <td>${log.source_type}</td>
                        <td><span style="color:var(--success,#22c55e)">${log.status}</span></td>
                        <td><a href="${log.file_path}" target="_blank" style="color:var(--accent-blue,#0284c7)">View File</a></td>
                        <td>${log.report_path ? `<a href="${log.report_path}" target="_blank" style="color:#16a34a">📄 Report</a>` : "—"}</td>
                        <td>${log.timestamp}</td>
                    </tr>`).join("")
                : `<tr><td colspan="6" style="padding:15px;text-align:center;color:var(--text-muted,#9ca3af)">No detection logs found in database.</td></tr>`;
        }

        if (loginBody) {
            loginBody.innerHTML = logins.length
                ? logins.map(log => `
                    <tr style="border-bottom:1px solid rgba(255,255,255,.05)">
                        <td>#${log.id}</td>
                        <td>${log.username}</td>
                        <td>${log.ip_address}</td>
                        <td>${log.login_time}</td>
                    </tr>`).join("")
                : `<tr><td colspan="4" style="padding:15px;text-align:center;color:var(--text-muted,#9ca3af)">No login history found.</td></tr>`;
        }
    } catch (err) {
        console.error("Detection history fetch failed:", err);
    }
}

// ==========================================
// 3c. ADMIN PANEL (user list, role toggle, delete)
// ==========================================
async function loadAdminUsers() {
    const body = document.getElementById("admin-users-body");
    if (!body) return;

    try {
        const res = await fetch("/api/admin/users", { credentials: "same-origin" });
        if (!res.ok) {
            body.innerHTML = `<tr><td colspan="6" style="padding:15px;text-align:center;color:var(--text-muted,#9ca3af)">Admin access required.</td></tr>`;
            return;
        }
        const users = await res.json();

        body.innerHTML = users.length
            ? users.map(u => `
                <tr style="border-bottom:1px solid rgba(255,255,255,.05)">
                    <td>#${u.id}</td>
                    <td>${u.full_name}</td>
                    <td>${u.username}</td>
                    <td>${u.email}</td>
                    <td>${u.role}</td>
                    <td style="display:flex;gap:8px;flex-wrap:wrap">
                        <button onclick="toggleUserRole(${u.id}, ${u.is_admin ? "false" : "true"})"
                            style="background:${u.is_admin ? "rgba(245,158,11,.2)" : "rgba(2,132,199,.2)"};color:${u.is_admin ? "#f59e0b" : "#0284c7"};border:1px solid ${u.is_admin ? "#f59e0b" : "#0284c7"};padding:5px 10px;border-radius:5px;cursor:pointer;font-size:.8rem">
                            ${u.is_admin ? "Revoke Admin" : "Make Admin"}
                        </button>
                        <button onclick="deleteUser(${u.id})"
                            style="background:rgba(239,68,68,.2);color:#ef4444;border:1px solid #ef4444;padding:5px 10px;border-radius:5px;cursor:pointer;font-size:.8rem">
                            Delete
                        </button>
                    </td>
                </tr>`).join("")
            : `<tr><td colspan="6" style="padding:15px;text-align:center;color:var(--text-muted,#9ca3af)">No users found.</td></tr>`;
    } catch (err) {
        console.error("Admin user list fetch failed:", err);
        body.innerHTML = `<tr><td colspan="6" style="padding:15px;text-align:center;color:#ef4444">Failed to load users.</td></tr>`;
    }
}

window.toggleUserRole = async function (userId, makeAdmin) {
    try {
        const res = await apiFetch(`/api/admin/users/${userId}/role`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ is_admin: makeAdmin }),
        });
        const data = await res.json();
        if (!res.ok) {
            alert(data.message || "Failed to update role.");
            return;
        }
        loadAdminUsers();
    } catch (err) {
        console.error("Role update failed:", err);
        alert("Could not update role. Please try again.");
    }
};

window.deleteUser = async function (userId) {
    if (!confirm("Delete this user? This cannot be undone.")) return;

    try {
        const res = await apiFetch(`/api/admin/users/${userId}`, { method: "DELETE" });
        const data = await res.json();
        if (!res.ok) {
            alert(data.message || "Failed to delete user.");
            return;
        }
        loadAdminUsers();
    } catch (err) {
        console.error("Delete user failed:", err);
        alert("Could not delete user. Please try again.");
    }
};

// ==========================================
// 4. LIVE PPE STATUS POLLING (Live Camera tab)
// ==========================================
const PPE_ITEMS = ["helmet", "vest", "gloves", "shoes", "glasses"];
let liveStatusInterval = null;

function startLiveStatusPolling() {
    if (liveStatusInterval) return;
    liveStatusInterval = setInterval(fetchLiveStatus, 1500);
    fetchLiveStatus();
}

function stopLiveStatusPolling() {
    if (liveStatusInterval) {
        clearInterval(liveStatusInterval);
        liveStatusInterval = null;
    }
}

async function fetchLiveStatus() {
    try {
        const res = await fetch("/api/live_status", { credentials: "same-origin" });
        if (!res.ok) return;
        const data = await res.json();

        PPE_ITEMS.forEach(item => {
            const el = document.getElementById(`status-${item}`);
            if (el && data[item]) {
                el.innerText = data[item].status;
                el.style.color = data[item].is_missing ? "#ef4444" : "#22c55e";
            }
        });

        // Recognized person(s) — populated server-side from the uploaded
        // profile photo of any registered user detected in the current
        // frame (see /api/live_status -> live_ppe_status["person_names"]
        // in app.py). Empty list just means nobody recognized right now,
        // not necessarily that nobody's on camera.
        const namesEl = document.getElementById("recognized-names");
        if (namesEl) {
            const names = Array.isArray(data.person_names) ? data.person_names : [];
            namesEl.innerText = names.length ? names.join(", ") : "No registered user recognized";
            namesEl.style.color = names.length ? "#22c55e" : "var(--text-muted, #9ca3af)";
        }

        const banner = document.getElementById("safety-alert-banner");
        const overall = document.getElementById("overall-status");

        if (banner) banner.style.display = data.alert ? "block" : "none";
        if (overall) {
            overall.innerText = data.alert_message || "Analyzing Feed...";
            overall.style.color = data.alert ? "#ef4444" : "#22c55e";
        }
    } catch (err) {
        console.error("Live status fetch failed:", err);
    }
}

// ==========================================
// 5. RECORDING CONTROLS
// ==========================================
let isRecording = false;
const recordBtn = document.getElementById("recordToggleBtn");
const recordingStatusEl = document.getElementById("recording-status");
const recordingResultEl = document.getElementById("recording-result");

if (recordBtn) {
    recordBtn.addEventListener("click", async () => {
        recordBtn.disabled = true;
        try {
            if (!isRecording) {
                const res = await apiFetch("/start_recording", { method: "POST" });
                const data = await res.json();
                if (res.ok && data.status === "success") {
                    isRecording = true;
                    recordBtn.innerText = "⏹ Stop Recording";
                    recordBtn.style.backgroundColor = "#ef4444";
                    if (recordingStatusEl) recordingStatusEl.innerText = "🔴 Recording in progress...";
                    if (recordingResultEl) recordingResultEl.style.display = "none";
                } else {
                    alert(data.message || "Could not start recording.");
                }
            } else {
                const res = await apiFetch("/stop_recording", { method: "POST" });
                const data = await res.json();
                if (res.ok && data.status === "success") {
                    isRecording = false;
                    recordBtn.innerText = "⏺ Start Recording";
                    recordBtn.style.backgroundColor = "#0284c7";
                    if (recordingStatusEl) recordingStatusEl.innerText = "";

                    if (recordingResultEl) {
                        recordingResultEl.style.display = "block";
                        recordingResultEl.innerHTML = `
                            <p><strong>✅ Recording saved</strong> — Duration: ${Number(data.duration_seconds).toFixed(1)}s</p>
                            <p>Total Frames: ${data.total_frames} | Violation Frames: ${data.violation_frames}</p>
                            <p>
                                <a href="${data.video_url}" target="_blank" style="color:#0284c7">🎬 View Recording</a> &nbsp;|&nbsp;
                                <a href="${data.report_url}" target="_blank" style="color:#16a34a">📄 Download Session Report</a>
                            </p>`;
                    }
                } else {
                    alert(data.message || "Could not stop recording.");
                }
            }
        } catch (err) {
            console.error("Recording control failed:", err);
            alert("Something went wrong. Please try again.");
        } finally {
            recordBtn.disabled = false;
        }
    });
}

// ==========================================
// 6. TARGET PPE CHECKLIST (Admin only — server also enforces this)
// ==========================================
window.saveTargets = async function () {
    if (saveTargetsBtn && saveTargetsBtn.disabled) return;

    const payload = {
        helmet: document.getElementById("target-helmet")?.checked ?? true,
        vest: document.getElementById("target-vest")?.checked ?? true,
        gloves: document.getElementById("target-gloves")?.checked ?? false,
        shoes: document.getElementById("target-shoes")?.checked ?? true,
        glasses: document.getElementById("target-glasses")?.checked ?? false,
    };

    setButtonBusy(saveTargetsBtn, true, "Saving...");
    try {
        const res = await apiFetch("/api/targets", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        alert(data.message || (res.ok ? "Saved successfully." : "Failed to save."));
    } catch (err) {
        console.error("Save targets failed:", err);
        alert("Could not save target rules. Please try again.");
    } finally {
        setButtonBusy(saveTargetsBtn, false);
    }
};

// ==========================================
// 7. PROFILE UPDATE (name, email, photo -> face registration)
// ==========================================
const profPhotoInput = document.getElementById("prof-photo");
const profPhotoPreview = document.getElementById("profile-photo-preview");
const profPhotoPlaceholder = document.getElementById("profile-photo-placeholder");
const profPhotoMessage = document.getElementById("prof-photo-message");

// Shows a photo (existing server URL, or a freshly chosen local file) in
// the round preview circle on the Profile tab; falls back to the 🧑 emoji
// placeholder when there's no photo yet.
function applyProfilePhotoPreview(photoUrlOrDataUri) {
    if (!profPhotoPreview || !profPhotoPlaceholder) return;
    if (photoUrlOrDataUri) {
        profPhotoPreview.src = photoUrlOrDataUri;
        profPhotoPreview.style.display = "block";
        profPhotoPlaceholder.style.display = "none";
    } else {
        profPhotoPreview.removeAttribute("src");
        profPhotoPreview.style.display = "none";
        profPhotoPlaceholder.style.display = "flex";
    }
}

// Live-preview the chosen file immediately, before it's uploaded.
if (profPhotoInput) {
    profPhotoInput.addEventListener("change", () => {
        const file = profPhotoInput.files && profPhotoInput.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => applyProfilePhotoPreview(reader.result);
        reader.readAsDataURL(file);
    });
}

window.updateProfile = async function (event) {
    event.preventDefault();

    const fullName = document.getElementById("prof-name").value.trim();
    const email = document.getElementById("prof-email").value.trim();
    const photoFile = profPhotoInput && profPhotoInput.files ? profPhotoInput.files[0] : null;
    const submitBtn = event.target.querySelector('button[type="submit"]');

    const formData = new FormData();
    formData.append("full_name", fullName);
    formData.append("email", email);
    if (photoFile) formData.append("photo", photoFile);

    if (profPhotoMessage) profPhotoMessage.style.display = "none";
    setButtonBusy(submitBtn, true, "Saving...");
    try {
        const res = await apiFetch("/api/update_profile", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (res.ok && data.status === "success") {
            if (headerUserName) headerUserName.innerText = data.full_name || fullName;
            if (data.profile_photo) {
                applyHeaderAvatar(data.profile_photo);
                applyProfilePhotoPreview(data.profile_photo);
            }
            if (profPhotoMessage && data.face_message) {
                profPhotoMessage.innerText = data.face_message;
                profPhotoMessage.style.color = data.profile_photo ? "#f59e0b" : "#ef4444";
                profPhotoMessage.style.display = "block";
            }
            if (photoFile) {
                profPhotoInput.value = ""; // clear the file picker after a successful upload
            }
            alert("Profile updated successfully.");
        } else {
            alert(data.message || "Failed to update profile.");
        }
    } catch (err) {
        console.error("Profile update failed:", err);
        alert("Could not update profile. Please try again.");
    } finally {
        setButtonBusy(submitBtn, false);
    }
};

// ==========================================
// 8. INIT ON PAGE LOAD
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
    "use strict";

    // Which tab to open on load is decided server-side (see the inline
    // script in index.html): "upload" right after a /detect submission
    // that produced a result, "dashboard" on every other normal page load.
    // This replaces the old hardcoded "always show #dashboard" behavior
    // that caused the Dashboard to flash over a just-generated result.
    const initialTab = window.__INITIAL_TAB__ === "upload" ? "upload" : "dashboard";
    const initialPage = activatePage(initialTab);

    if (initialTab === "dashboard") {
        loadDashboardStats();
    } else if (initialPage) {
        // Scroll the freshly rendered detection result into view instead of
        // leaving the user at the top of the Upload tab having to scroll down.
        const resultDisplay = initialPage.querySelector(".result-display");
        if (resultDisplay) resultDisplay.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    checkSession();
    startLiveStatusPolling();

    const uploadBox = document.querySelector(".upload-box");
    const fileInput = document.querySelector("#fileInput");

    if (uploadBox && fileInput) {
        ["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
            uploadBox.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ["dragenter", "dragover"].forEach(eventName => {
            uploadBox.addEventListener(eventName, () => uploadBox.classList.add("drag-over"), false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            uploadBox.addEventListener(eventName, () => uploadBox.classList.remove("drag-over"), false);
        });

        uploadBox.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                fileInput.files = files;
                updateUploadBoxText(files[0].name);
            }
        }, false);

        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                updateUploadBoxText(e.target.files[0].name);
            }
        });

        function updateUploadBoxText(fileName) {
            const h3 = uploadBox.querySelector("h3");
            const p = uploadBox.querySelector("p");
            if (h3) h3.innerText = `📄 Selected File: ${fileName}`;
            if (p) {
                p.innerText = "Ready to process! Click 'Run AI Detection' below.";
                p.style.color = "var(--accent-blue, #31d6ff)";
            }
        }
    }
});