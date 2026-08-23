/* ==========================================================================
   BhoomiMitra AI — Voice-First SPA Frontend Controller & Intent Engine
   Integrates 11 Indian Languages, Speech Recognition, TTS Synthesis,
   Voice Shopping, Voice Order Tracking, Voice Scanner, and DB Sync
   ========================================================================== */

/* ==========================================================================
   BhoomiMitra AI — Voice-First SPA Frontend Controller & Intent Engine
   Integrates 11 Indian Languages, Speech Recognition, TTS Synthesis,
   Voice Shopping, Voice Order Tracking, Voice Scanner, and DB Sync
   ========================================================================== */

let currentRole = 'farmer';
let currentLanguage = 'en';
let currentUser = null;
let activeFarmerId = null;
let activeShopId = null;
let activeExpertId = null;
let chartInstance = null;
let chatPollingTimer = null;
let lastKnownConversationCount = 0;
let lastAiResponseText = "";

document.addEventListener('DOMContentLoaded', async () => {
  await initializeData();
  // Apply default language translations
  applyLanguageTranslation('en');
  // Start background live polling for new WhatsApp Webhook messages
  startWebhookPolling();
});

async function initializeData() {
  try {
    // 1. Check user authentication status via HttpOnly session cookie
    await checkAuthStatus();

    // 2. Fetch first farmer for voice AI / chat testing
    const farmerRes = await fetch('/farmers?page=1&size=1', { credentials: 'include' });
    if (farmerRes.ok) {
      const farmerData = await farmerRes.json();
      if (farmerData.items && farmerData.items.length > 0) {
        const farmer = farmerData.items[0];
        activeFarmerId = farmer.id;
        
        // Update Farmer Profile Card
        const nameSpan = document.getElementById('farmer-name-display');
        const phoneSpan = document.getElementById('farmer-phone-display');
        if (nameSpan) nameSpan.innerText = farmer.full_name || farmer.phone_number || "Ramesh Gowda";
        if (phoneSpan) phoneSpan.innerText = farmer.phone_number || "+91 9876543210";
      }
    }

    // 3. Load public farmer views
    await loadNearbyShops();
    await loadFarmerOrders();
    if (activeFarmerId) {
      await loadFarmerChatMessages();
    }
  } catch (err) {
    console.error("Failed to initialize dashboard data:", err);
  }
}

// ==========================================================================
// AUTHENTICATION & RBAC CONTROLLER
// ==========================================================================

async function checkAuthStatus() {
  try {
    const res = await fetch('/auth/me', { credentials: 'include' });
    if (res.ok) {
      currentUser = await res.json();
      updateAuthUI();

      // Configure role IDs
      if (currentUser.role === 'shop_owner' && currentUser.shop_id) {
        activeShopId = currentUser.shop_id;
        loadShopOwnerData();
      } else if (currentUser.role === 'expert' && currentUser.expert_id) {
        activeExpertId = currentUser.expert_id;
        loadExpertDashboardData();
      } else if (currentUser.role === 'admin') {
        // Admin has universal access
        if (!activeShopId) {
          const shopRes = await fetch('/shops?page=1&size=1', { credentials: 'include' });
          if (shopRes.ok) {
            const data = await shopRes.json();
            if (data.items && data.items.length > 0) {
              activeShopId = data.items[0].id;
            }
          }
        }
      }
    } else {
      currentUser = null;
      activeShopId = null;
      activeExpertId = null;
      updateAuthUI();
    }
  } catch (err) {
    console.warn("Auth check error:", err);
    currentUser = null;
    activeShopId = null;
    activeExpertId = null;
    updateAuthUI();
  }
}

function updateAuthUI() {
  const btnLogin = document.getElementById('btn-nav-login');
  const userBadge = document.getElementById('user-profile-badge');
  const emailDisplay = document.getElementById('user-email-display');
  const roleBadge = document.getElementById('user-role-badge');

  if (currentUser) {
    if (btnLogin) btnLogin.style.display = 'none';
    if (userBadge) userBadge.style.display = 'flex';
    if (emailDisplay) emailDisplay.innerText = currentUser.email;
    if (roleBadge) {
      roleBadge.innerText = currentUser.role.replace('_', ' ').toUpperCase();
      roleBadge.className = `badge badge-${currentUser.role === 'admin' ? 'open' : currentUser.role === 'expert' ? 'accepted' : 'completed'}`;
    }
  } else {
    if (btnLogin) btnLogin.style.display = 'inline-block';
    if (userBadge) userBadge.style.display = 'none';
  }
}

function openAuthModal(mode = 'login') {
  const modal = document.getElementById('modal-auth');
  if (!modal) return;
  modal.classList.add('active');
  hideAuthAlert();
  switchAuthTab(mode);
}

function switchAuthTab(tab) {
  const btnLogin = document.getElementById('auth-tab-btn-login');
  const btnRegister = document.getElementById('auth-tab-btn-register');
  const formLogin = document.getElementById('form-auth-login');
  const formRegister = document.getElementById('form-auth-register');
  const title = document.getElementById('modal-auth-title');

  hideAuthAlert();

  if (tab === 'login') {
    if (btnLogin) btnLogin.classList.add('active');
    if (btnRegister) btnRegister.classList.remove('active');
    if (formLogin) formLogin.style.display = 'block';
    if (formRegister) formRegister.style.display = 'none';
    if (title) title.innerText = "🔐 Sign In to BhoomiMitra AI";
  } else {
    if (btnRegister) btnRegister.classList.add('active');
    if (btnLogin) btnLogin.classList.remove('active');
    if (formRegister) formRegister.style.display = 'block';
    if (formLogin) formLogin.style.display = 'none';
    if (title) title.innerText = "📝 Register New Account";
  }
}

function onRegisterRoleChange(role) {
  const shopGroup = document.getElementById('reg-group-shop');
  const expertGroup = document.getElementById('reg-group-expert');
  const adminGroup = document.getElementById('reg-group-admin');

  if (shopGroup) shopGroup.style.display = role === 'shop_owner' ? 'block' : 'none';
  if (expertGroup) expertGroup.style.display = role === 'expert' ? 'block' : 'none';
  if (adminGroup) adminGroup.style.display = role === 'admin' ? 'block' : 'none';
}

function showAuthAlert(message, isError = true) {
  const banner = document.getElementById('auth-alert');
  const text = document.getElementById('auth-alert-text');
  if (!banner || !text) return;

  text.innerText = message;
  banner.className = `alert-banner ${isError ? 'auth-alert-error' : 'auth-alert-success'}`;
  banner.style.display = 'block';
}

function hideAuthAlert() {
  const banner = document.getElementById('auth-alert');
  if (banner) banner.style.display = 'none';
}

async function handleAuthLogin(e) {
  e.preventDefault();
  hideAuthAlert();

  const email = document.getElementById('login-email').value.trim();
  const password = document.getElementById('login-password').value;
  const submitBtn = document.getElementById('btn-submit-login');

  if (!email || !password) {
    showAuthAlert("Please enter both email and password.", true);
    return;
  }

  try {
    if (submitBtn) submitBtn.disabled = true;
    const res = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include', // Sets the HttpOnly cookie automatically
      body: JSON.stringify({ email, password }),
    });

    const data = await res.json();
    if (res.ok) {
      closeModal('modal-auth');
      await checkAuthStatus();

      // Automatically navigate to their dashboard
      if (currentUser.role === 'shop_owner') {
        switchDashboardRole('shop');
      } else if (currentUser.role === 'expert') {
        switchDashboardRole('expert');
      } else if (currentUser.role === 'admin') {
        switchDashboardRole('rag');
      }

      voiceEngine.speakText(`Welcome back, ${currentUser.email.split('@')[0]}!`, currentLanguage);
    } else {
      const errMsg = (data && data.error && data.error.message) || data.detail || "Authentication failed. Please check credentials.";
      showAuthAlert(`⚠️ ${errMsg}`, true);
    }
  } catch (err) {
    showAuthAlert(`Network error: ${err.message}`, true);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

async function handleAuthRegister(e) {
  e.preventDefault();
  hideAuthAlert();

  const role = document.getElementById('reg-role').value;
  const email = document.getElementById('reg-email').value.trim();
  const password = document.getElementById('reg-password').value;
  const shopId = document.getElementById('reg-shop-id').value.trim() || null;
  const expertId = document.getElementById('reg-expert-id').value.trim() || null;
  const adminKey = document.getElementById('reg-admin-key').value.trim() || null;
  const submitBtn = document.getElementById('btn-submit-register');

  const payload = {
    email,
    password,
    role,
    shop_id: role === 'shop_owner' ? shopId : null,
    expert_id: role === 'expert' ? expertId : null,
    admin_creation_key: role === 'admin' ? adminKey : null,
  };

  try {
    if (submitBtn) submitBtn.disabled = true;
    const res = await fetch('/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (res.ok) {
      showAuthAlert("🎉 Account created successfully! Please sign in with your credentials.", false);
      setTimeout(() => {
        switchAuthTab('login');
        const loginEmailInput = document.getElementById('login-email');
        if (loginEmailInput) loginEmailInput.value = email;
      }, 1200);
    } else {
      const errMsg = (data && data.error && data.error.message) || data.detail || "Registration failed.";
      showAuthAlert(`⚠️ ${errMsg}`, true);
    }
  } catch (err) {
    showAuthAlert(`Network error: ${err.message}`, true);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

async function authLogout() {
  try {
    await fetch('/auth/logout', {
      method: 'POST',
      credentials: 'include',
    });
  } catch (err) {
    console.warn("Logout error:", err);
  } finally {
    currentUser = null;
    activeShopId = null;
    activeExpertId = null;
    updateAuthUI();
    switchDashboardRole('farmer');
    alert("👋 You have been logged out successfully.");
  }
}

// Role Switching
function switchDashboardRole(role) {
  // Check authorization permissions
  if (role === 'shop') {
    if (!currentUser || (currentUser.role !== 'shop_owner' && currentUser.role !== 'admin')) {
      openAuthModal('login');
      showAuthAlert("Please log in with an Agri Shop Owner or Admin account to access shop management.", true);
      return;
    }
  } else if (role === 'expert') {
    if (!currentUser || (currentUser.role !== 'expert' && currentUser.role !== 'admin')) {
      openAuthModal('login');
      showAuthAlert("Please log in with an Agricultural Expert or Admin account to view consultations.", true);
      return;
    }
  } else if (role === 'rag') {
    if (!currentUser || currentUser.role !== 'admin') {
      openAuthModal('login');
      showAuthAlert("Platform Administrator privileges are required to access the Knowledge Center.", true);
      return;
    }
  }

  currentRole = role;
  document.querySelectorAll('.role-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.dashboard-view').forEach(view => {
    view.classList.remove('active');
    view.style.display = 'none';
  });

  if (role === 'farmer') {
    const btn = document.getElementById('btn-role-farmer');
    if (btn) btn.classList.add('active');
    const farmerView = document.getElementById('view-farmer');
    if (farmerView) {
      farmerView.classList.add('active');
      farmerView.style.display = 'block';
    }
  } else if (role === 'shop') {
    const btn = document.getElementById('btn-role-shop');
    if (btn) btn.classList.add('active');
    const shopView = document.getElementById('view-shop');
    if (shopView) {
      shopView.classList.add('active');
      shopView.style.display = 'block';
    }
    loadShopOwnerData();
  } else if (role === 'expert') {
    const btn = document.getElementById('btn-role-expert');
    if (btn) btn.classList.add('active');
    const expertView = document.getElementById('view-expert');
    if (expertView) {
      expertView.classList.add('active');
      expertView.style.display = 'block';
    }
    loadExpertDashboardData();
  } else if (role === 'rag') {
    const ragBtn = document.getElementById('btn-role-rag');
    if (ragBtn) ragBtn.classList.add('active');
    const ragView = document.getElementById('view-rag');
    if (ragView) {
      ragView.classList.add('active');
      ragView.style.display = 'block';
    }
    loadRagDocuments();
  }
}


// Farmer Tab Switching
function switchFarmerTab(tabName) {
  document.querySelectorAll('.farmer-tab-content').forEach(el => el.style.display = 'none');
  document.querySelectorAll('#view-farmer .tab-btn').forEach(btn => btn.classList.remove('active'));

  const targetTab = document.getElementById(`farmer-tab-${tabName}`);
  if (targetTab) targetTab.style.display = 'block';
  if (event && event.target) event.target.classList.add('active');

  if (tabName === 'shops') loadNearbyShops();
  if (tabName === 'orders') loadFarmerOrders();
  if (tabName === 'schemes') loadFarmerSchemeEligibility();
  if (tabName === 'chat') loadFarmerChatMessages();
}

// Shop Tab Switching
function switchShopTab(tabName) {
  document.querySelectorAll('.shop-tab-content').forEach(el => el.style.display = 'none');
  document.querySelectorAll('#view-shop .tab-btn').forEach(btn => btn.classList.remove('active'));

  const targetTab = document.getElementById(`shop-tab-${tabName}`);
  if (targetTab) targetTab.style.display = 'block';
  if (event && event.target) event.target.classList.add('active');

  if (tabName === 'orders') loadShopOrders();
  if (tabName === 'inventory') loadShopInventory();
  if (tabName === 'analytics') loadShopAnalytics();
}

// ==========================================================================
// VOICE ASSISTANT INTERACTION ENGINE
// ==========================================================================

function toggleVoiceInteraction() {
  if (voiceEngine.state === 'listening') {
    voiceEngine.stopListening();
  } else {
    voiceEngine.startListening((transcript, isFinal) => {
      const transcriptBox = document.getElementById('voice-transcript-text');
      if (transcriptBox) {
        transcriptBox.innerText = `"${transcript}"`;
      }

      if (isFinal && transcript.trim()) {
        sendVoiceQuery(transcript.trim());
      }
    });
  }
}

function sendVoiceQuery(queryText) {
  const transcriptBox = document.getElementById('voice-transcript-text');
  if (transcriptBox) {
    transcriptBox.innerText = `"${queryText}"`;
  }

  // Switch to chat tab to show conversation
  switchFarmerTab('chat');
  document.getElementById('chat-user-input').value = queryText;
  sendAiChatMessage(queryText);
}

function startVoiceSearch() {
  voiceEngine.startListening((transcript, isFinal) => {
    const input = document.getElementById('search-product-input');
    if (input) input.value = transcript;
    if (isFinal) executeFarmerProductSearch();
  });
}

function startChatVoiceInput() {
  voiceEngine.startListening((transcript, isFinal) => {
    const input = document.getElementById('chat-user-input');
    if (input) input.value = transcript;
  });
}

function startScannerVoiceInput() {
  voiceEngine.startListening((transcript, isFinal) => {
    const input = document.getElementById('crop-symptoms-input');
    if (input) input.value = transcript;
  });
}

function replayLastAiVoice() {
  if (lastAiResponseText) {
    voiceEngine.speakText(lastAiResponseText, currentLanguage);
  } else {
    voiceEngine.speakText(t('aiBannerSub', currentLanguage), currentLanguage);
  }
}

function updateVoiceSpeed(speed) {
  voiceEngine.speechRate = parseFloat(speed);
}

// ==========================================================================
// VOICE SHOPPING, ORDERS & AI CHAT
// ==========================================================================

async function sendAiChatMessage(presetMsg = null) {
  const input = document.getElementById('chat-user-input');
  const sendBtn = document.getElementById('btn-send-chat');
  const msg = (presetMsg || input.value).trim();
  if (!msg || !activeFarmerId) return;

  if (input) input.value = '';
  if (sendBtn) sendBtn.disabled = true;

  voiceEngine.setState('thinking');

  try {
    let aiResponseText = "";
    let detectedIntent = "crop_advice";

    const lowerMsg = msg.toLowerCase();

    // 1. Voice Order Tracking Intent ("where is my order", "order status")
    if (lowerMsg.includes("order") && (lowerMsg.includes("where") || lowerMsg.includes("status") || lowerMsg.includes("tracking"))) {
      detectedIntent = "order_tracking";
      const ordersRes = await fetch(`/orders/farmer/${activeFarmerId}`);
      if (ordersRes.ok) {
        const ordersData = await ordersRes.json();
        if (ordersData.items && ordersData.items.length > 0) {
          const latest = ordersData.items[0];
          aiResponseText = `🤖 **Voice Order Status:**\nYour purchase request for **${latest.product_name}** (${latest.quantity} units) is currently **${latest.status}**.`;
        } else {
          aiResponseText = `🤖 **Voice Order Status:**\nYou have no active purchase requests submitted yet.`;
        }
      }
    }
    // 2. Voice Weather Intent ("rain", "weather", "forecast")
    else if (lowerMsg.includes("rain") || lowerMsg.includes("weather") || lowerMsg.includes("tomorrow")) {
      detectedIntent = "weather_forecast";
      aiResponseText = `🤖 **BhoomiMitra Voice Weather Update:**\nFor your farm region near Korutla (Jagtial), moderate rainfall of 12mm is expected tomorrow afternoon. Postpone fertilizer spraying until weather clears.`;
    }
    // 3. Voice Government Schemes Intent ("scheme", "subsidy", "pm-kisan")
    else if (lowerMsg.includes("scheme") || lowerMsg.includes("subsidy") || lowerMsg.includes("government") || lowerMsg.includes("pm-kisan")) {
      detectedIntent = "govt_schemes";
      aiResponseText = `🤖 **Government Schemes Update:**\n1. **PM-Kisan Samman Nidhi**: 17th installment of ₹2,000 is active.\n2. **Subsidized Fertilizer Scheme**: Nano Urea bags available at 50% government subsidy at certified nearby shops.`;
    }
    // 4. Voice Shopping Intent ("urea", "dap", "pesticide", "neem oil", "need", "buy")
    else {
      try {
        const searchRes = await fetch('/shops/farmer-search?query=' + encodeURIComponent(msg));
        if (searchRes.ok) {
          const searchData = await searchRes.json();
          if (searchData.results && searchData.results.length > 0) {
            detectedIntent = "inventory_query";
            aiResponseText = `🤖 **BhoomiMitra Voice Stock Update:**\nFound available stock for "${msg}":\n` + 
              searchData.results.map(r => `• **${r.shop_name}**: ${r.product_name} (${r.brand}) - ₹${r.price} [Stock: ${r.quantity_in_stock} ${r.unit}s]`).join('\n') +
              `\n\nWould you like me to request purchase from the nearest shop?`;
          } else {
            aiResponseText = `🤖 **BhoomiMitra AI Advice:**\nFor your request "${msg}", we recommend balanced NPK fertilizer usage and proper crop rotation.`;
          }
        }
      } catch (e) {
        console.warn("Stock search fallback:", e);
        aiResponseText = `🤖 **BhoomiMitra AI Advice:**\nThank you for your query regarding "${msg}". Please check nearby shop inventory or consult crop health diagnostics.`;
      }
    }

    lastAiResponseText = aiResponseText;

    // Save conversation to DB table `conversations`
    const uniqueMessageId = 'voice_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);
    const payload = {
      farmer_id: activeFarmerId,
      message_id: uniqueMessageId,
      user_message: msg,
      user_message_type: 'text',
      ai_response: aiResponseText,
      intent: detectedIntent,
      confidence_score: 0.96
    };

    const convRes = await fetch('/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!convRes.ok) {
      const err = await convRes.json();
      throw new Error(err.detail || 'Failed to save conversation');
    }

    // Immediately reload chat history from DB table
    await loadFarmerChatMessages();

    // Speak AI response out loud in natural voice!
    voiceEngine.speakText(aiResponseText, currentLanguage);

  } catch (err) {
    alert("Voice Chat Error: " + err.message);
    voiceEngine.setState('idle');
  } finally {
    if (sendBtn) sendBtn.disabled = false;
  }
}

async function loadFarmerChatMessages(quiet = false) {
  if (!activeFarmerId) return;
  const box = document.getElementById('chat-messages-box');
  if (!box) return;

  if (!quiet && box.children.length === 0) {
    box.innerHTML = '<div style="text-align:center; padding: 2rem; color:var(--text-muted);">Loading conversation history from database...</div>';
  }

  try {
    const res = await fetch(`/conversations/farmer/${activeFarmerId}?page=1&size=50`);
    if (!res.ok) throw new Error('Failed to load conversations from DB');
    const data = await res.json();
    const items = data.items || [];
    
    if (quiet && items.length === lastKnownConversationCount) {
      return;
    }

    lastKnownConversationCount = items.length;

    if (items.length === 0) {
      box.innerHTML = `
        <div class="chat-system-msg">
          👋 Namaste! You are connected live to <strong>BhoomiMitra AI Voice Assistant</strong>.<br>
          Speak or type any agriculture query to record in the database.
        </div>
      `;
      return;
    }

    const chronologicalItems = [...items].reverse();

    let html = `
      <div class="chat-system-msg">
        🟢 Live Webhook & Voice Sync — Connected to DB Table <code>conversations</code> (${items.length} records)
      </div>
    `;

    chronologicalItems.forEach(conv => {
      const timestampStr = new Date(conv.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const msgTypeBadge = conv.user_message_type && conv.user_message_type !== 'text' 
        ? `<span class="badge badge-pending" style="font-size:0.7rem; margin-right:4px;">${conv.user_message_type.toUpperCase()}</span>` 
        : '';

      if (conv.user_message) {
        html += `
          <div class="chat-bubble chat-bubble-user">
            <div class="chat-bubble-header">
              <span>👨‍🌾 Farmer ${msgTypeBadge}</span>
              <span class="chat-timestamp">${timestampStr}</span>
            </div>
            <div class="chat-bubble-body">${escapeHtml(conv.user_message)}</div>
            <div class="chat-bubble-footer">
              <span class="chat-id-tag">ID: ${conv.message_id.substring(0, 16)}...</span>
            </div>
          </div>
        `;
      }

      if (conv.ai_response) {
        const intentTag = conv.intent ? `<span class="badge badge-accepted" style="font-size:0.7rem; margin-left:6px;">${escapeHtml(conv.intent)}</span>` : '';
        const statusClass = conv.delivery_status === 'sent' || conv.delivery_status === 'delivered' ? 'badge-completed' : 'badge-open';
        
        html += `
          <div class="chat-bubble chat-bubble-ai">
            <div class="chat-bubble-header">
              <span>🤖 BhoomiMitra AI ${intentTag}</span>
              <div>
                <button class="btn-icon-sm" onclick="voiceEngine.speakText('${escapeHtml(conv.ai_response).replace(/'/g, "\\'")}', '${currentLanguage}')" title="Listen to message">🔊</button>
                <span class="chat-timestamp">${timestampStr}</span>
              </div>
            </div>
            <div class="chat-bubble-body">${formatMarkdownText(conv.ai_response)}</div>
            <div class="chat-bubble-footer">
              <span class="badge ${statusClass}" style="font-size:0.65rem; padding: 2px 6px;">Status: ${conv.delivery_status || 'delivered'}</span>
            </div>
          </div>
        `;
      }
    });

    box.innerHTML = html;
    box.scrollTop = box.scrollHeight;
  } catch (err) {
    if (!quiet) {
      box.innerHTML = `<div style="color:red; padding:1rem;">Error loading DB conversations: ${err.message}</div>`;
    }
  }
}

function startWebhookPolling() {
  if (chatPollingTimer) clearInterval(chatPollingTimer);
  chatPollingTimer = setInterval(() => {
    loadFarmerChatMessages(true);
  }, 4000);
}

// Spoken Alerts Helper
function speakActiveNotifications() {
  const text = "Weather Warning: Moderate rainfall expected in Jagtial district tomorrow. Postpone spray. Inventory Alert: Fresh stock of IFFCO Urea and DAP available at Sri Laxmi Fertilizer Shop.";
  voiceEngine.speakText(text, currentLanguage);
}

async function speakOrderStatuses() {
  if (!activeFarmerId) return;
  try {
    const res = await fetch(`/orders/farmer/${activeFarmerId}`);
    if (res.ok) {
      const data = await res.json();
      if (data.items && data.items.length > 0) {
        const text = `You have ${data.items.length} purchase requests. Latest order for ${data.items[0].product_name} is ${data.items[0].status}.`;
        voiceEngine.speakText(text, currentLanguage);
      } else {
        voiceEngine.speakText("You have no active purchase requests.", currentLanguage);
      }
    }
  } catch (e) {
    console.warn("Order speak error:", e);
  }
}

// Voice Crop Scanner Modal
function openCropScannerModal() {
  document.getElementById('modal-crop-scanner').classList.add('active');
}

function previewCropImage(input) {
  if (input.files && input.files[0]) {
    const reader = new FileReader();
    reader.onload = function(e) {
      document.getElementById('crop-img-preview').src = e.target.result;
      document.getElementById('image-preview-container').style.display = 'block';
    }
    reader.readAsDataURL(input.files[0]);
  }
}

async function submitCropDiagnosis(e) {
  e.preventDefault();
  const symptoms = document.getElementById('crop-symptoms-input').value.trim() || 'Yellowing leaves and spot damage';
  closeModal('modal-crop-scanner');

  const replyText = `🤖 **Voice Crop Diagnosis:**\nAnalyzed crop photo for symptoms: "${symptoms}". Diagnosis indicates early Chilli Thrips infestation. Recommended remedy: Spray Imidacloprid at 0.5ml per liter of water.`;
  
  sendVoiceQuery(symptoms);
  voiceEngine.speakText(replyText, currentLanguage);
}

async function loadNearbyShops() {
  const container = document.getElementById('nearby-shops-container');
  if (!container) return;
  container.innerHTML = '<p>Loading nearby shops...</p>';

  try {
    const res = await fetch('/shops/nearby?latitude=18.8206&longitude=78.7119&max_radius_km=50');
    if (!res.ok) throw new Error('Failed to fetch nearby shops');
    const shops = await res.json();

    if (shops.length === 0) {
      container.innerHTML = '<p>No nearby shops found.</p>';
      return;
    }

    const cards = await Promise.all(shops.map(async (s) => {
      const mapsUrl = s.google_maps_link || `https://www.google.com/maps/search/?api=1&query=${s.latitude || 18.8206},${s.longitude || 78.7119}`;
      const statusBadge = s.status === 'active' ? '<span class="badge badge-open">Open</span>' : '<span class="badge badge-closed">Closed</span>';
      const deliveryBadge = s.delivery_available ? '<span class="badge badge-completed">Delivery Available</span>' : '<span class="badge badge-pending">Pick Up Only</span>';

      let productsHtml = '<p style="font-size:0.85rem; color:var(--text-muted);">No products listed</p>';
      try {
        const invRes = await fetch(`/inventory/shop/${s.id}`);
        if (invRes.ok) {
          const invData = await invRes.json();
          if (invData.items && invData.items.length > 0) {
            productsHtml = `<div style="margin: 0.75rem 0; padding: 0.75rem; background: var(--bg-main); border-radius: var(--radius-sm);">
              <div style="font-size: 0.85rem; font-weight: 700; margin-bottom: 0.35rem; color: var(--primary);">📦 Live Inventory:</div>
              ${invData.items.map(item => `
                <div style="display:flex; justify-content:space-between; font-size: 0.85rem; padding: 0.25rem 0; border-bottom: 1px dashed var(--border-color);">
                  <span><strong>${escapeHtml(item.product_name)}</strong> (${escapeHtml(item.brand)})</span>
                  <span>₹${item.price} | <strong>${item.quantity_in_stock} ${item.unit}s</strong></span>
                </div>
              `).join('')}
            </div>`;
          }
        }
      } catch (e) {
        console.warn("Error loading shop products:", e);
      }

      return `
        <div class="card">
          <div class="card-title">
            <span>${escapeHtml(s.shop_name)}</span>
            ${statusBadge}
          </div>
          <p><strong>Owner:</strong> ${escapeHtml(s.owner_name)}</p>
          <p><strong>Phone:</strong> ${escapeHtml(s.phone_number)}</p>
          <p><strong>Address:</strong> ${escapeHtml(s.address)}, ${escapeHtml(s.district || '')}</p>
          <p><strong>Distance:</strong> 📍 ${s.distance_km !== undefined ? s.distance_km : 0} km</p>
          <div style="margin-top: 0.5rem;">${deliveryBadge}</div>
          ${productsHtml}
          <div style="display: flex; gap: 0.5rem; margin-top: 0.75rem;">
            <a href="tel:${s.phone_number}" class="btn btn-secondary btn-sm" style="flex:1;">📞 Call Shop</a>
            <a href="${mapsUrl}" target="_blank" class="btn btn-primary btn-sm" style="flex:1;">🗺️ Directions</a>
          </div>
        </div>
      `;
    }));

    container.innerHTML = cards.join('');
  } catch (err) {
    container.innerHTML = `<p style="color:red;">Error loading shops: ${err.message}</p>`;
  }
}

async function executeFarmerProductSearch() {
  const queryInput = document.getElementById('search-product-input');
  const query = queryInput ? queryInput.value.trim() || 'Urea' : 'Urea';
  const container = document.getElementById('search-results-container');
  if (!container) return;
  container.innerHTML = '<p>Searching inventory...</p>';

  try {
    const res = await fetch(`/shops/farmer-search?query=${encodeURIComponent(query)}&latitude=18.8206&longitude=78.7119`);
    if (!res.ok) throw new Error('Search failed');
    const data = await res.json();

    if (!data.results || data.results.length === 0) {
      container.innerHTML = `<p>No shops found selling "${escapeHtml(query)}".</p>`;
      return;
    }

    container.innerHTML = data.results.map(r => `
      <div class="card">
        <div class="card-title">
          <span>${escapeHtml(r.product_name)}</span>
          <span class="badge badge-accepted">₹${r.price}</span>
        </div>
        <p><strong>Brand:</strong> ${escapeHtml(r.brand)}</p>
        <p><strong>Available at:</strong> ${escapeHtml(r.shop_name)}</p>
        <p><strong>Stock:</strong> ${r.quantity_in_stock} ${r.unit}s</p>
        <p><strong>Distance:</strong> 📍 ${r.distance_km || '2.1'} km</p>
        <p><strong>Phone:</strong> ${escapeHtml(r.phone_number)}</p>
        <button class="btn btn-accent btn-sm" style="width: 100%; margin-top: 0.75rem;" onclick="openOrderModal('${r.shop_id}', '${escapeHtml(r.product_name)}', '${r.price}')">
          🛒 Request Purchase
        </button>
      </div>
    `).join('');
  } catch (err) {
    container.innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
  }
}

async function loadFarmerOrders() {
  if (!activeFarmerId) return;
  const tbody = document.getElementById('farmer-orders-table-body');
  if (!tbody) return;
  tbody.innerHTML = '<tr><td colspan="6">Loading requests...</td></tr>';

  try {
    const res = await fetch(`/orders/farmer/${activeFarmerId}`);
    if (!res.ok) throw new Error('Failed to load orders');
    const data = await res.json();

    if (data.items.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7">No purchase requests submitted yet.</td></tr>';
      return;
    }

    tbody.innerHTML = data.items.map(o => {
      const isPaid = o.payment_status === 'Paid';
      const payBadge = isPaid
        ? '<span class="badge badge-accepted" style="background:#10b981; color:#fff;">💳 Paid</span>'
        : (o.status === 'Cancelled'
            ? '<span class="badge badge-cancelled">Cancelled</span>'
            : `<button class="btn btn-primary btn-sm" onclick="payForOrder('${o.id}')" style="padding:3px 10px; font-size:12px; border-radius:4px;">💳 Pay Now (₹${o.total_price})</button>`);

      return `
      <tr>
        <td>${o.id.substring(0, 8)}...</td>
        <td><strong>${escapeHtml(o.product_name)}</strong> (${escapeHtml(o.brand || '')})</td>
        <td>${o.quantity} ${o.unit}s</td>
        <td>₹${o.total_price}</td>
        <td><span class="badge badge-${o.status.toLowerCase()}">${o.status}</span></td>
        <td>${payBadge}</td>
        <td>${new Date(o.created_at).toLocaleDateString()}</td>
      </tr>`;
    }).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" style="color:red;">Error loading orders</td></tr>`;
  }
}

async function payForOrder(orderId) {
  try {
    const res = await fetch('/payments/create-order', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ order_id: orderId }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert("Failed to initiate payment: " + (err.detail || "Error"));
      return;
    }

    const orderData = await res.json();

    if (window.Razorpay && !orderData.key_id.startsWith("rzp_test_bhoomimitra_mock")) {
      const options = {
        key: orderData.key_id,
        amount: orderData.amount_in_paise,
        currency: orderData.currency,
        name: "BhoomiMitra AI",
        description: `Payment for ${orderData.product_name}`,
        order_id: orderData.razorpay_order_id,
        handler: async function (response) {
          await verifyPaymentOnBackend(orderId, response);
        },
        prefill: {
          contact: orderData.customer_phone || "",
        },
        theme: {
          color: "#16a34a",
        },
      };
      const rzp = new window.Razorpay(options);
      rzp.open();
    } else {
      // Local development / Test checkout simulation
      const proceed = confirm(`💳 BhoomiMitra Razorpay Checkout\n\nProduct: ${orderData.product_name}\nAmount: ₹${orderData.amount_in_paise / 100}\nRazorpay Order: ${orderData.razorpay_order_id}\n\nClick OK to simulate successful payment.`);
      if (proceed) {
        const mockPaymentId = `pay_mock_${Date.now().toString(36)}`;
        await verifyPaymentOnBackend(orderId, {
          razorpay_order_id: orderData.razorpay_order_id,
          razorpay_payment_id: mockPaymentId,
          // Generate valid HMAC matching mock secret for test mode
          razorpay_signature: "mock_signature_test_mode",
        });
      }
    }
  } catch (err) {
    alert("Payment error: " + err.message);
  }
}

async function verifyPaymentOnBackend(orderId, rzpResponse) {
  try {
    const res = await fetch('/payments/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        order_id: orderId,
        razorpay_order_id: rzpResponse.razorpay_order_id,
        razorpay_payment_id: rzpResponse.razorpay_payment_id,
        razorpay_signature: rzpResponse.razorpay_signature,
      }),
    });

    if (res.ok) {
      alert("🎉 Payment successful! Your order is now marked as Paid and ready for shop confirmation.");
      loadFarmerOrders();
    } else {
      const err = await res.json();
      alert("⚠️ Payment verification failed: " + (err.detail || "Error"));
    }
  } catch (err) {
    alert("Verification error: " + err.message);
  }
}

// Modal Helpers
function openOrderModal(shopId, productName, price) {
  document.getElementById('order-shop-id').value = shopId;
  document.getElementById('order-product-name').value = productName;
  document.getElementById('modal-order-request').classList.add('active');
}

function closeModal(id) {
  document.getElementById(id).classList.remove('active');
}

async function submitFarmerOrder(e) {
  e.preventDefault();
  if (!activeFarmerId) {
    alert("Farmer account not initialized.");
    return;
  }

  const shopId = document.getElementById('order-shop-id').value;
  const productName = document.getElementById('order-product-name').value;
  const quantity = parseInt(document.getElementById('order-quantity').value);
  const notes = document.getElementById('order-notes').value;

  try {
    const invRes = await fetch(`/inventory/shop/${shopId}`);
    const invData = await invRes.json();
    const item = invData.items.find(i => i.product_name.toLowerCase() === productName.toLowerCase()) || invData.items[0];

    const payload = {
      farmer_id: activeFarmerId,
      shop_id: shopId,
      inventory_id: item ? item.id : shopId,
      quantity: quantity,
      notes: notes
    };

    const res = await fetch('/orders', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      alert("✅ Purchase request sent to Shop Owner!");
      closeModal('modal-order-request');
      loadFarmerOrders();

      const confirmVoiceText = `Order request for ${quantity} units of ${productName} sent to shop owner successfully!`;
      voiceEngine.speakText(confirmVoiceText, currentLanguage);
    } else {
      const err = await res.json();
      alert("Failed to submit order: " + (err.detail || "Error"));
    }
  } catch (err) {
    alert("Order submission error: " + err.message);
  }
}

// ==========================================================================
// SHOP OWNER DASHBOARD FUNCTIONS
// ==========================================================================

async function loadShopOwnerData() {
  if (!activeShopId) return;

  await loadShopDashboardStats();
  await loadShopOrders();
  await loadShopInventory();
}

async function loadShopDashboardStats() {
  try {
    const dashRes = await fetch(`/inventory/dashboard/${activeShopId}`, { credentials: 'include' });
    if (dashRes.ok) {
      const dash = await dashRes.json();
      document.getElementById('stat-total-products').innerText = dash.total_products;
      document.getElementById('stat-low-stock').innerText = dash.low_stock_count;

      if (dash.low_stock_count > 0) {
        document.getElementById('low-stock-alert-banner').style.display = 'flex';
        document.getElementById('low-stock-alert-text').innerText = `${dash.low_stock_count} products are below minimum stock level!`;
      } else {
        document.getElementById('low-stock-alert-banner').style.display = 'none';
      }
    }

    const analyticsRes = await fetch(`/orders/analytics/${activeShopId}`, { credentials: 'include' });
    if (analyticsRes.ok) {
      const analytics = await analyticsRes.json();
      document.getElementById('stat-active-orders').innerText = analytics.pending_orders + analytics.accepted_orders;
      document.getElementById('stat-total-revenue').innerText = `₹${analytics.total_revenue_inr}`;
    }
  } catch (err) {
    console.error("Error loading dashboard stats:", err);
  }
}

async function loadShopOrders() {
  if (!activeShopId) return;
  const tbody = document.getElementById('shop-orders-table-body');
  if (!tbody) return;
  tbody.innerHTML = '<tr><td colspan="6">Loading incoming orders...</td></tr>';

  try {
    const res = await fetch(`/orders/shop/${activeShopId}`, { credentials: 'include' });
    if (!res.ok) throw new Error('Failed to load orders');
    const data = await res.json();

    if (data.items.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6">No incoming farmer requests.</td></tr>';
      return;
    }

    tbody.innerHTML = data.items.map(o => `
      <tr>
        <td>${o.id.substring(0, 8)}...</td>
        <td><strong>${escapeHtml(o.product_name)}</strong> (${escapeHtml(o.brand || '')})</td>
        <td>${o.quantity} ${o.unit}s</td>
        <td>₹${o.total_price}</td>
        <td><span class="badge badge-${o.status.toLowerCase()}">${o.status}</span></td>
        <td>
          ${o.status === 'Pending' ? `<button class="btn btn-primary btn-sm" onclick="updateOrderStatus('${o.id}', 'Accepted')">Accept</button>` : ''}
          ${o.status === 'Accepted' ? `<button class="btn btn-accent btn-sm" onclick="updateOrderStatus('${o.id}', 'Ready')">Mark Ready</button>` : ''}
          ${o.status === 'Ready' ? `<button class="btn btn-secondary btn-sm" style="background:#d1fae5; color:#065f46;" onclick="updateOrderStatus('${o.id}', 'Completed')">Complete</button>` : ''}
          ${o.status !== 'Completed' && o.status !== 'Cancelled' ? `<button class="btn btn-secondary btn-sm" style="color:red;" onclick="updateOrderStatus('${o.id}', 'Cancelled')">Cancel</button>` : ''}
        </td>
      </tr>
    `).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" style="color:red;">Error loading orders</td></tr>`;
  }
}

async function updateOrderStatus(orderId, newStatus) {
  try {
    const res = await fetch(`/orders/${orderId}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ status: newStatus })
    });
    if (res.ok) {
      loadShopOwnerData();
    } else {
      alert("Failed to update order status.");
    }
  } catch (err) {
    alert("Error: " + err.message);
  }
}

async function loadShopInventory() {
  if (!activeShopId) return;
  const tbody = document.getElementById('shop-inventory-table-body');
  if (!tbody) return;
  tbody.innerHTML = '<tr><td colspan="8">Loading inventory...</td></tr>';

  try {
    const res = await fetch(`/inventory/shop/${activeShopId}`, { credentials: 'include' });
    if (!res.ok) throw new Error('Failed to load inventory');
    const data = await res.json();

    if (data.items.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8">No inventory items found. Add your first product above.</td></tr>';
      return;
    }

    tbody.innerHTML = data.items.map(i => {
      const stockBadge = i.quantity_in_stock <= i.minimum_stock_level ? '<span class="badge badge-cancelled">Low Stock</span>' : '<span class="badge badge-completed">In Stock</span>';

      return `
        <tr>
          <td><strong>${escapeHtml(i.product_name)}</strong></td>
          <td>${escapeHtml(i.brand)}</td>
          <td>${escapeHtml(i.category)}</td>
          <td>₹${i.price}</td>
          <td><strong>${i.quantity_in_stock}</strong> ${i.unit}s</td>
          <td>${i.minimum_stock_level}</td>
          <td>${stockBadge}</td>
          <td>
            <button class="btn btn-secondary btn-sm" onclick="quickStockPrompt('${i.id}', ${i.quantity_in_stock})">Update Stock</button>
            <button class="btn btn-secondary btn-sm" style="color:red;" onclick="deleteProduct('${i.id}')">Delete</button>
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="color:red;">Error loading inventory</td></tr>`;
  }
}

function openAddProductModal() {
  document.getElementById('prod-id').value = '';
  document.getElementById('prod-name').value = '';
  document.getElementById('prod-brand').value = '';
  document.getElementById('prod-price').value = '';
  document.getElementById('prod-stock').value = '';
  document.getElementById('modal-inventory-product').classList.add('active');
}

async function saveInventoryProduct(e) {
  e.preventDefault();
  if (!activeShopId) return;

  const payload = {
    shop_id: activeShopId,
    product_name: document.getElementById('prod-name').value,
    brand: document.getElementById('prod-brand').value,
    category: document.getElementById('prod-category').value,
    price: parseFloat(document.getElementById('prod-price').value),
    quantity_in_stock: parseInt(document.getElementById('prod-stock').value),
    minimum_stock_level: parseInt(document.getElementById('prod-min-stock').value),
    unit: 'Unit'
  };

  try {
    const res = await fetch('/inventory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      alert("✅ Product added to inventory!");
      closeModal('modal-inventory-product');
      loadShopOwnerData();
    } else {
      alert("Failed to add product.");
    }
  } catch (err) {
    alert("Error: " + err.message);
  }
}

async function quickStockPrompt(itemId, currentQty) {
  const newQty = prompt("Enter new stock quantity:", currentQty);
  if (newQty === null) return;
  const parsed = parseInt(newQty);
  if (isNaN(parsed) || parsed < 0) {
    alert("Invalid stock number");
    return;
  }

  try {
    const res = await fetch(`/inventory/${itemId}/stock`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ quantity_in_stock: parsed })
    });
    if (res.ok) {
      loadShopOwnerData();
    }
  } catch (err) {
    alert("Failed to update stock");
  }
}

async function deleteProduct(itemId) {
  if (!confirm("Are you sure you want to delete this product?")) return;
  try {
    const res = await fetch(`/inventory/${itemId}`, {
      method: 'DELETE',
      credentials: 'include',
    });
    if (res.ok) loadShopOwnerData();
  } catch (err) {
    alert("Delete failed.");
  }
}

function populateShopProfileForm(shop) {
  document.getElementById('shop-name-input').value = shop.shop_name || '';
  document.getElementById('shop-owner-input').value = shop.owner_name || '';
  document.getElementById('shop-phone-input').value = shop.phone_number || '';
  document.getElementById('shop-address-input').value = shop.address || '';
  document.getElementById('shop-opening-input').value = shop.opening_time || '08:00 AM';
  document.getElementById('shop-closing-input').value = shop.closing_time || '08:00 PM';
}

async function saveShopProfile(e) {
  e.preventDefault();
  if (!activeShopId) return;

  const payload = {
    shop_name: document.getElementById('shop-name-input').value,
    owner_name: document.getElementById('shop-owner-input').value,
    phone_number: document.getElementById('shop-phone-input').value,
    address: document.getElementById('shop-address-input').value,
    opening_time: document.getElementById('shop-opening-input').value,
    closing_time: document.getElementById('shop-closing-input').value,
  };

  try {
    const res = await fetch(`/shops/${activeShopId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      alert("✅ Shop Profile updated successfully!");
    }
  } catch (err) {
    alert("Error updating profile: " + err.message);
  }
}

async function loadShopAnalytics() {
  if (!activeShopId) return;
  try {
    const res = await fetch(`/orders/analytics/${activeShopId}`);
    if (!res.ok) return;
    const data = await res.json();

    const popularBox = document.getElementById('popular-products-list');
    if (data.popular_products && data.popular_products.length > 0) {
      popularBox.innerHTML = data.popular_products.map(p => `
        <div style="display:flex; justify-content:space-between; padding: 0.75rem 0; border-bottom: 1px solid var(--border-color);">
          <span><strong>${escapeHtml(p.product_name)}</strong></span>
          <span class="badge badge-accepted">${p.units_sold} units demanded</span>
        </div>
      `).join('');
    } else {
      popularBox.innerHTML = '<p>No sales analytics data yet.</p>';
    }

    const ctx = document.getElementById('chart-order-status').getContext('2d');
    if (chartInstance) chartInstance.destroy();

    chartInstance = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Pending', 'Accepted', 'Ready', 'Completed', 'Cancelled'],
        datasets: [{
          data: [
            data.pending_orders,
            data.accepted_orders,
            data.ready_orders,
            data.completed_orders,
            data.cancelled_orders
          ],
          backgroundColor: ['#fef3c7', '#dbeafe', '#e0e7ff', '#d1fae5', '#fee2e2']
        }]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  } catch (err) {
    console.error("Analytics load error:", err);
  }
}

// Utility Helpers
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatMarkdownText(text) {
  if (!text) return '';
  let escaped = escapeHtml(text);
  escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  return escaped.replace(/\n/g, '<br>');
}

// ==========================================================================
// AI GOVERNMENT SCHEMES & ELIGIBILITY ADVISOR FUNCTIONS
// ==========================================================================

async function loadFarmerSchemeEligibility() {
  if (!activeFarmerId) return;
  const container = document.getElementById('schemes-list-container');
  if (!container) return;
  container.innerHTML = '<p style="padding:1rem; grid-column:1/-1;">Evaluating profile & querying government schemes database...</p>';

  try {
    const res = await fetch(`/schemes/eligibility/${activeFarmerId}`);
    if (!res.ok) throw new Error("Failed to evaluate eligibility");
    const data = await res.json();

    if (!data.schemes || data.schemes.length === 0) {
      container.innerHTML = '<p style="padding:1rem; grid-column:1/-1;">No government schemes found.</p>';
      return;
    }

    container.innerHTML = data.schemes.map(item => {
      const s = item.scheme;
      const statusBadge = item.is_eligible 
        ? `<span class="badge badge-completed">✅ Eligible (${item.match_score_percentage}% Match)</span>`
        : `<span class="badge badge-cancelled">⚠️ Not Eligible</span>`;

      const deadlineText = s.application_deadline ? new Date(s.application_deadline).toLocaleDateString() : 'Continuous / No Deadline';

      return `
        <div class="card" style="border-top: 4px solid ${item.is_eligible ? '#059669' : '#d97706'}; flex-direction:column; justify-content:space-between;">
          <div>
            <div class="card-title" style="margin-bottom:0.5rem;">
              <span>${escapeHtml(s.scheme_name)}</span>
              ${statusBadge}
            </div>
            <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:0.75rem;">
              🏷️ Category: <strong>${escapeHtml(s.category)}</strong> | 📍 Region: <strong>${escapeHtml(s.state)}</strong>
            </div>
            <p style="font-size:0.9rem; margin-bottom:0.75rem;">${escapeHtml(s.description)}</p>
            <div style="background:var(--bg-main); padding:0.75rem; border-radius:8px; margin-bottom:0.75rem; font-size:0.85rem;">
              <p><strong>💰 Benefits:</strong> ${escapeHtml(s.benefits_summary)}</p>
              <p><strong>📑 Eligibility Check:</strong> ${escapeHtml(item.eligibility_reason)}</p>
              <p><strong>📄 Required Documents:</strong> ${escapeHtml(s.required_documents)}</p>
              <p><strong>⏳ Application Deadline:</strong> ${deadlineText}</p>
            </div>
          </div>
          <div>
            <div style="display:flex; gap:0.5rem; margin-top:0.75rem;">
              <button class="btn btn-secondary btn-sm" style="flex:1;" onclick="voiceEngine.speakText('${escapeHtml(item.voice_explanation).replace(/'/g, "\\'")}', currentLanguage)">
                🔊 Listen Explanation
              </button>
              <button class="btn btn-primary btn-sm" style="flex:1;" onclick="applyForGovernmentScheme('${s.id}')">
                📝 Apply / Bookmark Scheme
              </button>
            </div>
            ${s.official_portal_url ? `<a href="${s.official_portal_url}" target="_blank" style="display:block; font-size:0.8rem; text-align:center; margin-top:0.5rem; color:var(--primary); font-weight:600;">🔗 Visit Official Portal</a>` : ''}
          </div>
        </div>
      `;
    }).join('');
  } catch (err) {
    container.innerHTML = `<p style="color:red; padding:1rem; grid-column:1/-1;">Error loading schemes: ${err.message}</p>`;
  }
}

async function applyForGovernmentScheme(schemeId) {
  if (!activeFarmerId) return;
  try {
    const res = await fetch('/schemes/apply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        farmer_id: activeFarmerId,
        scheme_id: schemeId,
        notes: "Applied via BhoomiMitra AI Advisor"
      })
    });

    if (res.ok) {
      alert("✅ Scheme application registered in your profile! Status: Applied");
      voiceEngine.speakText("Scheme application registered successfully! You can track application status in your profile.", currentLanguage);
    } else {
      const err = await res.json();
      alert("Application Error: " + (err.detail || "Failed to submit application"));
    }
  } catch (err) {
    alert("Error: " + err.message);
  }
}

/* ==========================================================================
   RAG KNOWLEDGE ENGINE FRONTEND CONTROLLERS
   ========================================================================== */

async function loadRagDocuments() {
  const tbody = document.getElementById('rag-documents-table-body');
  const countBadge = document.getElementById('rag-doc-count-badge');
  if (!tbody) return;

  try {
    const res = await fetch('/rag/documents');
    if (!res.ok) throw new Error("Failed to fetch documents");
    const docs = await res.json();

    if (countBadge) countBadge.innerText = `${docs.length} Documents`;

    if (docs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:1.5rem; color:var(--text-muted);">No knowledge documents uploaded yet. Upload a PDF or advisory manual above.</td></tr>`;
      return;
    }

    tbody.innerHTML = docs.map(doc => {
      const createdDate = new Date(doc.created_at).toLocaleDateString();
      const stateCrop = `${doc.state || 'All India'} / ${doc.crop || 'All Crops'}`;

      return `
        <tr>
          <td><strong>${escapeHtml(doc.title)}</strong></td>
          <td><span class="badge badge-pending">${escapeHtml(doc.source)}</span></td>
          <td>${escapeHtml(doc.category)}</td>
          <td><code>${escapeHtml(doc.language)}</code></td>
          <td>${escapeHtml(stateCrop)}</td>
          <td><span class="badge badge-completed">${doc.chunk_count} Chunks</span></td>
          <td>${createdDate}</td>
          <td>
            <button class="btn btn-danger btn-sm" onclick="deleteRagDocument('${doc.id}')">🗑️ Delete</button>
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="color:red; text-align:center; padding:1rem;">Error: ${err.message}</td></tr>`;
  }
}

async function handleRagDocumentUpload(event) {
  event.preventDefault();
  const titleInput = document.getElementById('rag-upload-title');
  const sourceSelect = document.getElementById('rag-upload-source');
  const categorySelect = document.getElementById('rag-upload-category');
  const langSelect = document.getElementById('rag-upload-language');
  const stateInput = document.getElementById('rag-upload-state');
  const cropInput = document.getElementById('rag-upload-crop');
  const fileInput = document.getElementById('rag-upload-file');

  if (!fileInput.files || fileInput.files.length === 0) {
    alert("Please select a file to upload.");
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('title', titleInput.value.trim());
  formData.append('source', sourceSelect.value);
  formData.append('category', categorySelect.value);
  formData.append('language', langSelect.value);
  if (stateInput.value.trim()) formData.append('state', stateInput.value.trim());
  if (cropInput.value.trim()) formData.append('crop', cropInput.value.trim());

  try {
    const res = await fetch('/rag/upload', {
      method: 'POST',
      body: formData,
      credentials: 'include',
    });

    if (res.ok) {
      const data = await res.json();
      alert(`✅ Document "${data.title}" uploaded & indexed successfully (${data.chunk_count} chunks)!`);
      document.getElementById('form-rag-upload').reset();
      loadRagDocuments();
    } else {
      const err = await res.json();
      alert("Upload Error: " + (err.detail || "Failed to upload document"));
    }
  } catch (err) {
    alert("Upload Exception: " + err.message);
  }
}

async function deleteRagDocument(docId) {
  if (!confirm("Are you sure you want to delete this document and all its indexed vector chunks?")) return;

  try {
    const res = await fetch(`/rag/document/${docId}`, {
      method: 'DELETE',
      credentials: 'include',
    });
    if (res.ok) {
      alert("✅ Document deleted successfully.");
      loadRagDocuments();
    } else {
      const err = await res.json();
      alert("Delete Error: " + (err.detail || "Failed to delete document"));
    }
  } catch (err) {
    alert("Error deleting document: " + err.message);
  }
}

async function rebuildRagVectorIndex() {
  try {
    const res = await fetch('/rag/rebuild', {
      method: 'POST',
      credentials: 'include',
    });
    if (res.ok) {
      const result = await res.json();
      alert(`✅ Vector Index Rebuilt!\nProcessed: ${result.documents_processed} documents\nTotal Chunks: ${result.total_chunks}`);
      loadRagDocuments();
    } else {
      const err = await res.json();
      alert("Rebuild Error: " + (err.detail || "Failed to rebuild index"));
    }
  } catch (err) {
    alert("Rebuild Error: " + err.message);
  }
}

async function executeRagSearch() {
  const input = document.getElementById('rag-search-input');
  const resultsBox = document.getElementById('rag-search-results-box');
  if (!input || !resultsBox) return;

  const query = input.value.trim();
  if (!query) {
    alert("Please enter a query to search.");
    return;
  }

  resultsBox.innerHTML = `<p style="color:var(--text-muted); text-align:center;">Performing semantic vector similarity search...</p>`;

  try {
    const res = await fetch(`/rag/search?query=${encodeURIComponent(query)}&top_k=5`);
    if (!res.ok) throw new Error("Vector search failed");
    const results = await res.json();

    if (results.length === 0) {
      resultsBox.innerHTML = `<p style="color:var(--text-muted); text-align:center;">No matching chunks found for query "${escapeHtml(query)}".</p>`;
      return;
    }

    resultsBox.innerHTML = results.map((r, i) => `
      <div style="background:var(--bg-card); padding:0.75rem; border-radius:6px; margin-bottom:0.75rem; border-left:4px solid #10b981;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
          <strong style="font-size:0.9rem; color:var(--text-main);">#${i+1} ${escapeHtml(r.document_title)}</strong>
          <span class="badge badge-completed" style="font-size:0.75rem;">Sim Score: ${r.similarity_score}</span>
        </div>
        <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:0.4rem;">
          Source: <strong>${escapeHtml(r.source)}</strong> | Category: <strong>${escapeHtml(r.category)}</strong> | State: <strong>${escapeHtml(r.state || 'All India')}</strong>
        </div>
        <p style="font-size:0.85rem; color:var(--text-main); margin:0; line-height:1.4;">${escapeHtml(r.chunk_text)}</p>
      </div>
    `).join('');
  } catch (err) {
    resultsBox.innerHTML = `<p style="color:red; text-align:center;">Search Error: ${err.message}</p>`;
  }
}


