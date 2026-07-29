/* ==========================================================================
   BhoomiMitra AI — SPA Frontend Controller
   ========================================================================== */

let currentRole = 'farmer';
let activeFarmerId = null;
let activeShopId = null;
let chartInstance = null;

// Multi-language dictionary
const translations = {
  en: {
    title: "BhoomiMitra AI",
    nearbyShops: "Nearby Agriculture Shops",
    productSearch: "Product Search",
    myOrders: "My Order Requests",
  },
  te: {
    title: "భూమిమిత్ర AI",
    nearbyShops: "సమీప వ్యవసాయ దుకాణాలు",
    productSearch: "ఉత్పత్తి శోధన",
    myOrders: "నా కొనుగోలు అభ్యర్థనలు",
  },
  hi: {
    title: "भूमिमित्र AI",
    nearbyShops: "निकटतम कृषि दुकानें",
    productSearch: "उत्पाद खोज",
    myOrders: "मेरे ऑर्डर अनुरोध",
  },
  kn: {
    title: "ಭೂಮಿಮಿತ್ರ AI",
    nearbyShops: "ಸಮೀಪದ ಕೃಷಿ ಅಂಗಡಿಗಳು",
    productSearch: "ಉತ್ಪನ್ನ ಹುಡುಕಾಟ",
    myOrders: "ನನ್ನ ಆದೇಶ ವಿನಂತಿಗಳು",
  }
};

document.addEventListener('DOMContentLoaded', async () => {
  await initializeData();
});

async function initializeData() {
  try {
    // 1. Fetch first registered shop
    const shopRes = await fetch('/shops?page=1&size=1');
    if (shopRes.ok) {
      const shopData = await shopRes.json();
      if (shopData.items && shopData.items.length > 0) {
        activeShopId = shopData.items[0].id;
        populateShopProfileForm(shopData.items[0]);
      }
    }

    // 2. Fetch first farmer
    const farmerRes = await fetch('/farmers?page=1&size=1');
    if (farmerRes.ok) {
      const farmerData = await farmerRes.json();
      if (farmerData.items && farmerData.items.length > 0) {
        activeFarmerId = farmerData.items[0].id;
      }
    }

    // 3. Load initial views
    await loadNearbyShops();
    await loadFarmerOrders();
    if (activeShopId) {
      await loadShopOwnerData();
    }
  } catch (err) {
    console.error("Failed to initialize dashboard data:", err);
  }
}

// Role Switching
function switchDashboardRole(role) {
  currentRole = role;
  document.querySelectorAll('.role-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.dashboard-view').forEach(view => view.classList.remove('active'));

  if (role === 'farmer') {
    document.getElementById('btn-role-farmer').classList.add('active');
    document.getElementById('view-farmer').classList.add('active');
  } else {
    document.getElementById('btn-role-shop').classList.add('active');
    document.getElementById('view-shop').classList.add('active');
    loadShopOwnerData();
  }
}

function changeLanguage(lang) {
  const dict = translations[lang] || translations.en;
  console.log("Language changed to:", lang, dict);
}

// Farmer Tab Switching
function switchFarmerTab(tabName) {
  document.querySelectorAll('.farmer-tab-content').forEach(el => el.style.display = 'none');
  document.querySelectorAll('#view-farmer .tab-btn').forEach(btn => btn.classList.remove('active'));

  document.getElementById(`farmer-tab-${tabName}`).style.display = 'block';
  event.target.classList.add('active');

  if (tabName === 'shops') loadNearbyShops();
  if (tabName === 'orders') loadFarmerOrders();
}

// Shop Tab Switching
function switchShopTab(tabName) {
  document.querySelectorAll('.shop-tab-content').forEach(el => el.style.display = 'none');
  document.querySelectorAll('#view-shop .tab-btn').forEach(btn => btn.classList.remove('active'));

  document.getElementById(`shop-tab-${tabName}`).style.display = 'block';
  event.target.classList.add('active');

  if (tabName === 'orders') loadShopOrders();
  if (tabName === 'inventory') loadShopInventory();
  if (tabName === 'analytics') loadShopAnalytics();
}

// ==========================================================================
// FARMER DASHBOARD FUNCTIONS
// ==========================================================================

async function loadNearbyShops() {
  const container = document.getElementById('nearby-shops-container');
  container.innerHTML = '<p>Loading nearby shops...</p>';

  try {
    const res = await fetch('/shops/nearby?latitude=16.3067&longitude=80.4365&max_radius_km=50');
    if (!res.ok) throw new Error('Failed to fetch nearby shops');
    const shops = await res.json();

    if (shops.length === 0) {
      container.innerHTML = '<p>No nearby shops found.</p>';
      return;
    }

    const cards = await Promise.all(shops.map(async (s) => {
      const mapsUrl = s.google_maps_link || `https://www.google.com/maps/search/?api=1&query=${s.latitude || 16.3067},${s.longitude || 80.4365}`;
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
                  <span><strong>${item.product_name}</strong> (${item.brand})</span>
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
            <span>${s.shop_name}</span>
            ${statusBadge}
          </div>
          <p><strong>Owner:</strong> ${s.owner_name}</p>
          <p><strong>Phone:</strong> ${s.phone_number}</p>
          <p><strong>Address:</strong> ${s.address}, ${s.district || ''}</p>
          <p><strong>Distance:</strong> 📍 ${s.distance_km !== undefined ? s.distance_km : 2.1} km</p>
          <div style="margin-top: 0.5rem;">${deliveryBadge}</div>
          ${productsHtml}
          <div style="display: flex; gap: 0.5rem; margin-top: 0.75rem;">
            <a href="tel:${s.phone_number}" class="btn btn-secondary btn-sm" style="flex:1;">📞 Call Shop</a>
            <a href="${mapsUrl}" target="_blank" class="btn btn-primary btn-sm" style="flex:1;">🗺️ Get Directions</a>
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
  const query = document.getElementById('search-product-input').value.trim() || 'Urea';
  const container = document.getElementById('search-results-container');
  container.innerHTML = '<p>Searching inventory...</p>';

  try {
    const res = await fetch(`/shops/farmer-search?query=${encodeURIComponent(query)}&latitude=16.3067&longitude=80.4365`);
    if (!res.ok) throw new Error('Search failed');
    const data = await res.json();

    if (!data.results || data.results.length === 0) {
      container.innerHTML = `<p>No shops found selling "${query}".</p>`;
      return;
    }

    container.innerHTML = data.results.map(r => `
      <div class="card">
        <div class="card-title">
          <span>${r.product_name}</span>
          <span class="badge badge-accepted">₹${r.price}</span>
        </div>
        <p><strong>Brand:</strong> ${r.brand}</p>
        <p><strong>Available at:</strong> ${r.shop_name}</p>
        <p><strong>Stock:</strong> ${r.quantity_in_stock} ${r.unit}s</p>
        <p><strong>Distance:</strong> 📍 ${r.distance_km || '2.1'} km</p>
        <p><strong>Phone:</strong> ${r.phone_number}</p>
        <button class="btn btn-accent btn-sm" style="width: 100%; margin-top: 0.75rem;" onclick="openOrderModal('${r.shop_id}', '${r.product_name}', '${r.price}')">
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
  tbody.innerHTML = '<tr><td colspan="6">Loading requests...</td></tr>';

  try {
    const res = await fetch(`/orders/farmer/${activeFarmerId}`);
    if (!res.ok) throw new Error('Failed to load orders');
    const data = await res.json();

    if (data.items.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6">No purchase requests submitted yet.</td></tr>';
      return;
    }

    tbody.innerHTML = data.items.map(o => `
      <tr>
        <td>${o.id.substring(0, 8)}...</td>
        <td><strong>${o.product_name}</strong> (${o.brand || ''})</td>
        <td>${o.quantity} ${o.unit}s</td>
        <td>₹${o.total_price}</td>
        <td><span class="badge badge-${o.status.toLowerCase()}">${o.status}</span></td>
        <td>${new Date(o.created_at).toLocaleDateString()}</td>
      </tr>
    `).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" style="color:red;">Error loading orders</td></tr>`;
  }
}

function sendSampleAiQuery(text) {
  switchFarmerTab('chat');
  document.getElementById('chat-user-input').value = text;
  sendAiChatMessage();
}

async function sendAiChatMessage() {
  const input = document.getElementById('chat-user-input');
  const msg = input.value.trim();
  if (!msg) return;

  const box = document.getElementById('chat-messages-box');
  box.innerHTML += `<div style="background: var(--primary-light); color: var(--primary); padding: 0.75rem; border-radius: 8px; max-width: 80%; margin-left: auto; margin-bottom: 0.75rem;"><strong>You:</strong> ${msg}</div>`;
  input.value = '';

  box.scrollTop = box.scrollHeight;

  try {
    const res = await fetch('/shops/farmer-search?query=' + encodeURIComponent(msg));
    const searchData = await res.json();

    let reply = `🤖 **BhoomiMitra AI Advice:**\nFor your request regarding "${msg}", we recommend checking nearby certified inventory.`;
    if (searchData.results && searchData.results.length > 0) {
      reply += `\n\n🏬 **Nearby Shops Stock:**\n` + searchData.results.map(r => `• ${r.shop_name}: ${r.product_name} (${r.brand}) - ₹${r.price} [Stock: ${r.quantity_in_stock} ${r.unit}s]`).join('\n');
    }

    box.innerHTML += `<div style="background: var(--bg-card); border: 1px solid var(--border-color); padding: 0.75rem; border-radius: 8px; max-width: 80%; margin-bottom: 0.75rem; white-space: pre-line;">${reply}</div>`;
    box.scrollTop = box.scrollHeight;
  } catch (err) {
    box.innerHTML += `<div style="background: #fee2e2; padding: 0.75rem; border-radius: 8px; max-width: 80%; margin-bottom: 0.75rem;">AI service error. Please try again.</div>`;
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
    // Find inventory id
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
    const dashRes = await fetch(`/inventory/dashboard/${activeShopId}`);
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

    const analyticsRes = await fetch(`/orders/analytics/${activeShopId}`);
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
  tbody.innerHTML = '<tr><td colspan="6">Loading incoming orders...</td></tr>';

  try {
    const res = await fetch(`/orders/shop/${activeShopId}`);
    if (!res.ok) throw new Error('Failed to load orders');
    const data = await res.json();

    if (data.items.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6">No incoming farmer requests.</td></tr>';
      return;
    }

    tbody.innerHTML = data.items.map(o => `
      <tr>
        <td>${o.id.substring(0, 8)}...</td>
        <td><strong>${o.product_name}</strong> (${o.brand || ''})</td>
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
  tbody.innerHTML = '<tr><td colspan="8">Loading inventory...</td></tr>';

  try {
    const res = await fetch(`/inventory/shop/${activeShopId}`);
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
          <td><strong>${i.product_name}</strong></td>
          <td>${i.brand}</td>
          <td>${i.category}</td>
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
    const res = await fetch(`/inventory/${itemId}`, { method: 'DELETE' });
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
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      alert("✅ Shop Profile updated successfully!");
    }
  } catch (err) {
    alert("Error updating profile");
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
          <span><strong>${p.product_name}</strong></span>
          <span class="badge badge-accepted">${p.units_sold} units demanded</span>
        </div>
      `).join('');
    } else {
      popularBox.innerHTML = '<p>No sales analytics data yet.</p>';
    }

    // Chart.js rendering
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
