/**
 * QuickBasket Zepto-Style UI Logic
 * Full reactivity:
 *  - Category and Item dropdowns with auto-enable
 *  - Instant Top-5 per cart item retrieval
 *  - Multi-item pooling (1 item = 5 candidates, 2 items = 10 candidates, etc.)
 *  - Transparent Before vs. After re-ranking verification
 *  - Bulletproof click handling on cards & buttons
 */

document.addEventListener("DOMContentLoaded", () => {
  // Application State
  const state = {
    cart: [], // Array of item_name strings
    cartMeta: new Map(), // name -> item metadata
    allItemsMap: new Map(), // name -> full item metadata
    categories: [],
    currentCategory: "",
    categoryItems: [],
    selectedDropdownItem: null,
    currentModel: "model_1_cross_mlp_v3",
    retrievalMode: "hybrid_semantic", // "hybrid_semantic" (default SOTA) | "cross_category" | "same_category"
    isCrossCategory: true,
    isLoadingRecommendation: false
  };

  // DOM Elements
  const headerCartCount = document.getElementById("headerCartCount");
  const headerCartTotal = document.getElementById("headerCartTotal");
  const cartCountBadge = document.getElementById("cartCountBadge");
  const cartItemsRow = document.getElementById("cartItemsRow");
  const clearCartBtn = document.getElementById("clearCartBtn");
  const modelSelect = document.getElementById("modelSelect");

  const btnModeHybrid = document.getElementById("btnModeHybrid");
  const btnModeCross = document.getElementById("btnModeCross");
  const btnModeSame = document.getElementById("btnModeSame");
  const headerEngineBadgeText = document.getElementById("headerEngineBadgeText");
  const retrievalStagePill = document.getElementById("retrievalStagePill");
  const retrievalStatusStrip = document.getElementById("retrievalStatusStrip");
  const stripBadge = document.getElementById("stripBadge");
  const stripText = document.getElementById("stripText");
  const stripMetric = document.getElementById("stripMetric");

  const previewGrid = document.getElementById("previewGrid");
  const categoryDropdown = document.getElementById("categoryDropdown");
  const itemDropdown = document.getElementById("itemDropdown");
  const addSelectedDropdownBtn = document.getElementById("addSelectedDropdownBtn");
  const categoryShelfTitle = document.getElementById("categoryShelfTitle");
  const categoryItemCount = document.getElementById("categoryItemCount");
  const categoryCardsGrid = document.getElementById("categoryCardsGrid");

  const globalSearchInput = document.getElementById("globalSearchInput");
  const searchDropdown = document.getElementById("searchDropdown");

  const rankingPlaceholder = document.getElementById("rankingPlaceholder");
  const rankingActiveContainer = document.getElementById("rankingActiveContainer");
  const verificationBanner = document.getElementById("verificationBanner");
  const vBannerIcon = document.getElementById("vBannerIcon");
  const vBannerTitle = document.getElementById("vBannerTitle");
  const vBannerDesc = document.getElementById("vBannerDesc");
  const vShiftBadge = document.getElementById("vShiftBadge");
  const vLatency = document.getElementById("vLatency");

  const beforeCandidatesStack = document.getElementById("beforeCandidatesStack");
  const afterCandidatesStack = document.getElementById("afterCandidatesStack");
  const beforeItemCount = document.getElementById("beforeItemCount");
  const modelNameLabel = document.getElementById("modelNameLabel");
  const shiftTableBody = document.getElementById("shiftTableBody");

  // ----------------------------------------------------------------------------
  // 1. Initial Data Fetching
  // ----------------------------------------------------------------------------
  async function initializeApp() {
    try {
      await Promise.all([
        loadPreviewItems(),
        loadCategories()
      ]);
    } catch (err) {
      console.error("Initialization error:", err);
    }
  }

  // Load preview items
  async function loadPreviewItems() {
    try {
      const res = await fetch("/api/preview_items");
      const data = await res.json();
      const items = data.preview_items || [];
      items.forEach(itm => state.allItemsMap.set(itm.item_name, itm));
      renderProductCards(previewGrid, items);
    } catch (err) {
      previewGrid.innerHTML = `<p class="loading-state" style="color: #f43f5e;">Failed to load preview items.</p>`;
    }
  }

  // Load categories
  async function loadCategories() {
    try {
      const res = await fetch("/api/categories");
      const data = await res.json();
      state.categories = data.categories || [];

      categoryDropdown.innerHTML = `<option value="">-- Choose a Category --</option>` +
        state.categories.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");

      if (state.categories.length > 0) {
        // Auto-select first category
        categoryDropdown.value = state.categories[0];
        handleCategoryChange(state.categories[0]);
      }
    } catch (err) {
      categoryDropdown.innerHTML = `<option value="">Error loading categories</option>`;
    }
  }

  // ----------------------------------------------------------------------------
  // 2. Category & Item Dropdown Handler
  // ----------------------------------------------------------------------------
  async function handleCategoryChange(category) {
    state.currentCategory = category;
    if (!category) {
      itemDropdown.innerHTML = `<option value="">Select a category first</option>`;
      itemDropdown.disabled = true;
      addSelectedDropdownBtn.disabled = true;
      categoryCardsGrid.innerHTML = `<p class="empty-hint">Select a category above to view items</p>`;
      return;
    }

    itemDropdown.disabled = true;
    itemDropdown.innerHTML = `<option value="">Loading items...</option>`;
    categoryShelfTitle.textContent = `${category}`;
    categoryCardsGrid.innerHTML = `<div class="loading-state">Loading items for ${category}...</div>`;

    try {
      const res = await fetch(`/api/items_by_category?category=${encodeURIComponent(category)}&limit=60`);
      const data = await res.json();
      state.categoryItems = data.items || [];
      state.categoryItems.forEach(itm => state.allItemsMap.set(itm.item_name, itm));

      categoryItemCount.textContent = `${state.categoryItems.length} items available`;

      // Populate Item Dropdown
      if (state.categoryItems.length > 0) {
        itemDropdown.innerHTML = state.categoryItems.map(itm => `
          <option value="${escapeHtml(itm.item_name)}">
            ${escapeHtml(itm.item_name)} (₹${Math.round(itm.price_inr)})
          </option>
        `).join("");

        itemDropdown.disabled = false;
        // Auto-select first item and enable button!
        itemDropdown.value = state.categoryItems[0].item_name;
        state.selectedDropdownItem = state.categoryItems[0];
        addSelectedDropdownBtn.disabled = false;
      } else {
        itemDropdown.innerHTML = `<option value="">No items found</option>`;
        itemDropdown.disabled = true;
        addSelectedDropdownBtn.disabled = true;
      }

      // Render cards grid
      renderProductCards(categoryCardsGrid, state.categoryItems);
    } catch (err) {
      itemDropdown.innerHTML = `<option value="">Error loading items</option>`;
      categoryCardsGrid.innerHTML = `<p class="loading-state" style="color: #f43f5e;">Failed to load items.</p>`;
    }
  }

  categoryDropdown.addEventListener("change", (e) => {
    handleCategoryChange(e.target.value);
  });

  itemDropdown.addEventListener("change", (e) => {
    const selectedName = e.target.value;
    const item = state.categoryItems.find(i => i.item_name === selectedName);
    state.selectedDropdownItem = item || null;
    addSelectedDropdownBtn.disabled = !item;
  });

  addSelectedDropdownBtn.addEventListener("click", () => {
    if (state.selectedDropdownItem) {
      addToCart(state.selectedDropdownItem);
    }
  });

  // ----------------------------------------------------------------------------
  // 3. Render Product Cards (Zepto Style)
  // ----------------------------------------------------------------------------
  function renderProductCards(container, items) {
    if (!items || items.length === 0) {
      container.innerHTML = `<p class="empty-hint">No items found.</p>`;
      return;
    }

    container.innerHTML = items.map(itm => {
      const inCart = state.cart.includes(itm.item_name);
      return `
        <div class="product-card" data-name="${encodeURIComponent(itm.item_name)}" style="cursor: pointer;">
          <div class="card-badge-row">
            <div class="veg-icon" title="Vegetarian"><div class="veg-dot"></div></div>
            <span class="delivery-time-tag">⚡ 8 MINS</span>
          </div>
          <div class="card-img-box">${itm.icon || '📦'}</div>
          ${itm.semantic_id ? `<span class="card-sem-id" title="RQ-VAE Semantic ID">🧬 ${escapeHtml(itm.semantic_id)}</span>` : ''}
          <span class="card-pack">${escapeHtml(itm.pack_size || itm.category)}</span>
          <h4 class="card-title" title="${escapeHtml(itm.item_name)}">${escapeHtml(itm.item_name)}</h4>
          <div class="card-price-action">
            <div class="price-wrap">
              <span class="price-now">₹${Math.round(itm.price_inr)}</span>
            </div>
            <button class="btn-add-zepto ${inCart ? 'in-cart' : ''}" data-name="${encodeURIComponent(itm.item_name)}">
              ${inCart ? '✓ IN CART' : '+ ADD'}
            </button>
          </div>
        </div>
      `;
    }).join("");

    // Click handler on entire card AND button
    container.querySelectorAll(".product-card").forEach(card => {
      card.addEventListener("click", (e) => {
        const rawName = decodeURIComponent(card.dataset.name);
        const itm = state.allItemsMap.get(rawName);
        if (itm) {
          addToCart(itm);
        }
      });
    });
  }

  // ----------------------------------------------------------------------------
  // 4. Cart Management & Automatic Pipeline Trigger
  // ----------------------------------------------------------------------------
  function addToCart(item) {
    if (!state.cart.includes(item.item_name)) {
      state.cart.push(item.item_name);
      state.cartMeta.set(item.item_name, item);
      state.allItemsMap.set(item.item_name, item);
    }
    updateCartDisplay();
    refreshCardButtons();
    // AUTOMATICALLY TRIGGER RETRIEVAL & RE-RANKING!
    triggerRetrievalAndReranking();
  }

  function removeFromCart(itemName) {
    state.cart = state.cart.filter(n => n !== itemName);
    state.cartMeta.delete(itemName);
    updateCartDisplay();
    refreshCardButtons();

    if (state.cart.length > 0) {
      triggerRetrievalAndReranking();
    } else {
      resetToPlaceholder();
    }
  }

  function clearCart() {
    state.cart = [];
    state.cartMeta.clear();
    updateCartDisplay();
    refreshCardButtons();
    resetToPlaceholder();
  }

  clearCartBtn.addEventListener("click", clearCart);

  function updateCartDisplay() {
    const count = state.cart.length;
    headerCartCount.textContent = `${count} ${count === 1 ? 'Item' : 'Items'}`;
    cartCountBadge.textContent = count;

    let totalPrice = 0;
    state.cart.forEach(n => {
      const meta = state.cartMeta.get(n);
      totalPrice += meta ? meta.price_inr : 49;
    });
    headerCartTotal.textContent = `₹${Math.round(totalPrice)}`;

    if (count === 0) {
      cartItemsRow.innerHTML = `<p class="empty-cart-note">Your cart is empty. Add an item from preview or dropdown below to start!</p>`;
      clearCartBtn.style.display = "none";
      return;
    }

    clearCartBtn.style.display = "block";
    cartItemsRow.innerHTML = state.cart.map(name => {
      const meta = state.cartMeta.get(name) || { icon: '📦', item_name: name };
      return `
        <div class="cart-item-tag">
          <span>${meta.icon || '📦'}</span>
          <span>${escapeHtml(meta.item_name)}</span>
          ${meta.semantic_id ? `<span class="cart-tag-sem-id" title="Semantic ID">${escapeHtml(meta.semantic_id)}</span>` : ''}
          <button class="cart-tag-remove" data-name="${encodeURIComponent(name)}" title="Remove from cart">✕</button>
        </div>
      `;
    }).join("");

    cartItemsRow.querySelectorAll(".cart-tag-remove").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const rawName = decodeURIComponent(btn.dataset.name);
        removeFromCart(rawName);
      });
    });
  }

  function refreshCardButtons() {
    document.querySelectorAll(".product-card").forEach(card => {
      const rawName = decodeURIComponent(card.dataset.name);
      const inCart = state.cart.includes(rawName);
      const btn = card.querySelector(".btn-add-zepto");
      if (btn) {
        if (inCart) {
          btn.classList.add("in-cart");
          btn.textContent = "✓ IN CART";
        } else {
          btn.classList.remove("in-cart");
          btn.textContent = "+ ADD";
        }
      }
      if (inCart) {
        card.style.borderColor = "var(--zepto-green)";
        card.style.background = "rgba(16, 185, 129, 0.08)";
      } else {
        card.style.borderColor = "var(--border-subtle)";
        card.style.background = "var(--zepto-surface)";
      }
    });
  }

  function resetToPlaceholder() {
    rankingActiveContainer.classList.add("hidden");
    rankingPlaceholder.classList.remove("hidden");
  }

  // ----------------------------------------------------------------------------
  // 5. Multi-Item Retrieval & Neural Re-Ranking Pipeline
  // ----------------------------------------------------------------------------
  async function triggerRetrievalAndReranking() {
    if (state.cart.length === 0 || state.isLoadingRecommendation) return;

    state.isLoadingRecommendation = true;
    rankingPlaceholder.classList.add("hidden");
    rankingActiveContainer.classList.remove("hidden");

    vBannerTitle.textContent = "Running Retrieval & Neural Re-Ranking...";
    vBannerDesc.textContent = `Retrieving Top 5 candidates for each of the ${state.cart.length} items in your cart...`;
    vBannerIcon.textContent = "⏳";
    vShiftBadge.textContent = "Computing...";

    try {
      const payload = {
        cart_items: state.cart,
        model_name: modelSelect.value,
        top_k_per_item: 5,
        cross_category: state.isCrossCategory,
        retrieval_mode: state.retrievalMode
      };

      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Re-ranking failed");
      }

      const data = await res.json();
      renderVerificationResults(data);

    } catch (err) {
      vBannerTitle.textContent = "Error executing Re-Ranking";
      vBannerDesc.textContent = err.message;
      vBannerIcon.textContent = "⚠️";
      console.error("Re-rank error:", err);
    } finally {
      state.isLoadingRecommendation = false;
    }
  }

  // Retrieval Mode Switcher Handler
  function setRetrievalMode(mode) {
    state.retrievalMode = mode;
    state.isCrossCategory = (mode !== "same_category");

    if (btnModeHybrid) btnModeHybrid.classList.toggle("active", mode === "hybrid_semantic");
    if (btnModeCross) btnModeCross.classList.toggle("active", mode === "cross_category");
    if (btnModeSame) btnModeSame.classList.toggle("active", mode === "same_category");

    if (headerEngineBadgeText) {
      if (mode === "hybrid_semantic") {
        headerEngineBadgeText.textContent = "Retrieval: Hybrid Semantic ID (RQ-VAE + Two-Tower + Graph)";
      } else if (mode === "cross_category") {
        headerEngineBadgeText.textContent = "Retrieval: Two-Tower v5 (Cross-Category Dense)";
      } else {
        headerEngineBadgeText.textContent = "Retrieval: Two-Tower v5 (Same-Category Substitutes)";
      }
    }

    if (state.cart.length > 0) {
      triggerRetrievalAndReranking();
    }
  }

  if (btnModeHybrid) {
    btnModeHybrid.addEventListener("click", () => setRetrievalMode("hybrid_semantic"));
  }
  if (btnModeCross) {
    btnModeCross.addEventListener("click", () => setRetrievalMode("cross_category"));
  }
  if (btnModeSame) {
    btnModeSame.addEventListener("click", () => setRetrievalMode("same_category"));
  }

  modelSelect.addEventListener("change", () => {
    state.currentModel = modelSelect.value;
    if (state.cart.length > 0) {
      triggerRetrievalAndReranking();
    }
  });

  // ----------------------------------------------------------------------------
  // 6. Render Side-by-Side Comparison & Verification Table
  // ----------------------------------------------------------------------------
  function renderVerificationResults(data) {
    const {
      cart_count,
      total_candidates,
      re_ranking_occurred,
      shift_count,
      headline,
      elapsed_ms,
      model_used,
      before_ranking,
      after_ranking
    } = data;

    // Update Model Name Label
    let readableModel = "Model 1 (v3): Interaction Cross-MLP";
    if (model_used === "model_1_cross_mlp_v3") readableModel = "Model 1 (v3): Interaction Cross-MLP";
    if (model_used === "model_3_transformer_v3") readableModel = "Model 3 (v3): List-Aware Transformer";
    if (model_used === "model_2_dual_tower_v3") readableModel = "Model 2 (v3): Dual-Tower Bilinear";
    if (model_used === "model_1_cross_mlp") readableModel = "Model 1 (Legacy): Cross-MLP";
    if (model_used === "model_3_transformer") readableModel = "Model 3 (Legacy): List-Aware Transformer";
    if (model_used === "model_2_dual_tower") readableModel = "Model 2 (Legacy): Dual-Tower Bilinear";
    modelNameLabel.textContent = readableModel;
    beforeItemCount.textContent = total_candidates;

    const retrievalModelLabel = document.getElementById("retrievalModelLabel");
    if (retrievalModelLabel && data.retrieval_model) {
      retrievalModelLabel.textContent = data.retrieval_model;
    }

    if (retrievalStagePill) {
      if (data.retrieval_mode === "hybrid_semantic") {
        retrievalStagePill.textContent = "Stage 1: Hybrid Semantic Retrieval (RRF)";
      } else {
        retrievalStagePill.textContent = "Stage 1: Two-Tower Retrieval (v5 1:1)";
      }
    }

    // 1. Update Verification Status Banner
    if (re_ranking_occurred) {
      verificationBanner.className = "verification-banner";
      vBannerIcon.textContent = "🚀";
      vBannerTitle.textContent = `VERIFICATION: Re-Ranking Active & Order Changed!`;
      vBannerDesc.textContent = `${shift_count} of ${total_candidates} items changed rank positions. ${headline}`;
      vShiftBadge.textContent = `${shift_count} Items Shifted`;
      vShiftBadge.style.background = "#f59e0b";
      vShiftBadge.style.color = "#000";
    } else {
      verificationBanner.className = "verification-banner banner-no-shift";
      vBannerIcon.textContent = "🎯";
      vBannerTitle.textContent = `VERIFICATION: Re-Ranking Active — Order Confirmed Optimal`;
      vBannerDesc.textContent = `All ${total_candidates} candidates were evaluated. Model confirmed initial order as optimal.`;
      vShiftBadge.textContent = `Order Maintained`;
      vShiftBadge.style.background = "#10b981";
      vShiftBadge.style.color = "#fff";
    }

    vLatency.textContent = `Latency: ${elapsed_ms} ms`;

    // 1.5 Update Retrieval Status Strip
    if (retrievalStatusStrip && stripBadge && stripText && stripMetric) {
      if (data.retrieval_mode === "hybrid_semantic") {
        retrievalStatusStrip.className = "retrieval-status-strip strip-hybrid";
        stripBadge.className = "strip-badge badge-hybrid";
        stripBadge.innerHTML = "🧬 Hybrid Semantic ID Active (RRF)";
        stripText.textContent = "Fusing RQ-VAE Semantic IDs (40%), Two-Tower v5 (35%), and Co-Purchase Graph (25%). Base-code substitutes strictly suppressed (-inf).";
        stripMetric.textContent = "0.00% Substitute Leakage • 93.3% SOTA Hit@10";
      } else if (data.cross_category_enabled) {
        retrievalStatusStrip.className = "retrieval-status-strip";
        stripBadge.className = "strip-badge badge-cross";
        stripBadge.innerHTML = "✨ Cross-Category Complements Active";
        const catList = (data.cart_categories && data.cart_categories.length > 0) ? data.cart_categories.join(", ") : "Cart Aisles";
        stripText.textContent = `Strictly excluding cart categories (${catList}) — candidates retrieved from complementary aisles.`;
        stripMetric.textContent = `${data.cross_category_count || total_candidates} / ${total_candidates} Items Cross-Category (100%)`;
      } else {
        retrievalStatusStrip.className = "retrieval-status-strip strip-same";
        stripBadge.className = "strip-badge badge-same";
        stripBadge.innerHTML = "🔄 Same-Category Substitutes Active";
        stripText.textContent = "Direct similarity mode — candidates retrieved as nearest substitutes/variants within same category.";
        stripMetric.textContent = "Same-Category / Direct Mode";
      }
    }

    // 2. Render Stage 1: Before Re-Ranking
    beforeCandidatesStack.innerHTML = before_ranking.map(item => `
      <div class="candidate-card-row">
        <div class="cand-left">
          <div class="rank-badge ${item.initial_rank === 1 ? 'top-1' : (item.initial_rank === 2 ? 'top-2' : '')}">
            #${item.initial_rank}
          </div>
          <span style="font-size: 1.4rem;">${item.icon || '📦'}</span>
          <div class="cand-info">
            <span class="cand-title" title="${escapeHtml(item.item_name)}">${escapeHtml(item.item_name)}</span>
            <span class="cand-meta">
              <span class="cand-source-tag">For: ${escapeHtml(item.retrieved_by)} (${escapeHtml(item.retrieved_by_category)})</span> • 
              ${item.semantic_id ? `<span class="cand-semantic-id" title="RQ-VAE Semantic ID">🧬 ${escapeHtml(item.semantic_id)}</span> • ` : ''}
              ${item.tier && item.tier !== "None" ? `<span class="cand-tier-badge tier-${item.tier.toLowerCase()}">⭐ ${escapeHtml(item.tier.toUpperCase())}</span> • ` : ''}
              <span class="cand-cat-pill ${item.is_cross_category ? 'cat-pill-cross' : 'cat-pill-same'}">
                ${item.is_cross_category ? '✨ ' : '🔄 '}${escapeHtml(item.category)}
              </span>
              <span class="badge-cand-type ${item.is_cross_category ? 'badge-cross-comp' : 'badge-same-sub'}">
                ${item.is_cross_category ? 'Cross-Cat' : 'Same-Cat'}
              </span> • 
              ₹${Math.round(item.price_inr)}
            </span>
          </div>
        </div>
        <div class="cand-right">
          <span class="score-badge" style="color: #60a5fa;" title="Two-Tower Complementary Score">
            Score: ${item.similarity_score.toFixed(4)}
          </span>
        </div>
      </div>
    `).join("");

    // 3. Render Stage 2: After Re-Ranking
    afterCandidatesStack.innerHTML = after_ranking.map(item => {
      let rowClass = "";
      let deltaBadge = "";

      if (item.status_type === "promoted") {
        rowClass = "is-promoted";
        deltaBadge = `<span class="delta-badge badge-promoted">▲ +${item.rank_delta} (was #${item.initial_rank})</span>`;
      } else if (item.status_type === "demoted") {
        rowClass = "is-demoted";
        deltaBadge = `<span class="delta-badge badge-demoted">▼ ${item.rank_delta} (was #${item.initial_rank})</span>`;
      } else {
        deltaBadge = `<span class="delta-badge badge-neutral">— Unchanged (#${item.initial_rank})</span>`;
      }

      return `
        <div class="candidate-card-row ${rowClass}">
          <div class="cand-left">
            <div class="rank-badge ${item.rerank === 1 ? 'top-1' : (item.rerank === 2 ? 'top-2' : '')}">
              #${item.rerank}
            </div>
            <span style="font-size: 1.4rem;">${item.icon || '📦'}</span>
            <div class="cand-info">
              <span class="cand-title" title="${escapeHtml(item.item_name)}">${escapeHtml(item.item_name)}</span>
              <span class="cand-meta">
                <span class="cand-source-tag">For: ${escapeHtml(item.retrieved_by)} (${escapeHtml(item.retrieved_by_category)})</span> • 
                ${item.semantic_id ? `<span class="cand-semantic-id" title="RQ-VAE Semantic ID">🧬 ${escapeHtml(item.semantic_id)}</span> • ` : ''}
                ${item.tier && item.tier !== "None" ? `<span class="cand-tier-badge tier-${item.tier.toLowerCase()}">⭐ ${escapeHtml(item.tier.toUpperCase())}</span> • ` : ''}
                <span class="cand-cat-pill ${item.is_cross_category ? 'cat-pill-cross' : 'cat-pill-same'}">
                  ${item.is_cross_category ? '✨ ' : '🔄 '}${escapeHtml(item.category)}
                </span>
                <span class="badge-cand-type ${item.is_cross_category ? 'badge-cross-comp' : 'badge-same-sub'}">
                  ${item.is_cross_category ? 'Cross-Cat' : 'Same-Cat'}
                </span> • 
                ₹${Math.round(item.price_inr)}
              </span>
            </div>
          </div>
          <div class="cand-right">
            ${deltaBadge}
            <span class="score-badge" style="color: #c084fc;" title="Model Logit Score">
              Score: ${item.model_score > 0 ? '+' : ''}${item.model_score.toFixed(2)}
            </span>
          </div>
        </div>
      `;
    }).join("");

    // 4. Render Item-by-Item Shift Table
    shiftTableBody.innerHTML = after_ranking.map(item => {
      const isDiff = item.rank_delta !== 0;
      let statusHtml = "";
      if (item.rank_delta > 0) {
        statusHtml = `<span style="color: #10b981; font-weight: 700;">▲ Promoted (+${item.rank_delta})</span>`;
      } else if (item.rank_delta < 0) {
        statusHtml = `<span style="color: #f43f5e; font-weight: 700;">▼ Demoted (${item.rank_delta})</span>`;
      } else {
        statusHtml = `<span style="color: #94a3b8;">— Unchanged</span>`;
      }

      return `
        <tr>
          <td>
            <div style="display: flex; align-items: center; gap: 8px;">
              <span>${item.icon || '📦'}</span>
              <div>
                <strong style="color: #fff;">${escapeHtml(item.item_name)}</strong>
                ${item.semantic_id ? `<span class="table-sem-id" title="RQ-VAE Semantic ID">🧬 ${escapeHtml(item.semantic_id)}</span>` : ''}<br>
                <span class="cand-cat-pill ${item.is_cross_category ? 'cat-pill-cross' : 'cat-pill-same'}" style="font-size: 0.65rem;">
                  ${item.is_cross_category ? '✨ ' : '🔄 '}${escapeHtml(item.category)}
                </span>
                ${item.tier && item.tier !== "None" ? `<span class="cand-tier-badge tier-${item.tier.toLowerCase()}" style="font-size: 0.6rem;">⭐ ${escapeHtml(item.tier.toUpperCase())}</span>` : ''}
              </div>
            </div>
          </td>
          <td>
            <span style="color: #c084fc; font-weight: 600;">${escapeHtml(item.retrieved_by)}</span><br>
            <span style="font-size: 0.68rem; color: #94a3b8;">(${escapeHtml(item.retrieved_by_category)})</span>
          </td>
          <td><span class="score-badge">#${item.initial_rank}</span></td>
          <td><span class="score-badge" style="background: rgba(139, 61, 255, 0.2); color: #c084fc; font-weight: 800;">#${item.rerank}</span></td>
          <td>${statusHtml}</td>
          <td><span style="color: #60a5fa;">${item.similarity_score.toFixed(4)}</span></td>
          <td><strong style="color: #fff;">${item.model_score > 0 ? '+' : ''}${item.model_score.toFixed(2)}</strong></td>
          <td>
            ${isDiff 
              ? `<span style="background: rgba(245, 158, 11, 0.2); color: #f59e0b; padding: 2px 8px; border-radius: 4px; font-weight: 800;">YES (${Math.abs(item.rank_delta)} pos)</span>`
              : `<span style="color: #64748b; font-size: 0.75rem;">NO</span>`
            }
          </td>
        </tr>
      `;
    }).join("");

    // Scroll into view
    rankingActiveContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ----------------------------------------------------------------------------
  // 7. Global Search Dropdown
  // ----------------------------------------------------------------------------
  let searchTimer = null;
  globalSearchInput.addEventListener("input", () => {
    const q = globalSearchInput.value.trim();
    if (q.length < 2) {
      searchDropdown.classList.add("hidden");
      return;
    }

    clearTimeout(searchTimer);
    searchTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=12`);
        const data = await res.json();
        const results = data.results || [];

        if (results.length === 0) {
          searchDropdown.innerHTML = `<div class="search-row"><span style="color: #64748b;">No products found</span></div>`;
        } else {
          searchDropdown.innerHTML = results.map(itm => `
            <div class="search-row" data-name="${encodeURIComponent(itm.item_name)}">
              <div class="search-row-left">
                <span style="font-size: 1.3rem;">${itm.icon || '📦'}</span>
                <div>
                  <div class="search-row-name">${escapeHtml(itm.item_name)}</div>
                  <div class="search-row-cat">${escapeHtml(itm.category)} • ₹${Math.round(itm.price_inr)}</div>
                </div>
              </div>
              <button class="btn-add-zepto" style="padding: 4px 10px; font-size: 0.75rem;">+ ADD</button>
            </div>
          `).join("");

          searchDropdown.querySelectorAll(".search-row").forEach(el => {
            el.addEventListener("click", () => {
              const rawName = decodeURIComponent(el.dataset.name);
              const itm = state.allItemsMap.get(rawName);
              if (itm) {
                addToCart(itm);
              }
              searchDropdown.classList.add("hidden");
              globalSearchInput.value = "";
            });
          });
        }
        searchDropdown.classList.remove("hidden");
      } catch (err) {
        console.error(err);
      }
    }, 200);
  });

  document.addEventListener("click", (e) => {
    if (!globalSearchInput.contains(e.target) && !searchDropdown.contains(e.target)) {
      searchDropdown.classList.add("hidden");
    }
  });

  // Utility
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Start app
  initializeApp();
});
