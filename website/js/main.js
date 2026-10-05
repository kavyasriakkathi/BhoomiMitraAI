/**
 * BhoomiMitra AI — Public Website Interaction Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Hydrate dynamic fields from siteConfig if present
  applySiteConfiguration();

  // 2. Mobile Menu Toggle
  setupMobileMenu();

  // 3. Smooth Anchor Scrolling & Navbar Blur
  setupNavigation();

  // 4. Interactive Voice Simulator Demo for Farmers
  setupVoiceSimulator();
});

/**
 * Injects values from window.siteConfig into DOM elements with data-config attributes.
 */
function applySiteConfiguration() {
  const config = window.siteConfig;
  if (!config) return;

  // Hydrate WhatsApp Links
  document.querySelectorAll('[data-config="whatsapp-link"]').forEach(el => {
    el.setAttribute('href', config.urls.whatsapp);
    el.setAttribute('target', '_blank');
    el.setAttribute('rel', 'noopener noreferrer');
  });

  // Hydrate Email Links
  document.querySelectorAll('[data-config="contact-email"]').forEach(el => {
    el.setAttribute('href', `mailto:${config.urls.email}`);
    if (el.dataset.showText === "true") {
      el.textContent = config.urls.email;
    }
  });

  document.querySelectorAll('[data-config="partners-email"]').forEach(el => {
    el.setAttribute('href', `mailto:${config.urls.partnershipsEmail}`);
    if (el.dataset.showText === "true") {
      el.textContent = config.urls.partnershipsEmail;
    }
  });

  // Hydrate Socials
  document.querySelectorAll('[data-config="linkedin-link"]').forEach(el => {
    el.setAttribute('href', config.urls.linkedin);
    el.setAttribute('target', '_blank');
    el.setAttribute('rel', 'noopener noreferrer');
  });

  document.querySelectorAll('[data-config="github-link"]').forEach(el => {
    el.setAttribute('href', config.urls.github);
    el.setAttribute('target', '_blank');
    el.setAttribute('rel', 'noopener noreferrer');
  });

  // Hydrate Pilot Location
  document.querySelectorAll('[data-config="pilot-location"]').forEach(el => {
    el.textContent = config.pilot.location;
  });

  // Hydrate Current Year in Footer
  const yearEl = document.getElementById('current-year');
  if (yearEl) {
    yearEl.textContent = new Date().getFullYear();
  }
}

/**
 * Mobile navigation drawer logic
 */
function setupMobileMenu() {
  const toggleBtn = document.getElementById('mobile-toggle');
  const mobileNav = document.getElementById('mobile-nav');

  if (!toggleBtn || !mobileNav) return;

  toggleBtn.addEventListener('click', () => {
    const isOpen = mobileNav.classList.toggle('active');
    toggleBtn.setAttribute('aria-expanded', isOpen);
    
    // Toggle icon lines
    const spans = toggleBtn.querySelectorAll('span');
    if (isOpen) {
      spans[0].style.transform = 'translateY(9px) rotate(45deg)';
      spans[1].style.opacity = '0';
      spans[2].style.transform = 'translateY(-9px) rotate(-45deg)';
    } else {
      spans[0].style.transform = 'none';
      spans[1].style.opacity = '1';
      spans[2].style.transform = 'none';
    }
  });

  // Close menu when clicking link
  mobileNav.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      mobileNav.classList.remove('active');
      const spans = toggleBtn.querySelectorAll('span');
      spans[0].style.transform = 'none';
      spans[1].style.opacity = '1';
      spans[2].style.transform = 'none';
    });
  });
}

/**
 * Navigation scroll behavior
 */
function setupNavigation() {
  const navbar = document.querySelector('.navbar');

  window.addEventListener('scroll', () => {
    if (window.scrollY > 20) {
      navbar.style.boxShadow = '0 4px 20px rgba(15, 56, 44, 0.08)';
    } else {
      navbar.style.boxShadow = 'none';
    }
  });
}

/**
 * Interactive voice preview simulation
 */
function setupVoiceSimulator() {
  const playButtons = document.querySelectorAll('.voice-play-btn');

  playButtons.forEach(btn => {
    btn.addEventListener('click', function () {
      const isPlaying = this.classList.contains('playing');
      
      // Stop all
      playButtons.forEach(b => {
        b.classList.remove('playing');
        b.innerHTML = '▶ Play Audio';
      });

      if (!isPlaying) {
        this.classList.add('playing');
        this.innerHTML = '⏸ Playing...';

        // Simulate voice waveform animation for 3.5 seconds
        setTimeout(() => {
          this.classList.remove('playing');
          this.innerHTML = '▶ Play Audio';
        }, 3500);
      }
    });
  });
}
