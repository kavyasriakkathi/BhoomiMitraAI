/**
 * BhoomiMitra AI — Public Site Configuration
 * 
 * Central configuration file for all startup metadata, contact endpoints,
 * social links, pilot location, and configuration flags.
 * Update values here to update across the entire website.
 */

const siteConfig = {
  // Brand Information
  name: "BhoomiMitra AI",
  tagline: "From Farmer’s Voice to Smarter Decisions",
  positioning: "A WhatsApp-first, voice-first agricultural AI assistant that helps farmers ask farming questions in their own language, understand crop problems through images, receive agricultural guidance, and discover verified agricultural shops and stock.",
  foundedYear: 2026,
  status: "Early-stage Agricultural AI Initiative",

  // URLs & Direct Actions
  urls: {
    // Live Production WhatsApp Assistant (Verified from project data-deletion & compliance)
    whatsapp: "https://wa.me/917794852530?text=Namaste%20BhoomiMitra",
    whatsappDisplay: "+91 77948 52530",
    
    // Website URLs (Canonical domain target; public URL unassigned pending confirmation)
    website: null,
    canonical: "https://bhoomimitra.ai",

    // Social & Professional Profiles
    linkedin: "https://www.linkedin.com/company/bhoomimitra-ai",
    github: "https://github.com/kavyasriakkathi/BhoomiMitraAI",
    email: "contact@bhoomimitra.ai",
    partnershipsEmail: "partners@bhoomimitra.ai",
  },

  // Active Pilot Information
  pilot: {
    location: "Korutla, Telangana",
    region: "Jagtial District",
    stage: "Controlled Real-World Pilot",
    description: "We are validating BhoomiMitra through controlled real-world testing with farmers and verified agricultural shops.",
    // Quantitative stats are left unverified/configurable. Set showStats to true only when officially audited.
    showStats: false,
    stats: {
      activeFarmers: null, // e.g. "500+" when verified
      verifiedShops: null,  // e.g. "15+" when verified
      queriesResolved: null // e.g. "2,500+" when verified
    }
  },

  // Navigation Links
  navigation: [
    { label: "Home", href: "#hero" },
    { label: "Capabilities", href: "#highlights" },
    { label: "How It Works", href: "#how-it-works" },
    { label: "Examples", href: "#examples" },
    { label: "Why WhatsApp", href: "#why-whatsapp" },
    { label: "Intelligence", href: "#intelligence" },
    { label: "Safety & Trust", href: "#safety" },
    { label: "Pilot", href: "#pilot" },
    { label: "About", href: "#about" },
    { label: "Contact", href: "#contact" }
  ],

  // Footer Navigation
  footerLinks: [
    { label: "Home", href: "#hero" },
    { label: "How It Works", href: "#how-it-works" },
    { label: "Technology", href: "#intelligence" },
    { label: "Safety & Trust", href: "#safety" },
    { label: "Pilot", href: "#pilot" },
    { label: "About", href: "#about" },
    { label: "Contact", href: "#contact" },
    { label: "Privacy Policy", href: "privacy.html" },
    { label: "Terms of Service", href: "terms.html" }
  ],

  // Metadata for SEO
  meta: {
    title: "BhoomiMitra AI | From Farmer’s Voice to Smarter Decisions",
    description: "BhoomiMitra AI is a WhatsApp-first agricultural AI assistant helping farmers ask questions through voice, text and crop photos in their own language.",
    keywords: "BhoomiMitra AI, agricultural AI, WhatsApp farming assistant, crop advisory, crop disease identification, agri shop discovery, Telangana farmers, multilingual agri tech",
    author: "BhoomiMitra AI Team",
    ogImage: "/assets/og-image.png"
  }
};

// Also export to global window object for vanilla script usage without bundlers
if (typeof window !== "undefined") {
  window.siteConfig = siteConfig;
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = { siteConfig };
}
