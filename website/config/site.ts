/**
 * BhoomiMitra AI — Public Site Configuration (TypeScript definition)
 */

export interface SiteConfig {
  name: string;
  tagline: string;
  positioning: string;
  foundedYear: number;
  status: string;
  urls: {
    whatsapp: string;
    whatsappDisplay: string;
    website: string | null;
    canonical: string;
    linkedin: string;
    github: string;
    email: string;
    partnershipsEmail: string;
  };
  pilot: {
    location: string;
    region: string;
    stage: string;
    description: string;
    showStats: boolean;
    stats: {
      activeFarmers: string | null;
      verifiedShops: string | null;
      queriesResolved: string | null;
    };
  };
  navigation: Array<{ label: string; href: string }>;
  footerLinks: Array<{ label: string; href: string }>;
  meta: {
    title: string;
    description: string;
    keywords: string;
    author: string;
    ogImage: string;
  };
}

export const siteConfig: SiteConfig = {
  name: "BhoomiMitra AI",
  tagline: "From Farmer’s Voice to Smarter Decisions",
  positioning:
    "A WhatsApp-first, voice-first agricultural AI assistant that helps farmers ask farming questions in their own language, understand crop problems through images, receive agricultural guidance, and discover verified agricultural shops and stock.",
  foundedYear: 2026,
  status: "Early-stage Agricultural AI Initiative",

  urls: {
    // Live Production WhatsApp Assistant (Verified from project data-deletion & compliance)
    whatsapp: "https://wa.me/917794852530?text=Namaste%20BhoomiMitra",
    whatsappDisplay: "+91 77948 52530",
    // Website URLs (Canonical domain target; public URL unassigned pending confirmation)
    website: null,
    canonical: "https://bhoomimitra.ai",
    linkedin: "https://www.linkedin.com/company/bhoomimitra-ai",
    github: "https://github.com/kavyasriakkathi/BhoomiMitraAI",
    email: "contact@bhoomimitra.ai",
    partnershipsEmail: "partners@bhoomimitra.ai",
  },

  pilot: {
    location: "Korutla, Telangana",
    region: "Jagtial District",
    stage: "Controlled Real-World Pilot",
    description:
      "We are validating BhoomiMitra through controlled real-world testing with farmers and verified agricultural shops.",
    showStats: false,
    stats: {
      activeFarmers: null,
      verifiedShops: null,
      queriesResolved: null,
    },
  },

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
    { label: "Contact", href: "#contact" },
  ],

  footerLinks: [
    { label: "Home", href: "#hero" },
    { label: "How It Works", href: "#how-it-works" },
    { label: "Technology", href: "#intelligence" },
    { label: "Safety & Trust", href: "#safety" },
    { label: "Pilot", href: "#pilot" },
    { label: "About", href: "#about" },
    { label: "Contact", href: "#contact" },
    { label: "Privacy Policy", href: "privacy.html" },
    { label: "Terms of Service", href: "terms.html" },
  ],

  meta: {
    title: "BhoomiMitra AI | From Farmer’s Voice to Smarter Decisions",
    description:
      "BhoomiMitra AI is a WhatsApp-first agricultural AI assistant helping farmers ask questions through voice, text and crop photos in their own language.",
    keywords:
      "BhoomiMitra AI, agricultural AI, WhatsApp farming assistant, crop advisory, crop disease identification, agri shop discovery, Telangana farmers, multilingual agri tech",
    author: "BhoomiMitra AI Team",
    ogImage: "/assets/og-image.png",
  },
};
