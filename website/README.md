# BhoomiMitra AI — Public Product & Startup Website

> **Tagline:** “From Farmer’s Voice to Smarter Decisions”

This directory contains the public-facing startup website for **BhoomiMitra AI**, designed for:
- Startup applications (FICCI, Agri-tech awards, T-Hub, WE Hub, incubator applications)
- Investor and mentor presentations
- Farmer onboarding and QR code distribution
- Professional sharing on LinkedIn

---

## 📁 Directory Structure

```
website/
├── config/
│   ├── site.js          # Central JavaScript site configuration (WhatsApp, socials, pilot, emails)
│   └── site.ts          # Strongly typed TypeScript configuration
├── css/
│   └── style.css        # Clean, modern, responsive CSS stylesheet (deep green & earth tones)
├── js/
│   └── main.js          # Interactive behaviors (dynamic config hydration, mobile menu, audio demo)
├── assets/
│   ├── logo.svg         # BhoomiMitra AI vector logo
│   ├── favicon.svg      # Favicon
│   └── hero-flow.svg    # Farmer → Voice/Text/Photo → AI → Smarter Decision vector diagram
├── index.html           # Main 14-section startup landing page
├── privacy.html         # Early-stage privacy policy
├── terms.html           # Early-stage terms of service
└── README.md            # This documentation
```

---

## ⚙️ Central Configuration (`config/site.js`)

All startup links, numbers, and pilot details are kept in one single configuration file (`config/site.js` & `config/site.ts`).
To change:
- **WhatsApp link / phone number**: Edit `siteConfig.urls.whatsapp`
- **Contact Email**: Edit `siteConfig.urls.email`
- **Partnerships Email**: Edit `siteConfig.urls.partnershipsEmail`
- **Pilot Location**: Edit `siteConfig.pilot.location`
- **Social profiles**: Edit `siteConfig.urls.linkedin` and `siteConfig.urls.github`

All buttons and labels on the site will automatically update on page load.

---

## 🚀 Running Locally

You can preview the website using Python's built-in HTTP server:

```bash
# From the repository root:
.\.venv\Scripts\python.exe -m http.server 3000 --directory website
```

Then open [http://localhost:3000](http://localhost:3000) in your web browser.

---

## 🌐 Production Deployment

Because this website uses standards-compliant HTML5, CSS3, and JavaScript with zero build dependencies, it can be deployed instantly to:
- **Cloudflare Pages**: Point to `website/` directory
- **Vercel**: Set root directory to `website/`
- **Netlify**: Set publish directory to `website/`
- **GitHub Pages**: Serve from `website/` or root
- **AWS S3 / CloudFront**: Sync `website/` folder
