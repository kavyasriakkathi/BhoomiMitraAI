# 🌱 BhoomiMitra AI — WhatsApp Conversation Design

> **Document Type:** Product Requirement Document (PRD) & UX Flow  
> **Audience:** Engineering, PMs, and Content Teams  
> **Objective:** Define the complete WhatsApp UX for Indian farmers.

## Core Design Principles

1. **Voice-First:** Every text response must be accompanied by a natural-sounding voice note in the farmer's language.
2. **One Question at a Time:** Never ask multiple questions in a single message.
3. **Forgiving Inputs:** Handle typos, colloquialisms, mixed languages (e.g., "Tomato price kya hai?"), and unclear voice notes gracefully.
4. **Context-Aware:** If a farmer asks "How much urea?", the AI should know they are growing 2 acres of Paddy planted 40 days ago.
5. **Always a Way Out:** Always provide an option to talk to a human expert.

---

## 🧠 AI State Machine & Decision Logic

The AI decides the next question based on **Confidence Scoring & Intent Recognition**:

- **Confidence > 85%**: Direct response. (e.g., "Your crop is Cotton, the disease is Pink Bollworm. Use Spinosad.")
- **Confidence 50% - 85% (Low Confidence)**: Trigger Follow-up flow. AI identifies missing parameters and asks one targeted question. (e.g., "I see spots on the leaf. Are they brown or yellow?")
- **Confidence < 50%**: Trigger Expert Escalation flow immediately.
- **Intent**: Categorized via NLP (e.g., `intent: weather_query`, `intent: market_price`).

---

## 1. Onboarding & Registration (Flow 1 & 2)

**Goal:** Capture language, location, and crop profile with minimal friction.

### Flow 1: First-Time Registration
```mermaid
graph TD
    A[User: Sends 'Hi'] --> B[AI: Detects new number]
    B --> C[AI: "Namaste! Welcome to BhoomiMitra. Which language do you prefer? 1. English 2. हिन्दी 3. తెలుగు 4. ಕನ್ನಡ 5. தமிழ்"]
    
    C -->|Replies '3' or 'Telugu'| D[AI: Sets Language = Telugu]
    C -->|Unrecognized/Invalid| E[AI: "Please reply with 1, 2, 3, 4, or 5."]
    E --> C
    
    D --> F[AI (in Telugu): "To give you accurate weather and mandi prices, please share your location. Click the 📎 icon and select 'Location'."]
    
    F -->|Shares Location| G[AI: "Got it! Your district is Warangal. What crop are you currently growing?"]
    F -->|Sends text instead| H[AI: "I need your exact location. You can type your village name, or press the 📎 icon."]
    
    G -->|Replies 'Cotton'| I[AI: "Great. How many acres of Cotton?"]
    I -->|Replies '2'| J[AI: "When did you sow the seeds? (e.g., '10 days ago', or 'June 15')"]
    J -->|Replies 'June 15'| K[AI: "Setup complete! You can ask me about weather, prices, or send a photo of your crop anytime."]
```

### Flow 2: Returning Farmer
**User:** "Hi" / "Namaste"
**AI:** "Namaste Ramesh! Your Cotton crop is now 45 days old. It's going to rain lightly tomorrow. How can I help you today?"

---

## 2. Diagnostics & Recommendations (Flows 5, 6, 7, 8, 9, 21)

**Goal:** Diagnose issues via photos and recommend treatments.

### Image Upload Conversation Tree (Disease / Pest / Nutrient)
```mermaid
graph TD
    A[User: Sends photo of leaf] --> B[AI: Vision Model Analyzes Image]
    
    B --> C{Confidence Score}
    
    C -->|> 85% (Clear Disease)| D[AI: "I have analyzed the photo. Your Cotton crop has Pink Bollworm (disease detection)."]
    C -->|50-85% (Needs clarity)| E[AI: "I see some yellowing, but it's not very clear. Are the spots appearing on the top or bottom of the leaves?"]
    C -->|< 50% (Unclear/Blurry)| F[AI: "The photo is a bit blurry. Could you take a closer, clearer photo in good sunlight?"]
    
    E -->|Replies 'Bottom'| D
    
    D --> G[AI: "To treat this, you should spray Spinosad 45% SC. Would you like to know where to buy this nearby? (Yes/No)"]
    
    G -->|Yes| H[Trigger Flow 11: Nearby Shops]
    G -->|No| I[Trigger Feedback Flow]
```

### Invalid Image Handling:
**User:** *Sends a photo of a tractor/dog/selfie*
**AI:** "This looks like a great photo, but I can only analyze crop diseases! Please send a clear photo of your crop's leaves, stem, or fruit."

---

## 3. Crop Planning & Seeds (Flows 3, 4)

**User:** "What should I plant this Kharif season?"
**AI Engine Check:** Context = Warangal, Red Soil, 2 Acres.
**AI:** "In Warangal, considering the upcoming monsoon and your red soil, Cotton and Chilli are highly profitable. 
1. Cotton (Average profit: ₹40,000/acre)
2. Chilli (Average profit: ₹60,000/acre)
Which one would you like to know more about?"

**User:** "Cotton"
**AI:** "Excellent. For red soil in your area, the best seed varieties are Rasi 659 and Tulasi 117. Do you want me to find shops selling these seeds near you?"

---

## 4. Location-Based Services (Flows 10, 11, 12, 13)

### Weather & Irrigation (Flow 12 & 15)
**User:** "Will it rain today?"
**AI:** "Yes, there is an 80% chance of heavy rain in Warangal starting at 4 PM today."
**AI (Proactive Irrigation Advice):** "Since it will rain heavily, please stop any planned irrigation for your Cotton crop today to avoid waterlogging."

### Market Prices (Flow 13)
**User:** "Tomato price?"
**AI Engine Check:** Checks Agmarknet for user's district.
**AI:** "Today's Tomato prices near Warangal:
🟢 Enumamula Mandi: ₹2,500/quintal (Up by ₹200 from yesterday)
🟡 Jangaon Mandi: ₹2,200/quintal
Would you like directions to Enumamula Mandi?"

### Nearby Shops & Urea (Flow 10 & 11)
**User:** "Where can I find Urea?"
**AI:** "I found 3 agro shops near you with Urea in stock:
1. Kisan Seva Kendra (3km away) - ₹266/bag
2. Balaji Fertilizers (5km away) - ₹266/bag
Reply with a number (1 or 2) to get the shop owner's phone number and Google Maps location."

---

## 5. Proactive Alerts & Govt Schemes (Flows 14, 16)

### Harvest Reminders (Flow 16)
**System Trigger:** Cotton crop reaches 150 days.
**AI (Push Message):** "Namaste! Your Cotton crop is around 150 days old and should be ready for the first picking. The current market price at Enumamula mandi is ₹7,200/quintal. Let me know if you need help finding buyers!"

### Government Schemes (Flow 14)
**User:** "Any subsidy for drip irrigation?"
**AI:** "Yes! Under the PM Krishi Sinchayee Yojana, farmers in Telangana get a 90% subsidy on drip irrigation. 
To apply, you need:
- Pattadar Passbook
- Aadhaar Card
Shall I send you the application link, or connect you to an expert for help?"

---

## 6. Escalation, Voice, & Feedback (Flows 17, 18, 19, 20, 22)

### Voice Messages (Flow 20)
**User:** *Sends 10-second voice note in Telugu*
**System Action:** Whisper STT converts to text -> Translates to English -> LLM Processes -> Translates to Telugu -> Google TTS generates Voice.
**AI:** *Replies with a Text message AND a Voice Note*

### Expert Consultation & Emergency (Flow 17 & 18)
**User:** "My crop is dying overnight, help!" (Intent: Emergency)
**AI:** "I understand this is urgent. I am connecting you directly to our Agriculture Expert, Dr. Rao. He will reply to you on this chat within 15 minutes. 
Meanwhile, please send 2 clear photos of the dying crop."
**System Action:** Flags conversation as HIGH PRIORITY in Expert Dashboard.

### Feedback Collection (Flow 22)
**Rule:** Only ask for feedback after a completed diagnostic or recommendation cycle (max once per week to avoid spam).
**AI:** "Was this fertilizer recommendation helpful? 
Reply with:
👍 for Yes
👎 for No"
*If 👎:* "I am sorry to hear that. I will forward this to our agriculture experts so we can improve. Would you like to speak to an expert now?"

---

## 7. Handling Unexpected Input (Error Flows)

**Scenario A: Gibberish or Unrelated Text**
**User:** "Who will win the election?"
**AI:** "I am BhoomiMitra, your farming assistant! 🌾 I don't know about elections, but I can help you with crop diseases, market prices, weather, and fertilizers. How can I help with your farm today?"

**Scenario B: Multiple Intents**
**User:** "What is the price of cotton and what fertilizer to use?"
**AI:** "Let's take this one by one! 
First, the price of Cotton at Enumamula Mandi is ₹7,200/quintal. 
Now, regarding fertilizer, could you tell me how many days old your Cotton crop is?"
