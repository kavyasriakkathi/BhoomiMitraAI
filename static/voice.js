/* ==========================================================================
   BhoomiMitra AI — Voice Assistant & Speech Processing Engine
   Provides:
   - Speech-To-Text (STT) via Web Speech API / Backend STT API
   - Text-To-Speech (TTS) via Web Speech Synthesis / Backend Audio
   - Voice State Machine (idle, listening, thinking, speaking, paused)
   - Voice Waveform Animation & State Indicators
   - Multilingual Voice Recognition for 11 Indian Languages
   ========================================================================== */

class VoiceAssistantEngine {
  constructor() {
    this.currentLanguage = 'en';
    this.state = 'idle'; // idle | listening | thinking | speaking | paused
    this.recognition = null;
    this.synthesis = window.speechSynthesis || null;
    this.currentUtterance = null;
    this.speechRate = 1.0;
    this.speechPitch = 1.0;
    this.voiceGender = 'female';
    this.onResultCallback = null;
    this.onStateChangeCallback = null;
    this.isContinuous = false;
    this.wakeWordEnabled = true;

    this.initSpeechRecognition();
  }

  initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = false;
      this.recognition.interimResults = true;
      this.recognition.maxAlternatives = 1;

      this.recognition.onstart = () => {
        this.setState('listening');
      };

      this.recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }

        const transcript = finalTranscript || interimTranscript;
        if (this.onResultCallback && transcript) {
          this.onResultCallback(transcript, Boolean(finalTranscript));
        }
      };

      this.recognition.onerror = (event) => {
        console.warn("Speech recognition error:", event.error);
        if (this.state === 'listening') {
          this.setState('idle');
        }
      };

      this.recognition.onend = () => {
        if (this.state === 'listening') {
          this.setState('idle');
        }
      };
    } else {
      console.warn("Web Speech API recognition not supported in this browser. Falling back to mic recording.");
    }
  }

  setLanguage(langCode) {
    this.currentLanguage = langCode;
    const langMeta = I18N_DICTIONARY[langCode] || I18N_DICTIONARY.en;
    if (this.recognition) {
      this.recognition.lang = langMeta.code || 'en-IN';
    }
  }

  setState(newState) {
    this.state = newState;
    this.updateVisualizerState();
    if (this.onStateChangeCallback) {
      this.onStateChangeCallback(newState);
    }
  }

  updateVisualizerState() {
    const micBtn = document.getElementById('main-mic-btn');
    const statePill = document.getElementById('voice-state-pill');
    const waveContainer = document.getElementById('voice-waveform-container');
    const stateText = document.getElementById('voice-state-text');

    if (!micBtn || !statePill) return;

    // Reset state classes
    micBtn.classList.remove('listening', 'thinking', 'speaking');
    if (waveContainer) waveContainer.classList.remove('active', 'speaking-wave');

    const langMeta = I18N_DICTIONARY[this.currentLanguage] || I18N_DICTIONARY.en;

    switch (this.state) {
      case 'listening':
        micBtn.classList.add('listening');
        if (waveContainer) waveContainer.classList.add('active');
        statePill.innerHTML = `<span class="pulse-dot red"></span> ${t('micListening', this.currentLanguage)}`;
        if (stateText) stateText.innerText = t('micListening', this.currentLanguage);
        break;

      case 'thinking':
        micBtn.classList.add('thinking');
        statePill.innerHTML = `<span class="pulse-dot yellow"></span> ${t('micThinking', this.currentLanguage)}`;
        if (stateText) stateText.innerText = t('micThinking', this.currentLanguage);
        break;

      case 'speaking':
        micBtn.classList.add('speaking');
        if (waveContainer) waveContainer.classList.add('active', 'speaking-wave');
        statePill.innerHTML = `<span class="pulse-dot green"></span> ${t('micSpeaking', this.currentLanguage)}`;
        if (stateText) stateText.innerText = t('micSpeaking', this.currentLanguage);
        break;

      case 'idle':
      default:
        statePill.innerHTML = `<span class="pulse-dot green"></span> ${t('micInstruction', this.currentLanguage)}`;
        if (stateText) stateText.innerText = t('micInstruction', this.currentLanguage);
        break;
    }
  }

  startListening(onResult) {
    this.stopSpeaking();
    this.onResultCallback = onResult;

    if (this.recognition) {
      const langMeta = I18N_DICTIONARY[this.currentLanguage] || I18N_DICTIONARY.en;
      this.recognition.lang = langMeta.code || 'en-IN';
      try {
        this.recognition.start();
      } catch (e) {
        // Handle case where recognition was already active
        this.recognition.stop();
        setTimeout(() => this.recognition.start(), 200);
      }
    } else {
      alert("Speech recognition is not supported in this browser. Please type your query.");
    }
  }

  stopListening() {
    if (this.recognition) {
      this.recognition.stop();
    }
    this.setState('idle');
  }

  speakText(text, lang = this.currentLanguage, onComplete = null) {
    if (!this.synthesis || !text) {
      if (onComplete) onComplete();
      return;
    }

    this.stopSpeaking();
    this.setState('speaking');

    // Clean markdown asterisks for natural voice reading
    const cleanText = text.replace(/\*\*(.*?)\*\*/g, '$1')
                          .replace(/•/g, '')
                          .replace(/🤖/g, '')
                          .replace(/👨‍🌾/g, '')
                          .replace(/🏬/g, '');

    const utterance = new SpeechSynthesisUtterance(cleanText);
    const langMeta = I18N_DICTIONARY[lang] || I18N_DICTIONARY.en;
    utterance.lang = langMeta.code || 'en-IN';
    utterance.rate = this.speechRate;
    utterance.pitch = this.speechPitch;

    // Pick best matching voice if available
    const voices = this.synthesis.getVoices();
    const matchingVoice = voices.find(v => v.lang.startsWith(langMeta.code.substring(0, 2))) ||
                          voices.find(v => v.lang.includes('IN')) ||
                          voices[0];

    if (matchingVoice) {
      utterance.voice = matchingVoice;
    }

    utterance.onend = () => {
      this.setState('idle');
      if (onComplete) onComplete();
    };

    utterance.onerror = (e) => {
      console.warn("TTS Synthesis Error:", e);
      this.setState('idle');
      if (onComplete) onComplete();
    };

    this.currentUtterance = utterance;
    this.synthesis.speak(utterance);
  }

  pauseSpeaking() {
    if (this.synthesis && this.synthesis.speaking) {
      this.synthesis.pause();
      this.setState('paused');
    }
  }

  resumeSpeaking() {
    if (this.synthesis && this.synthesis.paused) {
      this.synthesis.resume();
      this.setState('speaking');
    }
  }

  stopSpeaking() {
    if (this.synthesis) {
      this.synthesis.cancel();
    }
    this.setState('idle');
  }
}

// Global singleton instance
const voiceEngine = new VoiceAssistantEngine();
