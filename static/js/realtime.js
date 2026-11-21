/**
 * Mintx Solutions AI Realtime Voice Studio - Frontend Controller
 * Full Bi-Directional PCM 24kHz Web Audio + Canvas Visualizer + OpenAI Realtime WS Proxy
 */

(function () {
  'use strict';

  // ── State Variables ────────────────────────────────────────────────────────
  let ws = null;
  let isConnected = false;
  let isMuted = false;
  let audioContext = null;
  let micStream = null;
  let micSourceNode = null;
  let micProcessorNode = null;
  let micAnalyserNode = null;
  let speakerAnalyserNode = null;
  let speakerGainNode = null;

  // Conversation Audio Recording Pipeline & Text Formatter
  let isConversationRecordingEnabled = true;
  let isActivelyRecording = false;
  let recordingStartTime = 0;
  let recordingTimerInterval = null;
  let recordedPcmChunks = [];
  let lastRecordedAudioBlob = null;
  let lastRecordedAudioUrl = null;
  let lastRecordedTranscriptText = '';
  let lastRecordedSessionId = '';
  let recMixerNode = null;
  let recProcessorNode = null;

  // Audio Playback Queue & Echo Shield
  let nextPlayTime = 0;
  const PLAYBACK_JITTER_BUFFER_SEC = 0.15;
  let pcmByteCarry = new Uint8Array(0);
  let activeAudioSources = [];
  let isAiSpeaking = false;
  let speechCooldownTimer = null;
  let pendingTextMessage = null;

  // Visualizer
  let canvas = null;
  let ctx = null;
  let animFrameId = null;
  let currentMicVolume = 0;
  let currentSpeakerVolume = 0;

  // UI State: 'idle' | 'listening' | 'thinking' | 'speaking' | 'connecting' | 'disconnected'
  let currentState = 'disconnected';

  // One-time recommended call settings: fast replies (low reasoning; facts come from RAG),
  // Saaras v4 hearing, and no barge-in (on laptop speakers the AI's own echo would cut its voice).
  if (!localStorage.getItem('recommended_call_settings_v1')) {
    localStorage.setItem('preferred_reasoning_effort', 'low');
    localStorage.setItem('preferred_stt_model', 'saaras:v4');
    localStorage.setItem('preferred_barge_in', 'false');
    localStorage.setItem('recommended_call_settings_v1', '1');
  }

  // Current session config: GPT-Live (gpt-live-1) hears, thinks and speaks with its own voice
  const LIVE_MODEL = 'gpt-live-1';
  const LIVE_VOICES = ['marin', 'cedar', 'alloy', 'ash', 'ballad', 'coral', 'echo', 'sage', 'shimmer', 'verse', 'quartz', 'gleam', 'willow'];
  const toLiveVoice = (v) => (LIVE_VOICES.includes(v) ? v : 'marin');
  const SARVAM_VOICE_IDS = ['sarvam', 'kavitha', 'kavya', 'priya', 'shreya', 'roopa', 'mani', 'gokul', 'vijay', 'elevenlabs'];
  const isSarvamVoiceId = (v) => SARVAM_VOICE_IDS.includes(v);
  let lastGptVoice = toLiveVoice(localStorage.getItem('preferred_gpt_voice'));
  let currentVoice = toLiveVoice(localStorage.getItem('preferred_voice') || lastGptVoice);
  localStorage.setItem('preferred_voice', currentVoice);
  let currentModel = LIVE_MODEL;
  localStorage.setItem('preferred_model', LIVE_MODEL);
  let currentReasoningModel = localStorage.getItem('preferred_reasoning_model') || 'gpt-5.6-sol';
  let audioMode = 'browser'; // 'browser' | 'host'
  let currentReasoningEffort = localStorage.getItem('preferred_reasoning_effort') || 'low';
  let currentTemperature = parseFloat(localStorage.getItem('preferred_temperature')) || 0.60;
  let currentSarvamSpeaker = localStorage.getItem('preferred_sarvam_speaker') || 'kavitha';
  let currentSarvamModel = localStorage.getItem('preferred_sarvam_model') || 'bulbul:v3';
  const DEFAULT_SARVAM_PACE = 1.25;
  // One-time upgrade: browsers still on an old default (1.0 / 1.2) move to the faster 1.25 default
  if (!localStorage.getItem('sarvam_pace_v125')) {
    const storedPace = parseFloat(localStorage.getItem('preferred_sarvam_pace'));
    if (!storedPace || storedPace === 1 || storedPace === 1.2) {
      localStorage.setItem('preferred_sarvam_pace', String(DEFAULT_SARVAM_PACE));
    }
    localStorage.setItem('sarvam_pace_v125', '1');
  }
  let currentSarvamPace = parseFloat(localStorage.getItem('preferred_sarvam_pace')) || DEFAULT_SARVAM_PACE;
  let currentSarvamLanguage = localStorage.getItem('preferred_sarvam_language') || 'ta-IN';

  // Legacy ElevenLabs compatibility state
  let rawElevenModel = localStorage.getItem('preferred_eleven_model');
  let currentElevenLabsModel = (!rawElevenModel || rawElevenModel === 'eleven_v4_turbo') ? 'eleven_turbo_v2_5' : rawElevenModel;
  let currentElevenLabsVoiceId = localStorage.getItem('preferred_eleven_voice_id') || 'mGboHvCVOXWYeFL8KTR0';
  let currentElevenLabsStability = parseFloat(localStorage.getItem('preferred_eleven_stability')) || 0.40;
  let currentElevenLabsSimilarity = parseFloat(localStorage.getItem('preferred_eleven_similarity')) || 0.80;
  let currentElevenLabsStyle = parseFloat(localStorage.getItem('preferred_eleven_style')) || 0.35;
  let currentElevenLabsSpeakerBoost = localStorage.getItem('preferred_eleven_speaker_boost') !== 'false';

  // STT (Speech-to-Text) & VAD Controls — tuned for fast, natural conversation
  // One-time upgrade: Sarvam Saaras v4 becomes the default caller speech recognizer
  if (!localStorage.getItem('stt_saaras_v4_default')) {
    localStorage.setItem('preferred_stt_model', 'saaras:v4');
    localStorage.setItem('stt_saaras_v4_default', '1');
  }
  let currentSttModel = localStorage.getItem('preferred_stt_model') || 'saaras:v4';
  let rawVadSilence = parseInt(localStorage.getItem('preferred_vad_silence_ms'));
  let currentVadSilenceMs = (!rawVadSilence || rawVadSilence < 450) ? 500 : rawVadSilence;
  let rawVadThreshold = parseFloat(localStorage.getItem('preferred_vad_threshold'));
  let currentVadThreshold = (!rawVadThreshold || rawVadThreshold < 0.48) ? 0.50 : rawVadThreshold;
  let rawVadPrefix = parseInt(localStorage.getItem('preferred_vad_prefix_ms'));
  let currentVadPrefixMs = (!rawVadPrefix || rawVadPrefix < 200) ? 300 : rawVadPrefix;
  let currentMicEcho = localStorage.getItem('preferred_mic_echo') !== 'false';
  let currentMicNoise = localStorage.getItem('preferred_mic_noise') !== 'false';
  let currentMicGain = localStorage.getItem('preferred_mic_gain') !== 'false';

  // Barge-in: while Yazhini speaks, mic audio is held back (echo shield) unless the caller
  // talks clearly louder than the speaker echo for ~250 ms.
  let currentBargeIn = localStorage.getItem('preferred_barge_in') === 'true';
  const BARGE_IN_RMS = 0.07;
  // Caller must also be louder than this fraction of what the speaker is playing right now
  const BARGE_IN_SPEAKER_RATIO = 0.6;
  const BARGE_IN_LOUD_CHUNKS = 4;
  const BARGE_IN_PREROLL_CHUNKS = 8;
  const BARGE_IN_DROP_AUDIO_MS = 500;
  let bargeInLoudCount = 0;
  let lastCallerSpeaking = null;
  let bargeInPreroll = [];
  let dropAiAudioUntil = 0;

  // RAG Embeddings & Vector DB Stack
  let currentEmbeddingModel = localStorage.getItem('preferred_embedding_model') || 'text-embedding-3-large';
  let currentVectorDb = localStorage.getItem('preferred_vector_db') || 'qdrant';

  // TTS (Text-to-Speech) Playback Controls
  let currentTtsSpeed = parseFloat(localStorage.getItem('preferred_tts_speed')) || 1.00;
  let currentTtsVolume = parseInt(localStorage.getItem('preferred_tts_volume')) || 100;
  // Default text: OpenAI does not speak; Sarvam AI is the primary voice engine
  let currentTtsModality = localStorage.getItem('preferred_tts_modality') || 'text';
  let currentTtsDelivery = localStorage.getItem('preferred_tts_delivery') || 'stream';

  // Transcripts
  let conversationHistory = [];
  let currentAssistantTranscript = '';
  let lastAssistantDisplayText = '';

  // Instructions State
  let instructionsList = [];
  let activeInstructionId = 'default-mintx';
  let activeInstruction = null;

  // Voice Training & Test Mode State
  let currentOperatingMode = 'live'; // 'live' | 'test'
  let isRecordingVoiceCorrection = false;
  let lastRecognizedTeachSpeech = '';
  let correctionAlreadyAppliedForCurrentRecording = false;
  let voiceTeachMediaRecorder = null;
  let voiceTeachAudioChunks = [];
  let speechRecognizer = null;
  let activeCorrectionTarget = { responseText: '', context: '' };
  let isModalRecording = false;
  let modalMediaRecorder = null;
  let modalAudioChunks = [];

  // Voice Metadata Dictionary
  const VOICE_INFO = {
    sarvam: { name: 'Yazhini (Sarvam AI)', tone: 'Warm & Natural Tamil Care', gender: 'Female', emoji: '🌟' },
    kavitha: { name: 'Kavitha (Sarvam AI)', tone: 'Natural South Indian Female', gender: 'Female', emoji: '🌟' },
    kavya: { name: 'Kavya (Sarvam AI)', tone: 'Warm & Polite Female', gender: 'Female', emoji: '🌟' },
    priya: { name: 'Priya (Sarvam AI)', tone: 'Clear & Conversational Female', gender: 'Female', emoji: '🌟' },
    shreya: { name: 'Shreya (Sarvam AI)', tone: 'Expressive Female', gender: 'Female', emoji: '🌟' },
    roopa: { name: 'Roopa (Sarvam AI)', tone: 'Formal & Articulate Female', gender: 'Female', emoji: '🌟' },
    mani: { name: 'Mani (Sarvam AI)', tone: 'Conversational South Indian Male', gender: 'Male', emoji: '🌟' },
    gokul: { name: 'Gokul (Sarvam AI)', tone: 'Energetic & Friendly Male', gender: 'Male', emoji: '🌟' },
    vijay: { name: 'Vijay (Sarvam AI)', tone: 'Deep & Confident Male', gender: 'Male', emoji: '🌟' },
    marin: { name: 'Marin', tone: 'Warm & Natural Female', gender: 'Female', emoji: '👩' },
    alloy: { name: 'Alloy', tone: 'Neutral & Balanced', gender: 'Neutral', emoji: '🤖' },
    coral: { name: 'Coral', tone: 'Cheerful & Expressive Female', gender: 'Female', emoji: '🌸' },
    echo: { name: 'Echo', tone: 'Smooth & Resonant Male', gender: 'Male', emoji: '👨' },
    nova: { name: 'Nova', tone: 'Bright & Energetic Female', gender: 'Female', emoji: '⚡' },
    shimmer: { name: 'Shimmer', tone: 'Clear & Lyrical Female', gender: 'Female', emoji: '✨' },
    ash: { name: 'Ash', tone: 'Conversational Casual Male', gender: 'Male', emoji: '🎙️' },
    sage: { name: 'Sage', tone: 'Calm & Professional Female', gender: 'Female', emoji: '🌿' },
    onyx: { name: 'Onyx', tone: 'Deep Authoritative Male', gender: 'Male', emoji: '🎩' },
    verse: { name: 'Verse', tone: 'Dynamic & Versatile Male', gender: 'Male', emoji: '🎭' },
    ballad: { name: 'Ballad', tone: 'Melodic & Gentle Male', gender: 'Male', emoji: '🎵' },
    fable: { name: 'Fable', tone: 'British Accent Male', gender: 'Male', emoji: '🇬🇧' },
  };

  // ── DOM Element References ────────────────────────────────────────────────
  const elements = {
    // Mode Switcher
    modeSwitcher: document.getElementById('mode-switcher'),
    modeBtnLive: document.getElementById('mode-btn-live'),
    modeBtnTest: document.getElementById('mode-btn-test'),

    // Status & Badges
    statusPill: document.getElementById('status-pill'),
    statusText: document.getElementById('status-text'),
    pingMetric: document.getElementById('metric-ping-val'),
    voiceBadge: document.getElementById('badge-voice-val'),
    modelBadge: document.getElementById('badge-model-val'),

    // Voice Switchers
    navVoiceSelect: document.getElementById('nav-voice-select'),
    sideVoiceSelect: document.getElementById('side-voice-select'),
    quickVoiceActiveLabel: document.getElementById('quick-voice-active-label'),
    btnVoiceChips: document.querySelectorAll('.btn-voice-chip[data-voice]'),
    btnVoiceEngines: document.querySelectorAll('.btn-voice-engine[data-engine]'),
    btnMoreVoices: document.getElementById('btn-more-voices'),
    voiceDropdownMenu: document.getElementById('voice-dropdown-menu'),
    voiceMenuItems: document.querySelectorAll('.voice-menu-item'),

    // Visualizer & Orb
    orbWrapper: document.getElementById('orb-wrapper'),
    coreOrb: document.getElementById('core-orb'),
    stateTitle: document.getElementById('state-title'),
    stateHelper: document.getElementById('state-helper'),
    visualizerCanvas: document.getElementById('audio-visualizer-canvas'),

    // Dual-Side Real-Time Stage
    sideCaller: document.getElementById('side-caller'),
    sideAssistant: document.getElementById('side-assistant'),
    callerTranscriptText: document.getElementById('caller-transcript-text'),
    callerMicStatus: document.getElementById('caller-mic-status'),
    callerLiveIndicator: document.getElementById('caller-live-indicator'),
    callerLangBadge: document.getElementById('caller-lang-badge'),
    callerLevelMeter: document.getElementById('caller-level-meter'),
    verbalAckBanner: document.getElementById('verbal-ack-banner'),
    verbalAckQuote: document.getElementById('verbal-ack-quote'),

    // Live Transcript
    liveTranscriptCard: document.getElementById('live-transcript-card'),
    liveSpeakerTag: document.getElementById('live-speaker-tag'),
    liveTranscriptText: document.getElementById('live-transcript-text'),
    btnCorrectLive: document.getElementById('btn-correct-live'),

    // Voice Training Console
    voiceTrainConsole: document.getElementById('voice-train-console'),
    btnVoiceTeach: document.getElementById('btn-voice-teach'),
    teachBtnText: document.getElementById('teach-btn-text'),
    teachMicRings: document.getElementById('teach-mic-rings'),
    teachMicIcon: document.getElementById('teach-mic-icon'),
    teachStatusDisplay: document.getElementById('teach-status-display'),
    teachLiveText: document.getElementById('teach-live-text'),
    teachQuickActions: document.getElementById('teach-quick-actions'),
    teachEditInput: document.getElementById('teach-edit-input'),
    btnCancelVoiceCorrection: document.getElementById('btn-cancel-voice-correction'),
    btnApplyVoiceCorrection: document.getElementById('btn-apply-voice-correction'),
    toggleAutoDetectCorrections: document.getElementById('toggle-auto-detect-corrections'),
    btnViewTrainingHistory: document.getElementById('btn-view-training-history'),
    trainHistoryCount: document.getElementById('train-history-count'),
    btnUndoCorrection: document.getElementById('btn-undo-correction'),
    trainChipBtns: document.querySelectorAll('.train-chip-btn'),
    trainAlertBanner: document.getElementById('train-alert-banner'),
    trainAlertTitle: document.getElementById('train-alert-title'),
    trainAlertDiff: document.getElementById('train-alert-diff'),
    btnCloseTrainAlert: document.getElementById('btn-close-train-alert'),

    // Interactive Correction Studio & Stage Customized Rules
    inputBridgeCorrection: document.getElementById('input-bridge-correction'),
    btnBridgeApplyCorrection: document.getElementById('btn-bridge-apply-correction'),
    btnBridgeSpeakCorrection: document.getElementById('btn-bridge-speak-correction'),
    bridgeMicIcon: document.getElementById('bridge-mic-icon'),
    bridgeMicText: document.getElementById('bridge-mic-text'),
    bridgeRulesList: document.getElementById('bridge-rules-list'),
    bridgeRulesCountBadge: document.getElementById('bridge-rules-count-badge'),
    btnBridgeRefreshRules: document.getElementById('btn-bridge-refresh-rules'),
    btnToggleStageRules: document.getElementById('btn-toggle-stage-rules'),
    stageRulesDrawer: document.getElementById('stage-rules-drawer'),
    stageRulesList: document.getElementById('stage-rules-list'),
    stageRulesCount: document.getElementById('stage-rules-count'),
    btnStageAddRule: document.getElementById('btn-stage-add-rule'),
    inputNewCustomRule: document.getElementById('input-new-custom-rule'),
    btnSubmitNewRule: document.getElementById('btn-submit-new-rule'),

    // Control Buttons
    btnMasterCall: document.getElementById('btn-master-call'),
    btnCallText: document.getElementById('btn-call-text'),
    btnToggleMic: document.getElementById('btn-toggle-mic'),
    btnInterrupt: document.getElementById('btn-interrupt'),
    btnAudioMode: document.getElementById('btn-audio-mode'),
    btnHistory: document.getElementById('btn-history'),
    btnSettings: document.getElementById('btn-settings'),

    // Instructions & Personas
    btnOpenInstructions: document.getElementById('btn-open-instructions'),
    badgeInstructionVal: document.getElementById('badge-instruction-val'),
    instructionsModal: document.getElementById('instructions-modal'),
    instructionsBackdrop: document.getElementById('instructions-backdrop'),
    btnCloseInstructions: document.getElementById('btn-close-instructions'),
    tabBtnList: document.getElementById('tab-btn-list'),
    tabBtnEditor: document.getElementById('tab-btn-editor'),
    paneInstructionsList: document.getElementById('pane-instructions-list'),
    paneInstructionsEditor: document.getElementById('pane-instructions-editor'),
    instructionsGrid: document.getElementById('instructions-grid'),
    btnToolbarAdd: document.getElementById('btn-toolbar-add'),
    instructionCount: document.getElementById('instruction-count'),
    tabEditorTitle: document.getElementById('tab-editor-title'),
    editorInstructionId: document.getElementById('editor-instruction-id'),
    editorName: document.getElementById('editor-name'),
    editorCategory: document.getElementById('editor-category'),
    editorDesc: document.getElementById('editor-desc'),
    editorPrompt: document.getElementById('editor-prompt'),
    editorCharCount: document.getElementById('editor-char-count'),
    btnEditorCancel: document.getElementById('btn-editor-cancel'),
    btnEditorSave: document.getElementById('btn-editor-save'),
    btnEditorSaveActivate: document.getElementById('btn-editor-save-activate'),
    templateChips: document.querySelectorAll('.template-chip'),
    selectInstructionPreset: document.getElementById('setting-instruction-preset'),
    btnSettingsManageInstructions: document.getElementById('btn-settings-manage-instructions'),
    appToast: document.getElementById('app-toast'),

    // Main Stage Starting Greeting & Persona Cards
    startingGreetingCard: document.getElementById('starting-greeting-card'),
    greetingPhraseDisplay: document.getElementById('greeting-phrase-display'),
    btnEditGreeting: document.getElementById('btn-edit-greeting'),
    btnTestGreeting: document.getElementById('btn-test-greeting'),
    greetingPresetsRow: document.getElementById('greeting-presets-row'),
    chipGreetingPresets: document.querySelectorAll('.chip-greeting-preset'),
    greetingEditDrawer: document.getElementById('greeting-edit-drawer'),
    greetingInputField: document.getElementById('greeting-input-field'),
    btnCancelGreeting: document.getElementById('btn-cancel-greeting'),
    btnSaveGreeting: document.getElementById('btn-save-greeting'),

    stageCard: document.getElementById('persona-stage-card'),
    stageName: document.getElementById('stage-instruction-name'),
    stageCategory: document.getElementById('stage-instruction-category'),
    stagePreview: document.getElementById('stage-instruction-preview'),
    btnStageModify: document.getElementById('btn-stage-modify'),
    btnStageAdd: document.getElementById('btn-stage-add'),

    // Text & Quick Chips
    textInputBar: document.getElementById('text-input-bar'),
    inputMessage: document.getElementById('input-message'),
    btnSendText: document.getElementById('btn-send-text'),
    quickChips: document.querySelectorAll('.chip-btn'),

    // History Drawer
    historyDrawer: document.getElementById('history-drawer'),
    drawerBackdrop: document.getElementById('drawer-backdrop'),
    btnCloseDrawer: document.getElementById('btn-close-drawer'),
    drawerBody: document.getElementById('drawer-body'),
    btnClearHistory: document.getElementById('btn-clear-history'),
    btnCopyHistory: document.getElementById('btn-copy-history'),

    // Training History Modal
    trainHistoryModal: document.getElementById('train-history-modal'),
    trainHistoryBackdrop: document.getElementById('train-history-backdrop'),
    btnCloseTrainHistory: document.getElementById('btn-close-train-history'),
    btnCloseTrainHistoryFooter: document.getElementById('btn-close-train-history-footer'),
    trainHistoryList: document.getElementById('train-history-list'),
    inputNewCustomRule: document.getElementById('input-new-custom-rule'),
    btnSubmitNewRule: document.getElementById('btn-submit-new-rule'),

    // Response Correct Modal
    responseCorrectModal: document.getElementById('response-correct-modal'),
    responseCorrectBackdrop: document.getElementById('response-correct-backdrop'),
    btnCloseResponseCorrect: document.getElementById('btn-close-response-correct'),
    modalTargetAiResponse: document.getElementById('modal-target-ai-response'),
    btnModalMic: document.getElementById('btn-modal-mic'),
    modalMicText: document.getElementById('modal-mic-text'),
    modalMicHint: document.getElementById('modal-mic-hint'),
    modalCorrectionInput: document.getElementById('modal-correction-input'),
    btnModalCancelCorrect: document.getElementById('btn-modal-cancel-correct'),
    btnModalSubmitCorrect: document.getElementById('btn-modal-submit-correct'),

    // Settings Modal
    settingsModal: document.getElementById('settings-modal'),
    settingsBackdrop: document.getElementById('settings-backdrop'),
    btnCloseSettings: document.getElementById('btn-close-settings'),
    btnCancelSettings: document.getElementById('btn-cancel-settings'),
    btnSaveSettings: document.getElementById('btn-save-settings'),
    btnSyncMainSettings: document.getElementById('btn-sync-main-settings'),
    selectVoice: document.getElementById('setting-voice'),
    selectReasoningModel: document.getElementById('setting-reasoning-model'),
    customReasoningModelGroup: document.getElementById('setting-custom-reasoning-model-group'),
    customReasoningModelInput: document.getElementById('setting-custom-reasoning-model'),
    badgeReasoningTier: document.getElementById('badge-reasoning-tier'),
    badgeReasoningVal: document.getElementById('badge-reasoning-val'),
    summaryReasoningModel: document.getElementById('summary-reasoning-model'),
    summaryRealtimeModel: document.getElementById('summary-realtime-model'),
    summarySttModel: document.getElementById('summary-stt-model'),
    summaryTtsEngine: document.getElementById('summary-tts-engine'),
    selectModel: document.getElementById('setting-model'),
    customModelGroup: document.getElementById('setting-custom-model-group'),
    customModelInput: document.getElementById('setting-custom-model'),
    selectReasoningEffort: document.getElementById('setting-reasoning-effort'),
    badgeReasoningEffortVal: document.getElementById('badge-reasoning-effort-val'),
    sliderTemperature: document.getElementById('setting-temperature'),
    labelTemperatureVal: document.getElementById('label-temperature-val'),
    // Sarvam AI Controls
    selectSarvamModel: document.getElementById('setting-sarvam-model') || document.getElementById('setting-elevenlabs-model'),
    badgeSarvamTier: document.getElementById('badge-sarvam-tier') || document.getElementById('badge-eleven-tier'),
    selectSarvamSpeaker: document.getElementById('setting-sarvam-speaker') || document.getElementById('setting-elevenlabs-voice-preset'),
    selectSarvamLanguage: document.getElementById('setting-sarvam-language'),
    sliderSarvamPace: document.getElementById('setting-sarvam-pace'),
    labelSarvamPaceVal: document.getElementById('label-sarvam-pace-val'),
    inputSarvamTestText: document.getElementById('setting-sarvam-test-text') || document.getElementById('setting-elevenlabs-test-text'),
    btnTestSarvam: document.getElementById('btn-test-sarvam') || document.getElementById('btn-test-elevenlabs'),
    sarvamTestStatus: document.getElementById('sarvam-test-status') || document.getElementById('elevenlabs-test-status'),

    // Legacy Fallbacks
    selectElevenLabsModel: document.getElementById('setting-sarvam-model') || document.getElementById('setting-elevenlabs-model'),
    customElevenLabsModelGroup: document.getElementById('setting-custom-elevenlabs-model-group'),
    customElevenLabsModelInput: document.getElementById('setting-custom-elevenlabs-model'),
    badgeElevenTier: document.getElementById('badge-sarvam-tier') || document.getElementById('badge-eleven-tier'),
    selectElevenLabsVoicePreset: document.getElementById('setting-sarvam-speaker') || document.getElementById('setting-elevenlabs-voice-preset'),
    customElevenLabsVoiceGroup: document.getElementById('setting-custom-elevenlabs-voice-group'),
    inputElevenLabsVoice: document.getElementById('setting-elevenlabs-voice'),
    presetAccStudio: document.getElementById('preset-acc-studio'),
    presetAccMultilingual: document.getElementById('preset-acc-multilingual'),
    presetAccFast: document.getElementById('preset-acc-fast'),
    sliderElevenLabsStability: document.getElementById('setting-elevenlabs-stability'),
    labelStabilityVal: document.getElementById('label-stability-val'),
    sliderElevenLabsSimilarity: document.getElementById('setting-elevenlabs-similarity'),
    labelSimilarityVal: document.getElementById('label-similarity-val'),
    sliderElevenLabsStyle: document.getElementById('setting-elevenlabs-style'),
    labelStyleVal: document.getElementById('label-style-val'),
    chkElevenLabsSpeakerBoost: document.getElementById('setting-elevenlabs-speaker-boost'),
    inputElevenLabsTestText: document.getElementById('setting-sarvam-test-text') || document.getElementById('setting-elevenlabs-test-text'),
    btnTestElevenLabs: document.getElementById('btn-test-sarvam') || document.getElementById('btn-test-elevenlabs'),
    elevenLabsTestStatus: document.getElementById('sarvam-test-status') || document.getElementById('elevenlabs-test-status'),
    selectAudioMode: document.getElementById('setting-audio-mode'),
    customInstructions: document.getElementById('setting-instructions'),

    // STT & VAD Controls
    selectSttModel: document.getElementById('setting-stt-model'),
    badgeSttTier: document.getElementById('badge-stt-tier'),
    sliderVadSilence: document.getElementById('setting-vad-silence'),
    labelVadSilenceVal: document.getElementById('label-vad-silence-val'),
    sliderVadThreshold: document.getElementById('setting-vad-threshold'),
    labelVadThresholdVal: document.getElementById('label-vad-threshold-val'),
    sliderVadPrefix: document.getElementById('setting-vad-prefix'),
    labelVadPrefixVal: document.getElementById('label-vad-prefix-val'),
    btnVadSnappy: document.getElementById('btn-vad-snappy'),
    btnVadBalanced: document.getElementById('btn-vad-balanced'),
    btnVadPatient: document.getElementById('btn-vad-patient'),
    chkMicEcho: document.getElementById('setting-mic-echo'),
    chkBargeIn: document.getElementById('setting-barge-in'),
    chkMicNoise: document.getElementById('setting-mic-noise'),
    chkMicGain: document.getElementById('setting-mic-gain'),

    // TTS Playback Controls
    sliderTtsSpeed: document.getElementById('setting-tts-speed'),
    labelTtsSpeedVal: document.getElementById('label-tts-speed-val'),
    sliderTtsVolume: document.getElementById('setting-tts-volume'),
    labelTtsVolumeVal: document.getElementById('label-tts-volume-val'),
    selectTtsModality: document.getElementById('setting-tts-modality'),
    selectTtsDelivery: document.getElementById('setting-tts-delivery'),
    btnSpeedSlow: document.getElementById('btn-speed-slow'),
    btnSpeedNormal: document.getElementById('btn-speed-normal'),
    btnSpeedFast: document.getElementById('btn-speed-fast'),
    btnVolQuiet: document.getElementById('btn-vol-quiet'),
    btnVolStandard: document.getElementById('btn-vol-standard'),
    btnVolBoost: document.getElementById('btn-vol-boost'),

    // Call Recording & Transcript Elements
    recStatusPill: document.getElementById('rec-status-pill'),
    recPillTimer: document.getElementById('rec-pill-timer'),
    btnToggleRecord: document.getElementById('btn-toggle-record'),
    btnDownloadTxt: document.getElementById('btn-download-txt'),
    recModal: document.getElementById('rec-modal'),
    recModalBackdrop: document.getElementById('rec-modal-backdrop'),
    btnCloseRecModal: document.getElementById('btn-close-rec-modal'),
    btnRecModalClose: document.getElementById('btn-rec-modal-close'),
    recSummaryDuration: document.getElementById('rec-summary-duration'),
    recSummaryTurns: document.getElementById('rec-summary-turns'),
    recSummaryVoice: document.getElementById('rec-summary-voice'),
    recAudioSize: document.getElementById('rec-audio-size'),
    recTxtSize: document.getElementById('rec-txt-size'),
    recAudioElement: document.getElementById('rec-audio-element'),
    recTranscriptPre: document.getElementById('rec-transcript-pre'),
    recServerMsg: document.getElementById('rec-server-msg'),
    btnDownloadAudioFile: document.getElementById('btn-download-audio-file'),
    btnDownloadTxtFile: document.getElementById('btn-download-txt-file'),
    btnDownloadBoth: document.getElementById('btn-download-both'),

    // 7-Layer Architecture Strip & Modal Elements
    archPipelineStrip: document.getElementById('arch-pipeline-strip'),
    btnOpenArchModal: document.getElementById('btn-open-arch-modal'),
    archBackdrop: document.getElementById('arch-backdrop'),
    archModal: document.getElementById('arch-modal'),
    btnCloseArch: document.getElementById('btn-close-arch'),
    btnCloseArchFooter: document.getElementById('btn-close-arch-footer'),
    archModalGrid: document.getElementById('arch-modal-grid'),
    chipSttName: document.getElementById('chip-stt-name'),
    chipLlmName: document.getElementById('chip-llm-name'),
    chipVectorName: document.getElementById('chip-vector-name'),
    chipTtsName: document.getElementById('chip-tts-name'),
    badgeQdrantStatus: document.getElementById('badge-qdrant-status'),
    qdrantInfoText: document.getElementById('qdrant-info-text'),
    badgeMongoStatus: document.getElementById('badge-mongo-status'),
  };

  // ── Lifecycle & Initialization ────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', async () => {
    initVisualizerCanvas();
    setupEventListeners();
    setupVoiceSwitching();
    setupInstructionHandlers();
    setupVoiceTrainingHandlers();
    setupRecordingHandlers();
    updateArchStrip();
    await loadMainServerSettings();
    await loadInstructions();
    loadTrainingHistory();
    fetchOpenAiKeyStatus();
    syncVoiceUI(currentVoice);
    setOperatingMode('test'); // Default to Interactive Test & Voice Training mode
    setAppState('disconnected');
  });

  // ── State Management & UI Updates ─────────────────────────────────────────
  function setAppState(state) {
    // Called for every audio chunk / text delta; re-applying the same state would restyle the whole page
    if (state === currentState && state !== 'disconnected') return;
    currentState = state;
    document.body.className = `state-${state}`;

    switch (state) {
      case 'disconnected':
        elements.statusPill.className = 'status-pill';
        elements.statusText.textContent = 'Disconnected';
        elements.stateTitle.textContent = 'Ready to Connect';
        elements.stateHelper.textContent = 'Click "Start Session" to begin realtime voice chat';
        elements.btnMasterCall.className = 'btn-master-call';
        elements.btnCallText.textContent = 'Start Session';
        elements.liveSpeakerTag.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>
          Mintx AI Assistant
        `;
        elements.liveTranscriptText.textContent = 'Session idle. Press Start Session to talk.';
        elements.liveTranscriptText.classList.remove('typing');
        break;

      case 'connecting':
        elements.statusPill.className = 'status-pill connecting';
        elements.statusText.textContent = 'Connecting...';
        elements.stateTitle.textContent = 'Establishing Link';
        elements.stateHelper.textContent = 'Connecting to OpenAI Realtime WebSocket API...';
        elements.btnCallText.textContent = 'Connecting...';
        break;

      case 'idle':
        elements.statusPill.className = 'status-pill connected';
        elements.statusText.textContent = 'Live Session';
        elements.stateTitle.textContent = 'Listening for Speech';
        elements.stateHelper.textContent = 'Speak naturally in English or Tamil...';
        elements.btnMasterCall.className = 'btn-master-call connected';
        elements.btnCallText.textContent = 'End Session';
        elements.liveTranscriptText.classList.remove('typing');
        if (lastAssistantDisplayText) {
          elements.liveTranscriptText.textContent = lastAssistantDisplayText;
        }
        elements.liveSpeakerTag.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>
          Mintx AI Assistant
        `;
        break;

      case 'listening':
        elements.statusPill.className = 'status-pill connected';
        elements.statusText.textContent = 'User Speaking';
        elements.stateTitle.textContent = 'Hearing You...';
        elements.stateHelper.textContent = 'Recording your voice stream at 24kHz PCM';
        elements.liveSpeakerTag.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
          You (Voice)
        `;
        // Keep last AI subtitles visible while user speaks
        if (lastAssistantDisplayText && elements.liveTranscriptText) {
          elements.liveTranscriptText.textContent = lastAssistantDisplayText;
        }
        break;

      case 'thinking':
        elements.statusPill.className = 'status-pill connected';
        elements.statusText.textContent = 'Thinking...';
        elements.stateTitle.textContent = 'AI Processing';
        elements.stateHelper.textContent = 'Generating realtime response...';
        break;

      case 'speaking':
        elements.statusPill.className = 'status-pill connected';
        elements.statusText.textContent = 'AI Speaking';
        elements.stateTitle.textContent = 'Mintx AI Assistant';
        elements.stateHelper.textContent = 'Streaming voice reply aloud...';
        elements.liveSpeakerTag.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>
          Mintx AI Assistant
        `;
        elements.liveTranscriptText.classList.add('typing');
        break;
    }
  }

  // ── Web Audio Engine (PCM 24kHz Capture & Playback) ───────────────────────
  async function initAudioContext() {
    if (!audioContext) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      audioContext = new AudioContextClass({ sampleRate: 24000 });
    }
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    if (!speakerAnalyserNode) {
      speakerAnalyserNode = audioContext.createAnalyser();
      speakerAnalyserNode.fftSize = 64;
      speakerAnalyserNode.smoothingTimeConstant = 0.8;

      // Volume gain node for TTS output control
      speakerGainNode = audioContext.createGain();
      speakerGainNode.gain.value = currentTtsVolume / 100;
      speakerAnalyserNode.connect(speakerGainNode);
      speakerGainNode.connect(audioContext.destination);
    }

    // Two-way conversation audio recording mixer
    if (!recMixerNode) {
      recMixerNode = audioContext.createGain();
      recMixerNode.gain.value = 1.0;

      // 4096 frames = ~170ms chunks at 24000Hz
      recProcessorNode = audioContext.createScriptProcessor(4096, 1, 1);
      recMixerNode.connect(recProcessorNode);

      // Keep processor alive with silent sink to destination
      const recSink = audioContext.createGain();
      recSink.gain.value = 0;
      recProcessorNode.connect(recSink);
      recSink.connect(audioContext.destination);

      recProcessorNode.onaudioprocess = (e) => {
        if (!isActivelyRecording) return;
        const inputData = e.inputBuffer.getChannelData(0);
        recordedPcmChunks.push(new Float32Array(inputData));
      };
    }
  }

  function downsampleTo24k(buffer, sampleRate) {
    if (sampleRate === 24000) return buffer;
    const ratio = sampleRate / 24000;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const origIndex = i * ratio;
      const indexLow = Math.floor(origIndex);
      const indexHigh = Math.min(indexLow + 1, buffer.length - 1);
      const interpolation = origIndex - indexLow;
      result[i] = buffer[indexLow] * (1 - interpolation) + buffer[indexHigh] * interpolation;
    }
    return result;
  }

  function encodeMicChunk(inputData) {
    const resampled = downsampleTo24k(inputData, audioContext.sampleRate);
    const pcm16 = new Int16Array(resampled.length);
    for (let i = 0; i < resampled.length; i++) {
      const s = Math.max(-1, Math.min(1, resampled[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return arrayBufferToBase64(pcm16.buffer);
  }

  const speakerSampleBuf = new Float32Array(64);
  function currentSpeakerRms() {
    if (!speakerAnalyserNode) return 0;
    speakerAnalyserNode.getFloatTimeDomainData(speakerSampleBuf);
    let sum = 0;
    for (let i = 0; i < speakerSampleBuf.length; i++) sum += speakerSampleBuf[i] * speakerSampleBuf[i];
    return Math.sqrt(sum / speakerSampleBuf.length);
  }

  function triggerBargeIn() {
    const preroll = bargeInPreroll;
    bargeInPreroll = [];
    bargeInLoudCount = 0;
    stopAllPlayback();
    // Audio already in flight from the interrupted reply must not restart playback
    dropAiAudioUntil = performance.now() + BARGE_IN_DROP_AUDIO_MS;
    sendWsEvent({ type: 'client.barge_in' });
    // Send the start of the caller's sentence that was captured while the AI was talking
    preroll.forEach((audio) => sendWsEvent({ type: 'input_audio_buffer.append', audio }));
    setAppState('listening');
  }

  async function startMicrophone() {
    try {
      await initAudioContext();

      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: currentMicEcho || currentModel === LIVE_MODEL,
          noiseSuppression: currentMicNoise,
          autoGainControl: currentMicGain,
        },
      });

      micSourceNode = audioContext.createMediaStreamSource(micStream);

      // Mic Analyser for Visualizer
      micAnalyserNode = audioContext.createAnalyser();
      micAnalyserNode.fftSize = 64;
      micAnalyserNode.smoothingTimeConstant = 0.7;
      micSourceNode.connect(micAnalyserNode);

      // ScriptProcessor for PCM Capture
      const bufferSize = 2048;
      micProcessorNode = audioContext.createScriptProcessor(bufferSize, 1, 1);

      micProcessorNode.onaudioprocess = (e) => {
        if (!isConnected || isMuted || audioMode !== 'browser') return;

        const inputData = e.inputBuffer.getChannelData(0);

        let sum = 0;
        for (let i = 0; i < inputData.length; i += 4) {
          sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / (inputData.length / 4));

        // Echo shield: while AI speaks, mic audio is held back so the AI never hears its own voice.
        // GPT-Live is full duplex and streams audio continuously, so the mic must always flow.
        if (isAiSpeaking && currentModel !== LIVE_MODEL) {
          if (!currentBargeIn) return;
          bargeInPreroll.push(encodeMicChunk(inputData));
          if (bargeInPreroll.length > BARGE_IN_PREROLL_CHUNKS) bargeInPreroll.shift();
          const loud = rms > BARGE_IN_RMS && rms > currentSpeakerRms() * BARGE_IN_SPEAKER_RATIO;
          bargeInLoudCount = loud ? bargeInLoudCount + 1 : 0;
          if (bargeInLoudCount < BARGE_IN_LOUD_CHUNKS) return;
          triggerBargeIn();
          return;
        }
        bargeInLoudCount = 0;
        bargeInPreroll = [];
        const callerSpeaking = rms > 0.018;
        if (callerSpeaking !== lastCallerSpeaking) {
          lastCallerSpeaking = callerSpeaking;
          if (elements.sideCaller) elements.sideCaller.classList.toggle('speaking', callerSpeaking);
          if (elements.callerMicStatus) elements.callerMicStatus.textContent = callerSpeaking ? 'Speaking...' : 'Mic Active';
          if (elements.callerLiveIndicator) elements.callerLiveIndicator.textContent = callerSpeaking ? 'SPEAKING' : 'LIVE MIC';
        }

        sendWsEvent({
          type: 'input_audio_buffer.append',
          audio: encodeMicChunk(inputData),
        });
      };

      // Mute node to prevent mic from feeding back through user speakers
      const muteNode = audioContext.createGain();
      muteNode.gain.value = 0;
      micSourceNode.connect(micProcessorNode);
      micProcessorNode.connect(muteNode);
      muteNode.connect(audioContext.destination);

      // Connect mic to recording mixer so user speech is captured
      if (recMixerNode) {
        micSourceNode.connect(recMixerNode);
      }
    } catch (err) {
      console.error('[Microphone Error]:', err);
      alert(`Could not access microphone: ${err.message}\nMake sure microphone permission is allowed.`);
    }
  }

  function stopMicrophone() {
    if (micStream) {
      micStream.getTracks().forEach((track) => track.stop());
      micStream = null;
    }
    if (micSourceNode) {
      try { micSourceNode.disconnect(); } catch (e) {}
      micSourceNode = null;
    }
    if (micProcessorNode) {
      try { micProcessorNode.disconnect(); } catch (e) {}
      micProcessorNode = null;
    }
  }

  function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
  }

  function base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }

  // ── Smooth PCM 24kHz Audio Playback Queue ────────────────────────────────
  function playAudioChunk(base64Pcm) {
    if (!audioContext || audioMode !== 'browser') return;

    if (audioContext.state === 'suspended') {
      audioContext.resume();
    }

    // Active AI speech playback: engage echo shield
    isAiSpeaking = true;
    if (speechCooldownTimer) {
      clearTimeout(speechCooldownTimer);
      speechCooldownTimer = null;
    }

    try {
      const incoming = new Uint8Array(base64ToArrayBuffer(base64Pcm));
      // Align to 16-bit samples — odd leftover bytes cause crackle / broken voice
      const merged = new Uint8Array(pcmByteCarry.length + incoming.length);
      merged.set(pcmByteCarry, 0);
      merged.set(incoming, pcmByteCarry.length);
      const evenLen = merged.length - (merged.length % 2);
      pcmByteCarry = evenLen < merged.length ? merged.slice(evenLen) : new Uint8Array(0);
      if (evenLen < 2) return;

      const usable = merged.slice(0, evenLen);
      const pcm16 = new Int16Array(usable.buffer, usable.byteOffset, usable.byteLength / 2);
      const float32 = new Float32Array(pcm16.length);

      for (let i = 0; i < pcm16.length; i++) {
        float32[i] = pcm16[i] / 32768.0;
      }

      const audioBuffer = audioContext.createBuffer(1, float32.length, 24000);
      audioBuffer.getChannelData(0).set(float32);

      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      // Sarvam AI speed is set server-side via pace; resampling here would only shift the pitch
      // GPT-Live streams in real time; any other rate would make the playback queue drift.
      const playRate = (isSarvamVoiceId(currentVoice) || currentModel === LIVE_MODEL) ? 1.0 : Math.max(0.85, Math.min(1.05, currentTtsSpeed || 1.0));
      source.playbackRate.value = playRate;
      source.connect(speakerAnalyserNode);

      // Connect assistant speech to recording mixer
      if (recMixerNode) {
        source.connect(recMixerNode);
      }

      // Jitter buffer: when playback starts (or the queue ran dry) schedule slightly ahead,
      // so chunks arriving a little late join seamlessly instead of stuttering.
      const currentTime = audioContext.currentTime;
      if (nextPlayTime < currentTime + 0.02) {
        nextPlayTime = currentTime + PLAYBACK_JITTER_BUFFER_SEC;
      }
      source.start(nextPlayTime);
      nextPlayTime += audioBuffer.duration / playRate;

      activeAudioSources.push(source);
      source.onended = () => {
        const index = activeAudioSources.indexOf(source);
        if (index > -1) activeAudioSources.splice(index, 1);
        if (activeAudioSources.length === 0) {
          // Playback finished: wait 450ms for room echo / speaker reverb to dissipate before enabling mic
          if (speechCooldownTimer) clearTimeout(speechCooldownTimer);
          speechCooldownTimer = setTimeout(() => {
            if (activeAudioSources.length === 0) {
              isAiSpeaking = false;
              if (currentState === 'speaking') {
                setAppState('idle');
              }
            }
          }, 450);
        }
      };

      setAppState('speaking');
    } catch (err) {
      console.error('[Playback Error]:', err);
    }
  }

  function stopAllPlayback() {
    activeAudioSources.forEach((src) => {
      try { src.stop(); } catch (e) {}
    });
    activeAudioSources = [];
    pcmByteCarry = new Uint8Array(0);
    if (speechCooldownTimer) {
      clearTimeout(speechCooldownTimer);
      speechCooldownTimer = null;
    }
    isAiSpeaking = false;
    if (audioContext) {
      nextPlayTime = audioContext.currentTime;
    }
    if (currentState === 'speaking') {
      setAppState('idle');
    }
  }

  // ── WebSocket Connection to FastAPI Proxy ────────────────────────────────
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/realtime?model=${encodeURIComponent(currentModel)}`;

    setAppState('connecting');

    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = async () => {
        console.log(`[WebSocket Connected] Model: ${currentModel} | Reasoning: ${currentReasoningEffort}`);
        isConnected = true;
        setAppState('idle');

        // Send initial session config with voice, model, accuracy, STT, TTS, and instructions
        sendWsEvent({
          type: 'client.init',
          voice: currentVoice,
          model: currentModel,
          audio_mode: audioMode,
          reasoning_effort: currentReasoningEffort,
          temperature: currentTemperature,
          stt_model: currentSttModel,
          vad_threshold: currentVadThreshold,
          vad_silence_ms: currentVadSilenceMs,
          vad_prefix_ms: currentVadPrefixMs,
          // Sarvam AI voice => OpenAI text output; GPT voice => OpenAI native audio
          modalities: isSarvamVoiceId(currentVoice) ? ['text'] : ['audio'],
          auto_greet: true,
          starting_greeting: (activeInstruction && activeInstruction.starting_greeting)
            || 'Vanakkam! MintX Solutions-ku welcome. Enna service venum nu sollunga?',
          tts_provider: isSarvamVoiceId(currentVoice) ? 'sarvam' : 'openai',
          sarvam_speaker: currentSarvamSpeaker || 'kavitha',
          sarvam_model: currentSarvamModel || 'bulbul:v3',
          sarvam_pace: currentSarvamPace || DEFAULT_SARVAM_PACE,
          sarvam_language_code: currentSarvamLanguage || 'ta-IN',
          instructions: (activeInstruction ? activeInstruction.instructions : ''),
        });
        localStorage.setItem('preferred_voice', currentVoice);

        if (audioMode === 'browser') {
          await startMicrophone();
        }

        // Auto-start 2-way call recording when session starts
        if (isConversationRecordingEnabled) {
          startCallRecording();
        }

        // If a message was sent before connection completed, transmit it cleanly now
        if (pendingTextMessage) {
          const queuedText = pendingTextMessage;
          pendingTextMessage = null;
          setTimeout(() => transmitUserText(queuedText), 200);
        }
      };

      ws.onmessage = (event) => {
        handleWsMessage(event.data);
      };

      ws.onerror = (err) => {
        console.error('[WebSocket Error]:', err);
        elements.statusPill.className = 'status-pill error';
        elements.statusText.textContent = 'Connection Error';
      };

      ws.onclose = () => {
        console.log('[WebSocket Closed]');
        disconnectSession();
      };
    } catch (err) {
      console.error('[WebSocket Connection Failed]:', err);
      disconnectSession();
    }
  }

  function sendWsEvent(eventObj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(eventObj));
    }
  }

  function disconnectSession() {
    isConnected = false;
    stopMicrophone();
    stopAllPlayback();

    // Finalize conversation audio recording and formatted transcript
    if (isActivelyRecording) {
      stopCallRecording();
    }

    if (ws) {
      try { ws.close(); } catch (e) {}
      ws = null;
    }

    if (elements.callerLangBadge) {
      elements.callerLangBadge.textContent = '🌐 Language: Auto-Detect';
      elements.callerLangBadge.style.background = 'rgba(16, 185, 129, 0.15)';
      elements.callerLangBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      elements.callerLangBadge.style.color = '#34d399';
    }

    setAppState('disconnected');
  }

  // Helper to ensure 'MintX Solutions' is NEVER decoded/displayed in Tamil script
  function normalizeBrandName(text) {
    if (!text) return text;
    let t = String(text);
    t = t.replace(/(?:மி[ன்ண்]ட்|மை[ன்ண்]ட்)[\s\-_]*(?:எக்ஸ்|க்ஸ்)[\s\-_]*(?:சொல்[யூயு][ஷச]ன்|சால்யூஷன்|சொலுஷன்)[ஸ்]*(?:(?:ு|்)?க்கு)?/gi, (m) => m.includes('க்கு') ? 'MintX Solutions-ku' : 'MintX Solutions');
    t = t.replace(/(?:மி[ன்ண்]டெக்ஸ்|மி[ன்ண்]டக்ஸ்|மை[ன்ண்]டெக்ஸ்|மின்தெக்ஸ்)[\s\-_]*(?:சொல்[யூயு][ஷச]ன்|சால்யூஷன்|சொலுஷன்)[ஸ்]*(?:(?:ு|்)?க்கு)?/gi, (m) => m.includes('க்கு') ? 'MintX Solutions-ku' : 'MintX Solutions');
    t = t.replace(/\bMint[\s\-_]*X\b[\s\-_]*(?:சொல்[யூயு][ஷச]ன்|சால்யூஷன்|சொலுஷன்)[ஸ்]*(?:(?:ு|்)?க்கு)?/gi, (m) => m.includes('க்கு') ? 'MintX Solutions-ku' : 'MintX Solutions');
    t = t.replace(/(?:மி[ன்ண்]ட்|மை[ன்ண்]ட்)[\s\-_]*(?:எக்ஸ்|க்ஸ்)(?:(?:ு|்)?க்கு)?/gi, (m) => m.includes('க்கு') ? 'MintX-ku' : 'MintX');
    t = t.replace(/(?:மி[ன்ண்]டெக்ஸ்|மி[ன்ண்]டக்ஸ்|மை[ன்ண்]டெக்ஸ்|மின்தெக்ஸ்)(?:(?:ு|்)?க்கு)?/gi, (m) => m.includes('க்கு') ? 'MintX-ku' : 'MintX');
    t = t.replace(/\b(?:mint|mind|min)[\s\-_]*x\s+solutions?\b/gi, 'MintX Solutions');
    t = t.replace(/\bmintex\s+solutions?\b/gi, 'MintX Solutions');
    t = t.replace(/\bmindx\s+solutions?\b/gi, 'MintX Solutions');
    t = t.replace(/\b(?:mint|mind|min)[\s\-_]+x\b/gi, 'MintX');
    t = t.replace(/\b(?:mindx|mintex|minx)\b/gi, 'MintX');
    t = t.replace(/\bmintx\b/gi, 'MintX');
    t = t.replace(/\bMintX Solutions[\s\-_]*(?:கு|க்கு)(?=[\s,.!?;:\"'”’…]|$)/g, 'MintX Solutions-ku');
    t = t.replace(/\bMintX[\s\-_]*(?:கு|க்கு)(?=[\s,.!?;:\"'”’…]|$)/g, 'MintX-ku');
    return t;
  }

  // ── WebSocket Event Handling ──────────────────────────────────────────────
  function handleWsMessage(rawMsg) {
    let data;
    try {
      data = JSON.parse(rawMsg);
    } catch (e) {
      return;
    }

    const type = data.type;

    // 1. Audio output chunks from OpenAI
    if (type === 'response.output_audio.delta' || type === 'response.audio.delta') {
      const deltaB64 = data.delta || '';
      if (deltaB64 && performance.now() >= dropAiAudioUntil) {
        playAudioChunk(deltaB64);
      }
    }

    // 1b. Tamil-script display for Tanglish speech (letters in Tamil, spoken style stays Tanglish)
    else if (type === 'assistant.transcript.display') {
      const displayText = normalizeBrandName(data.text || '');
      currentAssistantTranscript = displayText;
      lastAssistantDisplayText = displayText;
      if (elements.liveTranscriptText) {
        elements.liveTranscriptText.textContent = displayText || '…';
      }
      setAppState('speaking');
      if (data.final && displayText.trim()) {
        addMessageToHistory('assistant', displayText);
        // Keep subtitles visible — do NOT clear the live transcript box
        currentAssistantTranscript = '';
      }
    }

    // 2. Assistant spoken text transcript delta (fallback / greeting)
    else if (
      type === 'response.output_audio_transcript.delta' ||
      type === 'response.audio_transcript.delta' ||
      type === 'response.text.delta' ||
      type === 'response.output_text.delta'
    ) {
      const delta = data.delta || '';
      currentAssistantTranscript = normalizeBrandName(currentAssistantTranscript + delta);
      lastAssistantDisplayText = currentAssistantTranscript;
      if (elements.liveTranscriptText) {
        elements.liveTranscriptText.textContent = currentAssistantTranscript;
      }
      setAppState('speaking');
    }

    // 3. Assistant response completed
    else if (
      type === 'response.output_audio_transcript.done' ||
      type === 'response.audio_transcript.done' ||
      type === 'response.done'
    ) {
      // Prefer Tamil display event (assistant.transcript.display final=true).
      // Fallback for greeting deltas that already arrived in Tamil script.
      if (currentAssistantTranscript.trim()) {
        lastAssistantDisplayText = normalizeBrandName(currentAssistantTranscript);
        addMessageToHistory('assistant', lastAssistantDisplayText);
        currentAssistantTranscript = '';
      }
      // Keep last spoken line visible in the Mintx AI Assistant box
      if (elements.liveTranscriptText && lastAssistantDisplayText) {
        elements.liveTranscriptText.textContent = lastAssistantDisplayText;
      }
      if (elements.liveTranscriptText) {
        elements.liveTranscriptText.classList.remove('typing');
      }
      if (currentModel === LIVE_MODEL && isConnected) {
        setAppState('listening');
      }
    }

    // 4. User speech transcription completed
    else if (
      type === 'conversation.item.input_audio_transcription.completed' ||
      type === 'input_audio_transcription.completed'
    ) {
      const transcript = normalizeBrandName((data.transcript || '').trim());
      if (transcript) {
        if (elements.callerTranscriptText) {
          elements.callerTranscriptText.textContent = `"${transcript}"`;
        }
        addMessageToHistory('user', transcript);
        setAppState('thinking');

        // Check if Auto-Detect Spoken Corrections is enabled
        if (
          !elements.toggleAutoDetectCorrections ||
          elements.toggleAutoDetectCorrections.checked
        ) {
          const lower = transcript.toLowerCase();
          const correctionTriggers = [
            'correction', 'கரெக்ஷன்', 'மாத்து', 'மாற்று', 'ரூல்', 'விதி', 'change', 'rule',
            'starting greeting', 'starting word', 'starting phrase', 'greeting-a', 'start every call',
            'இப்படி சொல்லு', 'இப்படி சொல்லாதே', 'அப்படி சொல்லாத', "don't say", "dont say",
            'update rule', 'change rule', 'say this instead', 'விலை', 'price', 'whatsapp', 'வாட்ஸ்அப்',
            'கேள்வி', 'question', 'கேளு', 'fix pannu', 'fix பண்ணு', 'opening greeting', 'welcome message',
            'welcome சொல்லு', 'vanakkam sollu', 'வணக்கம் சொல்லு', 'டெவலப்பர்', 'developer',
            'மட்டும் பேசு', 'speak only', 'mix பண்ணி பேசாத', 'train', 'ட்ரெயின்', 'training', 'அப்டேட் பண்ணு'
          ];
          const hasTrigger = correctionTriggers.some(t => lower.includes(t.toLowerCase()));
          if (hasTrigger) {
            console.log('[Auto-Correction Trigger Detected in Call]:', transcript);
            // Silence AI immediately so it does not speak customer questions over the correction
            stopAllPlayback();
            sendWsEvent({ type: 'response.cancel' });
            applyVoiceCorrectionText(transcript, currentAssistantTranscript);
            showToast(`🎙️ Spoken correction recognized & self-training: "${transcript}"`);
          }
        }
      }
    }

    // 4b. Customer language auto-detected in real-time
    else if (type === 'customer.language.detected') {
      const lang = data.language || 'en';
      const displayName = data.display_name || (lang === 'en' ? 'English' : 'Tamil (Tanglish)');
      if (elements.callerLangBadge) {
        elements.callerLangBadge.textContent = `🌐 ${displayName}`;
        elements.callerLangBadge.style.background = lang === 'en' ? 'rgba(59, 130, 246, 0.2)' : 'rgba(168, 85, 247, 0.2)';
        elements.callerLangBadge.style.borderColor = lang === 'en' ? 'rgba(59, 130, 246, 0.6)' : 'rgba(168, 85, 247, 0.6)';
        elements.callerLangBadge.style.color = lang === 'en' ? '#60a5fa' : '#c084fc';
      }
    }

    // 5. Speech VAD Detection Events
    else if (type === 'input_audio_buffer.speech_started') {
      // Caller started talking: cut off any AI audio still queued in the browser
      if (activeAudioSources.length > 0) {
        stopAllPlayback();
        dropAiAudioUntil = performance.now() + BARGE_IN_DROP_AUDIO_MS;
      }
      if (!isAiSpeaking) {
        setAppState('listening');
      }
    } else if (type === 'input_audio_buffer.speech_stopped') {
      if (!isAiSpeaking) {
        setAppState('thinking');
      }
    } else if (type === 'caller.transcript.partial') {
      if (elements.callerTranscriptText && data.text) {
        elements.callerTranscriptText.textContent = `"${data.text}…"`;
      }
    } else if (type === 'stt.engine') {
      console.log('[STT] Caller speech engine:', data.engine);
      if (data.message) showToast(`⚠️ ${data.message}`);
    }

    // 6. Ping / Telemetry
    else if (type === 'telemetry.ping') {
      if (data.latency_ms !== undefined) {
        elements.pingMetric.textContent = `${data.latency_ms}ms`;
      }
    }

    // 6a-0. Sarvam voice unavailable (e.g. no credits): continue the call with a GPT voice.
    // Not saved as the preferred voice, so Sarvam comes back automatically after a reload.
    else if (type === 'voice.fallback') {
      currentVoice = data.voice || 'marin';
      syncVoiceUI(currentVoice);
      showToast(`⚠️ ${data.message || 'Switched to GPT voice'}`);
      disconnectSession();
      setTimeout(async () => {
        await initAudioContext();
        connectWebSocket();
      }, 400);
    }

    // 6a. GPT voice already spoke in this session; OpenAI needs a fresh session for a different GPT voice
    else if (type === 'voice.reconnect_required') {
      const info = VOICE_INFO[currentVoice] || { name: currentVoice };
      showToast(`🔄 Restarting session with ${info.name} voice...`);
      disconnectSession();
      setTimeout(async () => {
        await initAudioContext();
        connectWebSocket();
      }, 400);
    }

    // 6b. Live Correction / Self-Training Applied by ChatGPT Tool Call
    else if (type === 'correction.applied') {
      console.log('[WebSocket Event] Live Self-Training Correction Applied:', data);
      handleCorrectionApplied(data, data.feedback || '', true);
      showToast(`🧠 ChatGPT Self-Trained: "${data.summary_en || 'Rule updated'}"`);
    }

    // 7. Error Handling
    else if (type === 'error') {
      console.warn('[OpenAI Realtime Error]:', data.error);
      const errMsg = data.error?.message || JSON.stringify(data.error);
      elements.stateHelper.textContent = `Error: ${errMsg}`;

      const isQuota = data.error?.code === 'insufficient_quota' ||
                      errMsg.includes('credit_balance_exhausted') ||
                      errMsg.includes('insufficient_quota') ||
                      errMsg.includes('1013');

      if (isQuota) {
        const banner = document.getElementById('quota-alert-banner');
        if (banner) banner.style.display = 'flex';
        showToast('⚠️ OpenAI Account Credit Balance Exhausted ($0.00). Please recharge credits or enter a new API key in Settings.');
        if (elements.statusPill) {
          elements.statusPill.className = 'status-pill error';
          elements.statusText.textContent = 'Quota Exhausted';
        }
      } else {
        showToast(`❌ ${errMsg}`);
      }
    }
  }

  // ── Conversation History Management ───────────────────────────────────────
  function addMessageToHistory(role, text) {
    if (!text || !text.trim()) return;

    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    conversationHistory.push({ role, text, time: timeStr });

    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;

    const isTamil = /[\u0B80-\u0BFF]/.test(text);
    const langBadge = isTamil ? ' • Tamil' : '';

    let actionsHtml = '';
    if (role === 'assistant') {
      actionsHtml = `
        <div class="chat-bubble-actions">
          <button type="button" class="btn-bubble-correct" title="Speak a voice correction for this response">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
            🎙️ Correct Response
          </button>
        </div>
      `;
    }

    bubble.innerHTML = `
      <div class="bubble-meta">
        ${role === 'assistant' ? 'Mintx AI' : 'You'}${langBadge} • ${timeStr}
      </div>
      <div class="bubble-content">${escapeHtml(text)}</div>
      ${actionsHtml}
    `;

    const correctBtn = bubble.querySelector('.btn-bubble-correct');
    if (correctBtn) {
      correctBtn.addEventListener('click', () => {
        openResponseCorrectionModal(text);
      });
    }

    elements.drawerBody.appendChild(bubble);
    elements.drawerBody.scrollTop = elements.drawerBody.scrollHeight;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ── Visualizer Canvas Animation ───────────────────────────────────────────
  function initVisualizerCanvas() {
    canvas = elements.visualizerCanvas;
    if (!canvas) return;
    ctx = canvas.getContext('2d');

    // Handle high-DPI displays
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const rect = canvas.getBoundingClientRect();
    canvas.width = (rect.width || 440) * dpr;
    canvas.height = (rect.height || 440) * dpr;
    ctx.scale(dpr, dpr);

    startVisualizerLoop(dpr);
  }

  function startVisualizerLoop(dpr) {
    const width = canvas.width / dpr;
    const height = canvas.height / dpr;
    const centerX = width / 2;
    const centerY = height / 2;

    let angleOffset = 0;
    const freqData = new Uint8Array(32);

    function averageLevel(analyser) {
      analyser.getByteFrequencyData(freqData);
      let sum = 0;
      for (let i = 0; i < freqData.length; i++) sum += freqData[i];
      return sum / freqData.length / 255;
    }

    function render(time) {
      animFrameId = requestAnimationFrame(render);

      ctx.clearRect(0, 0, width, height);

      const micVol = (micAnalyserNode && !isMuted) ? averageLevel(micAnalyserNode) : 0;
      currentMicVolume += (micVol - currentMicVolume) * 0.25;

      const spkVol = (speakerAnalyserNode && activeAudioSources.length > 0) ? averageLevel(speakerAnalyserNode) : 0;
      currentSpeakerVolume += (spkVol - currentSpeakerVolume) * 0.25;

      const activeVol = currentState === 'speaking' ? currentSpeakerVolume : currentMicVolume;
      angleOffset += 0.015 + activeVol * 0.05;

      // Base radius of the inner orb
      const baseRadius = 88;

      // Draw multi-layered glowing wave rings
      const rings = [
        { scale: 1.05 + activeVol * 0.35, alpha: 0.45 + activeVol * 0.4, color: getRingColor(0) },
        { scale: 1.25 + activeVol * 0.55, alpha: 0.25 + activeVol * 0.3, color: getRingColor(1) },
        { scale: 1.48 + activeVol * 0.75, alpha: 0.12 + activeVol * 0.2, color: getRingColor(2) },
      ];

      rings.forEach((ring, idx) => {
        ctx.beginPath();
        const segments = 36;
        for (let i = 0; i <= segments; i++) {
          const theta = (i / segments) * Math.PI * 2;
          const wave = Math.sin(theta * 4 + angleOffset * (idx + 1)) * (4 + activeVol * 16);
          const r = baseRadius * ring.scale + wave;
          const x = centerX + Math.cos(theta) * r;
          const y = centerY + Math.sin(theta) * r;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();

        // Soft glow as a wide translucent stroke — canvas shadowBlur is far too slow per frame
        ctx.strokeStyle = ring.color;
        ctx.globalAlpha = 0.18;
        ctx.lineWidth = 10 + activeVol * 14;
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.lineWidth = 2 + activeVol * 3;
        ctx.stroke();
      });
    }

    render(0);
  }

  function getRingColor(index) {
    if (currentState === 'speaking') {
      // Emerald / Mint Neon
      return index === 0 ? 'rgba(16, 185, 129, 0.8)' : index === 1 ? 'rgba(6, 182, 212, 0.5)' : 'rgba(52, 211, 153, 0.25)';
    } else if (currentState === 'listening') {
      // Vivid Sky Blue
      return index === 0 ? 'rgba(59, 130, 246, 0.85)' : index === 1 ? 'rgba(14, 165, 233, 0.5)' : 'rgba(96, 165, 250, 0.25)';
    } else if (currentState === 'thinking') {
      // Purple / Violet Glow
      return index === 0 ? 'rgba(168, 85, 247, 0.85)' : index === 1 ? 'rgba(192, 132, 252, 0.5)' : 'rgba(216, 180, 254, 0.25)';
    }
    // Idle Cyan
    return index === 0 ? 'rgba(6, 182, 212, 0.4)' : index === 1 ? 'rgba(59, 130, 246, 0.2)' : 'rgba(99, 102, 241, 0.1)';
  }

  // ── Voice Switching Management ────────────────────────────────────────────

  // ── Voice Switching Management ────────────────────────────────────────────

  function syncVoiceUI(voiceId) {
    if (!voiceId) return;
    const sarvamVoices = ['sarvam', 'kavitha', 'kavya', 'priya', 'shreya', 'roopa', 'mani', 'gokul', 'vijay'];
    const isSarvam = sarvamVoices.includes(voiceId) || voiceId === 'elevenlabs';
    const effectiveVoice = (voiceId === 'elevenlabs') ? 'sarvam' : voiceId;
    const info = VOICE_INFO[effectiveVoice] || { name: effectiveVoice.charAt(0).toUpperCase() + effectiveVoice.slice(1), tone: 'Realtime Voice', gender: '', emoji: '🎙️' };

    // 1. Sync Top Nav Select
    if (elements.navVoiceSelect) {
      if (elements.navVoiceSelect.querySelector(`option[value="${effectiveVoice}"]`)) {
        elements.navVoiceSelect.value = effectiveVoice;
      } else if (isSarvam) {
        elements.navVoiceSelect.value = 'sarvam';
      }
    }

    // 2. Sync Side B Select
    if (elements.sideVoiceSelect) {
      if (elements.sideVoiceSelect.querySelector(`option[value="${effectiveVoice}"]`)) {
        elements.sideVoiceSelect.value = effectiveVoice;
      } else if (isSarvam) {
        elements.sideVoiceSelect.value = 'sarvam';
      }
    }

    // 3. Sync Settings Modal Select
    if (elements.selectVoice) {
      if (elements.selectVoice.querySelector(`option[value="${effectiveVoice}"]`)) {
        elements.selectVoice.value = effectiveVoice;
      } else if (isSarvam) {
        elements.selectVoice.value = 'sarvam';
      }
    }

    // 4. Sync Settings Modal Sarvam Speaker preset
    if (elements.selectSarvamSpeaker) {
      const targetSpeaker = (effectiveVoice !== 'sarvam' && sarvamVoices.includes(effectiveVoice)) ? effectiveVoice : currentSarvamSpeaker;
      if (elements.selectSarvamSpeaker.querySelector(`option[value="${targetSpeaker}"]`)) {
        elements.selectSarvamSpeaker.value = targetSpeaker;
      }
    }

    // 5. Sync Legacy Badge if present
    if (elements.voiceBadge) {
      elements.voiceBadge.textContent = info.name.toUpperCase();
    }

    if (elements.btnVoiceEngines) {
      elements.btnVoiceEngines.forEach((btn) => {
        btn.classList.toggle('active', btn.getAttribute('data-engine') === (isSarvam ? 'sarvam' : 'gpt'));
      });
    }

    // 6. Sync Quick Voice Active Label
    if (elements.quickVoiceActiveLabel) {
      elements.quickVoiceActiveLabel.innerHTML = `Active: <strong>${info.name}</strong> (${info.tone})`;
    }

    // 7. Sync Quick Voice Chips active state
    if (elements.btnVoiceChips) {
      elements.btnVoiceChips.forEach((btn) => {
        const chipVoice = btn.getAttribute('data-voice');
        if (chipVoice === effectiveVoice || (isSarvam && chipVoice === 'sarvam')) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });
    }

    // 8. Sync Dropdown Menu items active state
    if (elements.voiceMenuItems) {
      elements.voiceMenuItems.forEach((item) => {
        const itemVoice = item.getAttribute('data-voice');
        if (itemVoice === effectiveVoice || (isSarvam && itemVoice === 'sarvam')) {
          item.classList.add('active');
        } else {
          item.classList.remove('active');
        }
      });
    }

    if (elements.chipTtsName) elements.chipTtsName.textContent = `GPT-Live voice (${effectiveVoice})`;
    if (elements.summaryTtsEngine) elements.summaryTtsEngine.textContent = `GPT-Live voice (${effectiveVoice})`;
  }

  function switchVoice(voiceId, notify = true) {
    if (!voiceId) return;
    voiceId = toLiveVoice(voiceId);

    const sarvamVoices = ['sarvam', 'kavitha', 'kavya', 'priya', 'shreya', 'roopa', 'mani', 'gokul', 'vijay'];
    const isSarvam = sarvamVoices.includes(voiceId) || voiceId === 'elevenlabs';

    if (isSarvam) {
      currentVoice = (voiceId === 'elevenlabs') ? 'sarvam' : voiceId;
      if (voiceId !== 'sarvam' && voiceId !== 'elevenlabs') {
        currentSarvamSpeaker = voiceId;
        localStorage.setItem('preferred_sarvam_speaker', voiceId);
      }
      currentTtsModality = 'text';
      localStorage.setItem('preferred_tts_modality', 'text');
      if (elements.selectTtsModality) elements.selectTtsModality.value = 'text';
    } else {
      currentVoice = voiceId;
      lastGptVoice = voiceId;
      localStorage.setItem('preferred_gpt_voice', voiceId);
    }

    localStorage.setItem('preferred_voice', currentVoice);
    syncVoiceUI(currentVoice);

    const info = VOICE_INFO[currentVoice] || { name: currentVoice, tone: 'Voice' };

    // Cut off ongoing speech so the new voice takes effect immediately
    stopAllPlayback();

    // GPT-Live fixes the voice at session start, so a live call restarts with the new voice.
    if (currentModel === LIVE_MODEL) {
      if (isConnected) {
        showToast(`🔄 Restarting call with ${info.name} voice...`);
        disconnectSession();
        setTimeout(async () => {
          await initAudioContext();
          connectWebSocket();
        }, 400);
      } else if (notify) {
        showToast(`🎙️ Switched Voice to ${info.name} (${info.tone})`);
      }
      return;
    }

    // If connected to OpenAI Realtime WebSocket live session, hot-reload voice immediately!
    if (isConnected && ws && ws.readyState === WebSocket.OPEN) {
      // If AI was speaking, cancel current in-flight response so old voice stops immediately
      if (currentState === 'speaking') {
        sendWsEvent({ type: 'response.cancel' });
      }

      sendWsEvent({
        type: 'client.switch_voice',
        voice: currentVoice,
        model: currentModel,
        instructions: (elements.customInstructions && elements.customInstructions.value.trim()) ? elements.customInstructions.value.trim() : (activeInstruction ? activeInstruction.instructions : ''),
        tts_provider: isSarvam ? 'sarvam' : 'openai',
        sarvam_speaker: currentSarvamSpeaker,
        sarvam_model: currentSarvamModel,
        sarvam_pace: currentSarvamPace,
        sarvam_language_code: currentSarvamLanguage,
      });
      console.log(`[WebSocket] Hot-switched voice to: ${currentVoice} (Sarvam Speaker: ${currentSarvamSpeaker})`);
    }

    if (notify) {
      showToast(`🎙️ Switched Voice to ${info.name} (${info.tone})`);
    }
  }

  function setupVoiceSwitching() {
    // 1. Top Nav Voice Select
    if (elements.navVoiceSelect) {
      elements.navVoiceSelect.addEventListener('change', (e) => {
        switchVoice(e.target.value);
      });
    }

    // 2. Side B Voice Select
    if (elements.sideVoiceSelect) {
      elements.sideVoiceSelect.addEventListener('change', (e) => {
        switchVoice(e.target.value);
      });
    }

    // 3. Settings Modal Voice Select
    if (elements.selectVoice) {
      elements.selectVoice.addEventListener('change', (e) => {
        switchVoice(e.target.value);
      });
    }

    // 4. Settings Modal Sarvam Speaker Select
    if (elements.selectSarvamSpeaker) {
      elements.selectSarvamSpeaker.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val) {
          currentSarvamSpeaker = val;
          localStorage.setItem('preferred_sarvam_speaker', val);
          switchVoice(val);
        }
      });
    }

    // 5. Sarvam AI / GPT voice engine toggle
    if (elements.btnVoiceEngines) {
      elements.btnVoiceEngines.forEach((btn) => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const engine = btn.getAttribute('data-engine');
          if (engine === 'gpt' && isSarvamVoiceId(currentVoice)) {
            switchVoice(lastGptVoice || 'marin');
          } else if (engine === 'sarvam' && !isSarvamVoiceId(currentVoice)) {
            switchVoice(currentSarvamSpeaker && currentSarvamSpeaker !== 'kavitha' ? currentSarvamSpeaker : 'sarvam');
          }
        });
      });
    }

    // 6. Quick GPT Voice Chips
    if (elements.btnVoiceChips) {
      elements.btnVoiceChips.forEach((btn) => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const voiceId = btn.getAttribute('data-voice');
          if (voiceId) {
            switchVoice(voiceId);
          }
        });
      });
    }

    // 7. "More Voices" Dropdown Trigger
    if (elements.btnMoreVoices && elements.voiceDropdownMenu) {
      elements.btnMoreVoices.addEventListener('click', (e) => {
        e.stopPropagation();
        const isOpen = elements.voiceDropdownMenu.style.display === 'flex';
        elements.voiceDropdownMenu.style.display = isOpen ? 'none' : 'flex';
      });

      // Close dropdown when clicking outside
      document.addEventListener('click', (e) => {
        if (elements.voiceDropdownMenu && !e.target.closest('.voice-dropdown-wrapper')) {
          elements.voiceDropdownMenu.style.display = 'none';
        }
      });
    }

    // 8. Dropdown Menu Items
    if (elements.voiceMenuItems) {
      elements.voiceMenuItems.forEach((item) => {
        item.addEventListener('click', (e) => {
          e.stopPropagation();
          const voiceId = item.getAttribute('data-voice');
          if (voiceId) {
            switchVoice(voiceId);
            if (elements.voiceDropdownMenu) {
              elements.voiceDropdownMenu.style.display = 'none';
            }
          }
        });
      });
    }
  }

  // ── User Event Listeners ──────────────────────────────────────────────────
  function setupEventListeners() {
    // 1. Start / End Call Button
    elements.btnMasterCall.addEventListener('click', async () => {
      if (ws && ws.readyState === WebSocket.CONNECTING) return;
      if (isConnected) {
        disconnectSession();
      } else {
        await initAudioContext();
        connectWebSocket();
      }
    });

    // 2. Mute Mic Toggle
    elements.btnToggleMic.addEventListener('click', () => {
      isMuted = !isMuted;
      elements.btnToggleMic.classList.toggle('active', isMuted);
      elements.btnToggleMic.setAttribute('data-tooltip', isMuted ? 'Unmute Mic' : 'Mute Mic');
      if (isMuted) {
        elements.stateHelper.textContent = 'Microphone is muted';
      } else if (isConnected) {
        elements.stateHelper.textContent = 'Speak naturally in English or Tamil...';
      }
    });

    // 3. Interrupt AI Button
    elements.btnInterrupt.addEventListener('click', () => {
      stopAllPlayback();
      sendWsEvent({ type: 'response.cancel' });
    });

    // 4. Audio Mode Toggle (Browser Web Audio vs Host Machine sounddevice)
    elements.btnAudioMode.addEventListener('click', () => {
      audioMode = audioMode === 'browser' ? 'host' : 'browser';
      updateAudioModeUI();
      if (isConnected) {
        sendWsEvent({
          type: 'session.update_mode',
          audio_mode: audioMode,
        });
        if (audioMode === 'browser') {
          startMicrophone();
        } else {
          stopMicrophone();
        }
      }
    });

    // 5. Send Text Message (Hybrid Input)
    elements.btnSendText.addEventListener('click', sendTextMessage);
    elements.inputMessage.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        sendTextMessage();
      }
    });

    // 6. Quick Prompt Chips
    elements.quickChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        const text = chip.getAttribute('data-prompt') || chip.textContent.trim();
        elements.inputMessage.value = text;
        sendTextMessage();
      });
    });

    // 7. Conversation History Drawer
    elements.btnHistory.addEventListener('click', () => {
      elements.historyDrawer.classList.add('open');
      elements.drawerBackdrop.classList.add('open');
    });

    elements.btnCloseDrawer.addEventListener('click', closeHistoryDrawer);
    elements.drawerBackdrop.addEventListener('click', closeHistoryDrawer);

    elements.btnClearHistory.addEventListener('click', () => {
      conversationHistory = [];
      elements.drawerBody.innerHTML = '';
    });

    elements.btnCopyHistory.addEventListener('click', () => {
      const fullText = conversationHistory
        .map((m) => `[${m.time}] ${m.role.toUpperCase()}: ${m.text}`)
        .join('\n\n');
      navigator.clipboard.writeText(fullText);
      alert('Conversation history copied to clipboard!');
    });

    // 8. Settings Modal
    elements.btnSettings.addEventListener('click', openSettingsModal);
    elements.btnCloseSettings.addEventListener('click', closeSettingsModal);
    if (elements.btnCancelSettings) {
      elements.btnCancelSettings.addEventListener('click', closeSettingsModal);
    }
    if (elements.btnSyncMainSettings) {
      elements.btnSyncMainSettings.addEventListener('click', () => {
        loadMainServerSettings(true);
      });
    }
    elements.settingsBackdrop.addEventListener('click', (e) => {
      if (e.target === elements.settingsBackdrop) {
        closeSettingsModal();
      }
    });

    // 8b. 7-Layer Architecture Modal & Pipeline Strip
    if (elements.btnOpenArchModal) {
      elements.btnOpenArchModal.addEventListener('click', openArchModal);
    }
    if (elements.btnCloseArch) {
      elements.btnCloseArch.addEventListener('click', closeArchModal);
    }
    if (elements.btnCloseArchFooter) {
      elements.btnCloseArchFooter.addEventListener('click', closeArchModal);
    }
    if (elements.archBackdrop) {
      elements.archBackdrop.addEventListener('click', (e) => {
        if (e.target === elements.archBackdrop) {
          closeArchModal();
        }
      });
    }
    document.querySelectorAll('.arch-chip').forEach(chip => {
      chip.addEventListener('click', openArchModal);
      chip.style.cursor = 'pointer';
    });

    // Custom Reasoning Model input toggle
    if (elements.selectReasoningModel) {
      elements.selectReasoningModel.addEventListener('change', () => {
        const val = elements.selectReasoningModel.value;
        const isCustom = val === 'custom';
        if (elements.customReasoningModelGroup) {
          elements.customReasoningModelGroup.style.display = isCustom ? 'block' : 'none';
          if (isCustom && elements.customReasoningModelInput) {
            elements.customReasoningModelInput.focus();
          }
        }
        if (elements.badgeReasoningTier) {
          elements.badgeReasoningTier.textContent = isCustom ? 'CUSTOM REASONING' : (val === 'gpt-5.6-sol' ? 'GPT-5.6-SOL FLAGSHIP' : val.toUpperCase());
        }
        if (elements.summaryReasoningModel) {
          elements.summaryReasoningModel.textContent = isCustom ? (elements.customReasoningModelInput ? elements.customReasoningModelInput.value || 'custom' : 'custom') : val;
        }
      });
    }

    // Custom Model input toggle
    if (elements.selectModel) {
      elements.selectModel.addEventListener('change', () => {
        const isCustom = elements.selectModel.value === 'custom';
        if (elements.customModelGroup) {
          elements.customModelGroup.style.display = isCustom ? 'block' : 'none';
          if (isCustom && elements.customModelInput) {
            elements.customModelInput.focus();
          }
        }
        if (elements.summaryRealtimeModel) {
          elements.summaryRealtimeModel.textContent = isCustom ? (elements.customModelInput ? elements.customModelInput.value || 'custom' : 'custom') : elements.selectModel.value;
        }
      });
    }

    // Reasoning effort changes
    if (elements.selectReasoningEffort) {
      elements.selectReasoningEffort.addEventListener('change', () => {
        if (elements.badgeReasoningEffortVal) {
          elements.badgeReasoningEffortVal.textContent = elements.selectReasoningEffort.value.toUpperCase();
        }
      });
    }

    // Temperature slider live display
    if (elements.sliderTemperature) {
      elements.sliderTemperature.addEventListener('input', () => {
        const val = parseFloat(elements.sliderTemperature.value);
        const desc = val <= 0.3 ? 'Strict Factual' : val <= 0.65 ? 'High Accuracy' : 'Creative';
        if (elements.labelTemperatureVal) {
          elements.labelTemperatureVal.textContent = `${val.toFixed(2)} (${desc})`;
        }
      });
    }

    // Sarvam Pace slider live display
    if (elements.sliderSarvamPace) {
      elements.sliderSarvamPace.addEventListener('input', () => {
        const val = parseFloat(elements.sliderSarvamPace.value);
        const desc = val <= 0.8 ? 'Slow' : val <= 1.2 ? 'Natural Speed' : 'Fast';
        if (elements.labelSarvamPaceVal) {
          elements.labelSarvamPaceVal.textContent = `${val.toFixed(2)}x (${desc})`;
        }
      });
    }

    // In-Modal Live Audio Testing (Sarvam AI)
    const btnTest = elements.btnTestSarvam || elements.btnTestElevenLabs;
    if (btnTest) {
      btnTest.addEventListener('click', async () => {
        const inputTestText = elements.inputSarvamTestText || elements.inputElevenLabsTestText;
        const text = inputTestText ? inputTestText.value.trim() : '';
        const speaker = (elements.selectSarvamSpeaker && elements.selectSarvamSpeaker.value) ? elements.selectSarvamSpeaker.value : currentSarvamSpeaker;
        const model = (elements.selectSarvamModel && elements.selectSarvamModel.value) ? elements.selectSarvamModel.value : currentSarvamModel;
        const pace = elements.sliderSarvamPace ? parseFloat(elements.sliderSarvamPace.value) : currentSarvamPace;
        const language = elements.selectSarvamLanguage ? elements.selectSarvamLanguage.value : currentSarvamLanguage;

        const testStatus = elements.sarvamTestStatus || elements.elevenLabsTestStatus;
        if (testStatus) {
          testStatus.className = 'test-status-tag loading';
          testStatus.textContent = 'Synthesizing with Sarvam AI...';
        }
        btnTest.disabled = true;

        try {
          const res = await fetch('/api/sarvam/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: text || 'வணக்கம்! Mint X Solutions-ku welcome. Enna service venum nu sollunga?',
              speaker: speaker || 'kavitha',
              model: model || 'bulbul:v3',
              pace: pace || 1.0,
              language_code: language || 'ta-IN'
            })
          });
          const result = await res.json();
          if (!res.ok || result.status !== 'ok') {
            throw new Error(result.message || 'Synthesis failed');
          }

          if (testStatus) {
            testStatus.className = 'test-status-tag playing';
            testStatus.textContent = 'Playing 🔊';
          }

          const audio = new Audio('data:audio/wav;base64,' + result.audio_base64);
          audio.onended = () => {
            if (testStatus) {
              testStatus.className = 'test-status-tag';
              testStatus.textContent = 'Finished';
            }
          };
          await audio.play();
        } catch (err) {
          console.error('[Sarvam AI Test Audio Error]:', err);
          if (testStatus) {
            testStatus.className = 'test-status-tag error';
            testStatus.textContent = 'Error: ' + err.message;
          }
        } finally {
          btnTest.disabled = false;
        }
      });
    }

    // ── STT & VAD Slider Live Displays ──────────────────────────────────────
    // STT Model badge
    if (elements.selectSttModel) {
      elements.selectSttModel.addEventListener('change', () => {
        const val = elements.selectSttModel.value;
        const tierMap = {
          'saaras:v4': 'SARVAM SAARAS V4 REALTIME',
          'gpt-live-transcribe': 'OPENAI GPT LIVE TRANSCRIBE (HIGH MODEL)',
          'gpt-transcribe': 'GPT-TRANSCRIBE NEXT-GEN',
          'whisper-1': 'WHISPER-1 HIGH ACCURACY',
          'gpt-realtime-whisper': 'REALTIME WHISPER LOW LATENCY',
          'none': 'STT DISABLED'
        };
        if (elements.badgeSttTier) elements.badgeSttTier.textContent = tierMap[val] || val.toUpperCase();
        if (elements.summarySttModel) elements.summarySttModel.textContent = val;
        if (elements.chipSttName) elements.chipSttName.textContent = val === 'gpt-live-transcribe' ? 'GPT Live Transcribe' : val;
      });
    }

    // VAD Silence slider live display
    if (elements.sliderVadSilence) {
      elements.sliderVadSilence.addEventListener('input', () => {
        const val = parseInt(elements.sliderVadSilence.value);
        const desc = val <= 300 ? 'Snappy' : val <= 600 ? 'Conversational Balanced' : val <= 900 ? 'Patient' : 'Extended Thoughtful';
        if (elements.labelVadSilenceVal) elements.labelVadSilenceVal.textContent = `${val}ms (${desc})`;
      });
    }

    // VAD Threshold slider live display
    if (elements.sliderVadThreshold) {
      elements.sliderVadThreshold.addEventListener('input', () => {
        const val = parseFloat(elements.sliderVadThreshold.value);
        const desc = val <= 0.3 ? 'Very Sensitive' : val <= 0.55 ? 'Standard' : 'Noisy Environment';
        if (elements.labelVadThresholdVal) elements.labelVadThresholdVal.textContent = `${val.toFixed(2)} (${desc})`;
      });
    }

    // VAD Prefix slider live display
    if (elements.sliderVadPrefix) {
      elements.sliderVadPrefix.addEventListener('input', () => {
        const val = parseInt(elements.sliderVadPrefix.value);
        const desc = val <= 150 ? 'Snappy' : val <= 350 ? 'Crisp Syllables' : 'Extended Buffer';
        if (elements.labelVadPrefixVal) elements.labelVadPrefixVal.textContent = `${val}ms (${desc})`;
      });
    }

    // VAD Quick Preset Chips
    const vadPresetButtons = [elements.btnVadSnappy, elements.btnVadBalanced, elements.btnVadPatient];
    vadPresetButtons.forEach(btn => {
      if (!btn) return;
      btn.addEventListener('click', () => {
        vadPresetButtons.forEach(b => b?.classList.remove('active'));
        btn.classList.add('active');
        const silence = parseInt(btn.getAttribute('data-silence'));
        const threshold = parseFloat(btn.getAttribute('data-threshold'));
        if (elements.sliderVadSilence) {
          elements.sliderVadSilence.value = silence;
          elements.sliderVadSilence.dispatchEvent(new Event('input'));
        }
        if (elements.sliderVadThreshold) {
          elements.sliderVadThreshold.value = threshold;
          elements.sliderVadThreshold.dispatchEvent(new Event('input'));
        }
      });
    });

    // ── TTS Slider Live Displays ────────────────────────────────────────────
    // TTS Speed slider live display
    if (elements.sliderTtsSpeed) {
      elements.sliderTtsSpeed.addEventListener('input', () => {
        const val = parseFloat(elements.sliderTtsSpeed.value);
        const desc = val <= 0.85 ? 'Deliberate' : val <= 1.05 ? 'Natural Speed' : val <= 1.25 ? 'Brisk' : 'Fast';
        if (elements.labelTtsSpeedVal) elements.labelTtsSpeedVal.textContent = `${val.toFixed(2)}x (${desc})`;
      });
    }

    // TTS Volume slider live display
    if (elements.sliderTtsVolume) {
      elements.sliderTtsVolume.addEventListener('input', () => {
        const val = parseInt(elements.sliderTtsVolume.value);
        const desc = val <= 50 ? 'Soft Whisper' : val <= 110 ? 'Standard Output' : 'Boosted Loud';
        if (elements.labelTtsVolumeVal) elements.labelTtsVolumeVal.textContent = `${val}% (${desc})`;
        // Live volume adjustment
        if (speakerGainNode) speakerGainNode.gain.value = val / 100;
      });
    }

    // TTS Speed Quick Preset Chips
    const speedPresetButtons = [elements.btnSpeedSlow, elements.btnSpeedNormal, elements.btnSpeedFast];
    speedPresetButtons.forEach(btn => {
      if (!btn) return;
      btn.addEventListener('click', () => {
        speedPresetButtons.forEach(b => b?.classList.remove('active'));
        btn.classList.add('active');
        const speed = parseFloat(btn.getAttribute('data-speed'));
        if (elements.sliderTtsSpeed) {
          elements.sliderTtsSpeed.value = speed;
          elements.sliderTtsSpeed.dispatchEvent(new Event('input'));
        }
      });
    });

    // TTS Volume Quick Preset Chips
    const volPresetButtons = [elements.btnVolQuiet, elements.btnVolStandard, elements.btnVolBoost];
    volPresetButtons.forEach(btn => {
      if (!btn) return;
      btn.addEventListener('click', () => {
        volPresetButtons.forEach(b => b?.classList.remove('active'));
        btn.classList.add('active');
        const volume = parseInt(btn.getAttribute('data-volume'));
        if (elements.sliderTtsVolume) {
          elements.sliderTtsVolume.value = volume;
          elements.sliderTtsVolume.dispatchEvent(new Event('input'));
        }
      });
    });

    elements.btnSaveSettings.addEventListener('click', async () => {
      const prevModel = currentModel;

      // Voice: detect if selected value is Sarvam AI voice vs OpenAI native
      if (elements.selectVoice) {
        const selectedVoiceVal = elements.selectVoice.value;
        const sarvamVoices = ['sarvam', 'kavitha', 'kavya', 'priya', 'shreya', 'roopa', 'mani', 'gokul', 'vijay'];
        const isSarvamVoice = sarvamVoices.includes(selectedVoiceVal) || selectedVoiceVal === 'elevenlabs';
        if (isSarvamVoice) {
          currentVoice = (selectedVoiceVal === 'elevenlabs') ? 'sarvam' : selectedVoiceVal;
          if (selectedVoiceVal !== 'sarvam' && selectedVoiceVal !== 'elevenlabs') {
            currentSarvamSpeaker = selectedVoiceVal;
            localStorage.setItem('preferred_sarvam_speaker', selectedVoiceVal);
          }
        } else {
          currentVoice = selectedVoiceVal;
        }
        currentVoice = toLiveVoice(currentVoice);
      }

      const isCustomModel = elements.selectModel && elements.selectModel.value === 'custom';
      if (isCustomModel && elements.customModelInput && elements.customModelInput.value.trim()) {
        currentModel = elements.customModelInput.value.trim();
      } else if (elements.selectModel) {
        currentModel = elements.selectModel.value;
      }

      // 🧠 Reasoning & Requirements Model (gpt-5.6-sol)
      const isCustomReasoning = elements.selectReasoningModel && elements.selectReasoningModel.value === 'custom';
      if (isCustomReasoning && elements.customReasoningModelInput && elements.customReasoningModelInput.value.trim()) {
        currentReasoningModel = elements.customReasoningModelInput.value.trim();
      } else if (elements.selectReasoningModel) {
        currentReasoningModel = elements.selectReasoningModel.value;
      }

      if (elements.selectReasoningEffort) {
        currentReasoningEffort = elements.selectReasoningEffort.value;
      }
      if (elements.sliderTemperature) {
        currentTemperature = parseFloat(elements.sliderTemperature.value);
      }

      // Sarvam AI Model & Voice settings
      if (elements.selectSarvamModel) {
        currentSarvamModel = elements.selectSarvamModel.value;
      }
      if (elements.selectSarvamSpeaker) {
        currentSarvamSpeaker = elements.selectSarvamSpeaker.value;
      }
      if (elements.selectSarvamLanguage) {
        currentSarvamLanguage = elements.selectSarvamLanguage.value;
      }
      if (elements.sliderSarvamPace) {
        currentSarvamPace = parseFloat(elements.sliderSarvamPace.value);
      }

      audioMode = elements.selectAudioMode ? elements.selectAudioMode.value : audioMode;

      // STT & VAD Controls
      if (elements.selectSttModel) currentSttModel = elements.selectSttModel.value;
      if (elements.sliderVadSilence) currentVadSilenceMs = parseInt(elements.sliderVadSilence.value);
      if (elements.sliderVadThreshold) currentVadThreshold = parseFloat(elements.sliderVadThreshold.value);
      if (elements.sliderVadPrefix) currentVadPrefixMs = parseInt(elements.sliderVadPrefix.value);
      if (elements.chkMicEcho) currentMicEcho = elements.chkMicEcho.checked;
      if (elements.chkMicNoise) currentMicNoise = elements.chkMicNoise.checked;
      if (elements.chkMicGain) currentMicGain = elements.chkMicGain.checked;
      if (elements.chkBargeIn) currentBargeIn = elements.chkBargeIn.checked;

      // TTS Playback Controls
      if (elements.sliderTtsSpeed) currentTtsSpeed = parseFloat(elements.sliderTtsSpeed.value);
      if (elements.sliderTtsVolume) {
        currentTtsVolume = parseInt(elements.sliderTtsVolume.value);
        if (speakerGainNode) speakerGainNode.gain.value = currentTtsVolume / 100;
      }
      if (elements.selectTtsModality) currentTtsModality = elements.selectTtsModality.value;
      if (elements.selectTtsDelivery) currentTtsDelivery = elements.selectTtsDelivery.value;

      // Save to localStorage
      localStorage.setItem('preferred_voice', currentVoice);
      localStorage.setItem('preferred_model', currentModel);
      localStorage.setItem('preferred_reasoning_model', currentReasoningModel);
      localStorage.setItem('preferred_reasoning_effort', currentReasoningEffort);
      localStorage.setItem('preferred_temperature', currentTemperature);
      localStorage.setItem('preferred_sarvam_model', currentSarvamModel);
      localStorage.setItem('preferred_sarvam_speaker', currentSarvamSpeaker);
      localStorage.setItem('preferred_sarvam_language', currentSarvamLanguage);
      localStorage.setItem('preferred_sarvam_pace', currentSarvamPace);
      localStorage.setItem('preferred_stt_model', currentSttModel);
      localStorage.setItem('preferred_vad_silence_ms', currentVadSilenceMs);
      localStorage.setItem('preferred_vad_threshold', currentVadThreshold);
      localStorage.setItem('preferred_vad_prefix_ms', currentVadPrefixMs);
      localStorage.setItem('preferred_mic_echo', currentMicEcho);
      localStorage.setItem('preferred_mic_noise', currentMicNoise);
      localStorage.setItem('preferred_mic_gain', currentMicGain);
      localStorage.setItem('preferred_barge_in', currentBargeIn);
      localStorage.setItem('preferred_tts_speed', currentTtsSpeed);
      localStorage.setItem('preferred_tts_volume', currentTtsVolume);
      localStorage.setItem('preferred_tts_modality', currentTtsModality);
      localStorage.setItem('preferred_tts_delivery', currentTtsDelivery);
      localStorage.setItem('preferred_embedding_model', currentEmbeddingModel);
      localStorage.setItem('preferred_vector_db', currentVectorDb);

      // Sync Voice across all UI components (Nav select, side select, chips, labels)
      syncVoiceUI(currentVoice);
      updateArchStrip();

      if (elements.badgeReasoningVal) elements.badgeReasoningVal.textContent = `🧠 ${currentReasoningModel}`;
      if (elements.modelBadge) elements.modelBadge.textContent = currentModel;
      updateAudioModeUI();

      // Persist model settings to server (.env and in-memory runtime)
      try {
        fetch('/api/models/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            reasoning_model: currentReasoningModel,
            realtime_model: currentModel,
            transcription_model: currentSttModel,
            stt_model: currentSttModel,
            embedding_model: currentEmbeddingModel,
            vector_db: currentVectorDb,
            voice: currentVoice,
            tts_provider: 'sarvam',
            sarvam_speaker: currentSarvamSpeaker,
            sarvam_model: currentSarvamModel,
            sarvam_pace: currentSarvamPace,
            sarvam_language_code: currentSarvamLanguage,
          })
        }).catch(e => console.warn('[Settings] Error saving to .env:', e));
      } catch (e) {
        console.warn('[Settings] Error triggering save to .env:', e);
      }

      const newPrompt = elements.customInstructions ? elements.customInstructions.value.trim() : '';

      // If preset select changed, activate that preset
      const selectedPresetId = elements.selectInstructionPreset ? elements.selectInstructionPreset.value : null;
      if (selectedPresetId && selectedPresetId !== activeInstructionId) {
        await activateInstruction(selectedPresetId, false);
      } else if (newPrompt && (!activeInstruction || newPrompt !== activeInstruction.instructions)) {
        // User directly modified the prompt in settings textarea
        await updateActiveInstructionPrompt(newPrompt);
      }

      // If connected and model changed, reconnect websocket so OpenAI connects with new model
      if (isConnected && prevModel !== currentModel) {
        showToast(`🔄 Switching AI model to ${currentModel}... Reconnecting session...`);
        disconnectSession();
        setTimeout(() => {
          connectWebSocket();
        }, 300);
      } else if (isConnected) {
        sendWsEvent({
          type: 'session.update',
          voice: currentVoice,
          model: currentModel,
          audio_mode: audioMode,
          reasoning_effort: currentReasoningEffort,
          temperature: currentTemperature,
          stt_model: currentSttModel,
          vad_threshold: currentVadThreshold,
          vad_silence_ms: currentVadSilenceMs,
          vad_prefix_ms: currentVadPrefixMs,
          modalities: isSarvamVoiceId(currentVoice) ? ['text'] : ['audio'],
          tts_provider: isSarvamVoiceId(currentVoice) ? 'sarvam' : 'openai',
          sarvam_speaker: currentSarvamSpeaker,
          sarvam_model: currentSarvamModel,
          sarvam_pace: currentSarvamPace,
          sarvam_language_code: currentSarvamLanguage,
          instructions: newPrompt || (activeInstruction ? activeInstruction.instructions : ''),
        });
      }

      // Get voice display name for toast
      const voiceDisplayName = (VOICE_INFO[currentVoice] || { name: currentVoice }).name;
      showToast(`✅ Applied: 🧠 ${currentReasoningModel} • 🎙️ ${currentModel} • 🗣️ ${currentSttModel} • 🔊 ${voiceDisplayName}`);
      closeSettingsModal();
    });

    // 9. Click on Core Orb to Toggle Session
    elements.orbWrapper.addEventListener('click', () => {
      elements.btnMasterCall.click();
    });

    // 10. Escape Key Closes Any Open Modal / Drawer
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeSettingsModal();
        closeArchModal();
        closeInstructionsModal();
        closeHistoryDrawer();
        closeRecModal();
      }
    });

    // 11. OpenAI API Key & Quota Management Controls
    const btnSaveKey = document.getElementById('btn-save-openai-key');
    if (btnSaveKey) {
      btnSaveKey.addEventListener('click', saveOpenAiKey);
    }
    const inputKey = document.getElementById('input-openai-api-key');
    if (inputKey) {
      inputKey.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          saveOpenAiKey();
        }
      });
    }
    const btnBannerUpdate = document.getElementById('btn-banner-update-key');
    if (btnBannerUpdate) {
      btnBannerUpdate.addEventListener('click', () => {
        openSettingsModal();
        setTimeout(() => {
          const kInput = document.getElementById('input-openai-api-key');
          if (kInput) kInput.focus();
        }, 300);
      });
    }
  }

  async function fetchOpenAiKeyStatus() {
    try {
      const res = await fetch('/api/openai/status');
      const data = await res.json();
      const badge = document.getElementById('openai-key-status-badge');
      const text = document.getElementById('openai-status-text');
      const banner = document.getElementById('quota-alert-banner');

      if (data.status === 'quota_exhausted') {
        if (badge) {
          badge.textContent = '0 CREDITS (EXHAUSTED)';
          badge.style.background = 'rgba(239, 68, 68, 0.25)';
          badge.style.color = '#f87171';
          badge.style.borderColor = 'rgba(239, 68, 68, 0.5)';
        }
        if (text) {
          text.innerHTML = `<span style="color: #f87171; font-weight: 600;">⚠️ Account credit balance exhausted:</span> ${data.message || 'Please recharge credits or enter a new API key.'}`;
        }
        if (banner) banner.style.display = 'flex';
      } else if (data.status === 'ok') {
        if (badge) {
          badge.textContent = 'ACTIVE (CREDITS OK ✅)';
          badge.style.background = 'rgba(16, 185, 129, 0.2)';
          badge.style.color = '#34d399';
          badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        }
        if (text) {
          text.innerHTML = `<span style="color: #34d399;">Key Configured:</span> <code>${data.masked_key}</code> — Credits available.`;
        }
        if (banner) banner.style.display = 'none';
      } else if (data.status === 'missing') {
        if (badge) {
          badge.textContent = 'MISSING KEY';
          badge.style.background = 'rgba(239, 68, 68, 0.25)';
          badge.style.color = '#f87171';
        }
        if (text) {
          text.innerHTML = `<span style="color: #f87171;">No OPENAI_API_KEY set.</span> Enter your key below.`;
        }
        if (banner) banner.style.display = 'flex';
      }
    } catch (e) {
      console.warn('Could not fetch OpenAI status:', e);
    }
  }

  async function saveOpenAiKey() {
    const input = document.getElementById('input-openai-api-key');
    const btn = document.getElementById('btn-save-openai-key');
    const key = (input ? input.value : '').trim();

    if (!key) {
      showToast('⚠️ Please paste a valid OpenAI API key starting with sk-');
      return;
    }

    if (btn) {
      btn.disabled = true;
      btn.textContent = 'Validating...';
    }

    try {
      const res = await fetch('/api/openai/key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key })
      });
      const data = await res.json();

      if (res.ok && data.status === 'ok') {
        showToast('✅ OpenAI API Key validated and saved successfully!');
        if (input) input.value = '';
        await fetchOpenAiKeyStatus();
      } else {
        showToast(`❌ ${data.message || 'Key validation failed'}`);
        await fetchOpenAiKeyStatus();
      }
    } catch (err) {
      showToast(`❌ Network error while saving key: ${err.message}`);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = 'Save & Test';
      }
    }
  }

  function updateAudioModeUI() {
    const isBrowser = audioMode === 'browser';
    elements.btnAudioMode.setAttribute(
      'data-tooltip',
      isBrowser ? 'Audio: Browser Mic & Speakers' : 'Audio: Host Machine (sounddevice)'
    );
    if (elements.selectAudioMode) {
      elements.selectAudioMode.value = audioMode;
    }
  }

  function closeHistoryDrawer() {
    elements.historyDrawer.classList.remove('open');
    elements.drawerBackdrop.classList.remove('open');
  }

  function updateArchStrip() {
    if (elements.chipSttName) {
      elements.chipSttName.textContent = 'GPT-Live (built-in)';
    }
    if (elements.chipLlmName) {
      elements.chipLlmName.textContent = currentModel;
    }
    if (elements.chipTtsName) {
      elements.chipTtsName.textContent = `GPT-Live voice (${currentVoice})`;
    }
    if (elements.chipVectorName) {
      elements.chipVectorName.textContent = 'Qdrant (mintx)';
    }
  }

  async function openArchModal() {
    if (!elements.archBackdrop || !elements.archModal) return;

    elements.archBackdrop.style.display = 'flex';
    requestAnimationFrame(() => {
      elements.archBackdrop.classList.add('open');
      elements.archModal.classList.add('open');
    });

    if (elements.archModalGrid) {
      elements.archModalGrid.innerHTML = `
        <div style="grid-column: 1 / -1; text-align: center; padding: 2.5rem; color: #94a3b8;">
          <div class="spinner" style="margin: 0 auto 1rem; border-color: rgba(99, 102, 241, 0.3); border-top-color: #6366f1;"></div>
          Fetching live diagnostics for 7-Layer Architecture Stack...
        </div>
      `;
    }

    try {
      const res = await fetch('/api/architecture');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderArchModalGrid(data);
    } catch (e) {
      console.error('[Arch Modal Error]', e);
      if (elements.archModalGrid) {
        elements.archModalGrid.innerHTML = `
          <div style="grid-column: 1 / -1; color: #f87171; text-align: center; padding: 1.5rem;">
            Failed to load architecture diagnostics: ${e.message}
          </div>
        `;
      }
    }
  }

  function closeArchModal() {
    if (!elements.archBackdrop || !elements.archModal) return;
    elements.archBackdrop.classList.remove('open');
    elements.archModal.classList.remove('open');
    setTimeout(() => {
      elements.archBackdrop.style.display = 'none';
    }, 250);
  }

  function renderArchModalGrid(data) {
    if (!elements.archModalGrid) return;
    const layers = [
      {
        num: 1,
        icon: '🎤',
        layer: 'STT',
        name: 'OpenAI GPT Live Transcribe',
        badge: 'REALTIME STREAMING',
        badgeClass: 'badge-tag-indigo',
        purpose: 'Real-time customer speech → text',
        specs: [
          { label: 'Active Model', val: data.stt?.active_model || 'gpt-live-transcribe' },
          { label: 'VAD Prefix', val: `${currentVadPrefixMs}ms` },
          { label: 'VAD Silence', val: `${currentVadSilenceMs}ms` },
          { label: 'Sample Rate', val: '24,000 Hz' }
        ]
      },
      {
        num: 2,
        icon: '🧠',
        layer: 'LLM',
        name: 'OpenAI Latest Realtime / Reasoning',
        badge: 'REASONING & DECISION',
        badgeClass: 'badge-tag-purple',
        purpose: 'Conversation + RAG + decision making',
        specs: [
          { label: 'Realtime Model', val: data.llm?.realtime_model || 'gpt-realtime-2.1' },
          { label: 'Reasoning Model', val: data.llm?.reasoning_model || 'gpt-5.6-sol' },
          { label: 'Reasoning Effort', val: currentReasoningEffort.toUpperCase() },
          { label: 'Temperature', val: currentTemperature.toFixed(2) }
        ]
      },
      {
        num: 3,
        icon: '🔎',
        layer: 'Embeddings',
        name: 'text-embedding-3-large',
        badge: '3072 DIMENSIONS',
        badgeClass: 'badge-tag-cyan',
        purpose: 'RAG semantic search & knowledge indexing',
        specs: [
          { label: 'Model', val: data.embeddings?.model || 'text-embedding-3-large' },
          { label: 'Dimensions', val: `${data.embeddings?.dimensions || 3072} dims` },
          { label: 'Distance Metric', val: data.embeddings?.metric || 'Cosine' },
          { label: 'Cache Status', val: data.embeddings?.cache_loaded ? 'Pre-indexed' : 'Live' }
        ]
      },
      {
        num: 4,
        icon: '📚',
        layer: 'Vector DB',
        name: 'Qdrant Vector Database',
        badge: data.vector_db?.status === 'online' ? 'HNSW INDEX ONLINE' : 'STANDBY',
        badgeClass: 'badge-tag-green',
        purpose: 'MintX knowledge retrieval',
        specs: [
          { label: 'Database', val: 'Qdrant' },
          { label: 'Collection', val: data.vector_db?.collection || 'mintx_knowledge' },
          { label: 'Indexed Vectors', val: `${data.vector_db?.indexed_vectors || 12} items` },
          { label: 'Storage Mode', val: data.vector_db?.storage_path ? 'Persistent Local' : 'In-Memory' }
        ]
      },
      {
        num: 5,
        icon: '🔊',
        layer: 'TTS',
        name: 'Eleven v4 Turbo',
        badge: '~100MS HIGH QUALITY',
        badgeClass: 'badge-tag-yellow',
        purpose: 'Real-time, high-quality voice synthesis',
        specs: [
          { label: 'Engine', val: 'ElevenLabs v4 Turbo' },
          { label: 'Model ID', val: data.tts?.active_model || 'eleven_v4_turbo' },
          { label: 'Voice ID', val: data.tts?.voice_id || currentElevenLabsVoiceId },
          { label: 'Target Latency', val: '~100 ms' }
        ]
      },
      {
        num: 6,
        icon: '💾',
        layer: 'Database',
        name: 'MongoDB',
        badge: data.database?.status === 'connected' ? 'CONNECTED' : 'STANDBY',
        badgeClass: 'badge-tag-green',
        purpose: 'Customer/lead details storage',
        specs: [
          { label: 'Database Name', val: data.database?.database || 'ai_phone_assistant' },
          { label: 'Leads Collection', val: data.database?.leads_collection || 'leads' },
          { label: 'Driver', val: 'Motor / PyMongo' },
          { label: 'Sync State', val: 'Live customer collection' }
        ]
      },
      {
        num: 7,
        icon: '⚡',
        layer: 'Backend',
        name: 'FastAPI + WebSocket',
        badge: 'FULL-DUPLEX 24KHZ',
        badgeClass: 'badge-tag-purple',
        purpose: 'Real-time call pipeline & audio routing',
        specs: [
          { label: 'Framework', val: 'FastAPI + Uvicorn ASGI' },
          { label: 'Protocol', val: 'WebSocket /ws/realtime' },
          { label: 'Audio Stream', val: 'PCM 16-bit 24kHz Mono' },
          { label: 'Pipeline Type', val: 'Realtime Streaming WS' }
        ]
      }
    ];

    elements.archModalGrid.innerHTML = layers.map(l => `
      <div class="arch-card">
        <div class="arch-card-header">
          <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.4rem;">${l.icon}</span>
            <div>
              <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Layer ${l.num}: ${l.layer}</div>
              <h4 style="font-size: 0.95rem; font-weight: 700; color: #fff; margin: 0;">${l.name}</h4>
            </div>
          </div>
          <span class="badge-tag ${l.badgeClass}">${l.badge}</span>
        </div>
        <div class="arch-card-purpose">${l.purpose}</div>
        <div class="arch-specs-list">
          ${l.specs.map(s => `
            <div class="arch-spec-item">
              <span class="arch-spec-label">${s.label}</span>
              <span class="arch-spec-val">${s.val}</span>
            </div>
          `).join('')}
        </div>
      </div>
    `).join('');
  }

  function updateElevenLabsTierBadge(modelId) {
    if (!elements.badgeElevenTier) return;
    const tierMap = {
      eleven_v4_turbo: 'ELEVEN V4 TURBO (~100MS HIGH MODEL)',
      eleven_turbo_v2_5: 'REAL-TIME FAST ACCURACY',
      eleven_v3: 'STUDIO ACCURACY',
      eleven_multilingual_v2: 'MULTILINGUAL ACCURACY',
      eleven_flash_v2_5: 'FLASH SPEED',
      eleven_turbo_v2: 'CONVERSATIONAL',
      eleven_multilingual_v1: 'LEGACY MODEL',
      custom: 'CUSTOM MODEL'
    };
    elements.badgeElevenTier.textContent = tierMap[modelId] || 'CUSTOM MODEL';
  }

  function openSettingsModal() {
    fetchOpenAiKeyStatus();
    // Restore settings voice select
    if (elements.selectVoice) {
      elements.selectVoice.value = currentVoice;
    }

    // Populate Reasoning & Customer Requirements Model
    const standardReasoningModels = ['gpt-5.6-sol', 'gpt-5.6', 'o3-mini', 'gpt-4o', 'gpt-4o-mini'];
    if (elements.selectReasoningModel) {
      if (standardReasoningModels.includes(currentReasoningModel)) {
        elements.selectReasoningModel.value = currentReasoningModel;
        if (elements.customReasoningModelGroup) elements.customReasoningModelGroup.style.display = 'none';
      } else {
        elements.selectReasoningModel.value = 'custom';
        if (elements.customReasoningModelGroup) elements.customReasoningModelGroup.style.display = 'block';
        if (elements.customReasoningModelInput) elements.customReasoningModelInput.value = currentReasoningModel;
      }
      if (elements.badgeReasoningTier) {
        elements.badgeReasoningTier.textContent = currentReasoningModel === 'gpt-5.6-sol' ? 'GPT-5.6-SOL FLAGSHIP' : currentReasoningModel.toUpperCase();
      }
    }

    // Update Multi-Model Pipeline Architecture Overview Cards
    if (elements.summaryReasoningModel) elements.summaryReasoningModel.textContent = currentReasoningModel;
    if (elements.summaryRealtimeModel) elements.summaryRealtimeModel.textContent = currentModel;
    if (elements.summarySttModel) elements.summarySttModel.textContent = 'GPT-Live (built-in)';
    if (elements.summaryTtsEngine) elements.summaryTtsEngine.textContent = `GPT-Live voice (${currentVoice})`;

    const standardModels = [LIVE_MODEL];
    if (elements.selectModel) {
      if (standardModels.includes(currentModel)) {
        elements.selectModel.value = currentModel;
        if (elements.customModelGroup) elements.customModelGroup.style.display = 'none';
      } else {
        elements.selectModel.value = 'custom';
        if (elements.customModelGroup) elements.customModelGroup.style.display = 'block';
        if (elements.customModelInput) elements.customModelInput.value = currentModel;
      }
    }

    if (elements.selectReasoningEffort) {
      elements.selectReasoningEffort.value = currentReasoningEffort;
      if (elements.badgeReasoningEffortVal) elements.badgeReasoningEffortVal.textContent = currentReasoningEffort.toUpperCase();
    }

    if (elements.sliderTemperature) {
      elements.sliderTemperature.value = currentTemperature;
      if (elements.labelTemperatureVal) {
        const desc = currentTemperature <= 0.3 ? 'Strict Factual' : currentTemperature <= 0.65 ? 'High Accuracy' : 'Creative';
        elements.labelTemperatureVal.textContent = `${currentTemperature.toFixed(2)} (${desc})`;
      }
    }

    if (elements.selectAudioMode) {
      elements.selectAudioMode.value = audioMode;
    }

    // Populate Sarvam AI model, speaker, language, and pace
    if (elements.selectSarvamModel) {
      elements.selectSarvamModel.value = currentSarvamModel;
    }
    if (elements.selectSarvamSpeaker) {
      elements.selectSarvamSpeaker.value = currentSarvamSpeaker;
    }
    if (elements.selectSarvamLanguage) {
      elements.selectSarvamLanguage.value = currentSarvamLanguage;
    }
    if (elements.sliderSarvamPace) {
      elements.sliderSarvamPace.value = currentSarvamPace;
      if (elements.labelSarvamPaceVal) {
        const desc = currentSarvamPace <= 0.8 ? 'Slow' : currentSarvamPace <= 1.2 ? 'Natural Speed' : 'Fast';
        elements.labelSarvamPaceVal.textContent = `${currentSarvamPace.toFixed(2)}x (${desc})`;
      }
    }

    // ── Populate STT & VAD Controls ──
    if (elements.selectSttModel) {
      elements.selectSttModel.value = currentSttModel;
      const sttTierMap = {
        'saaras:v4': 'SARVAM SAARAS V4 REALTIME',
        'gpt-live-transcribe': 'OPENAI GPT LIVE TRANSCRIBE (HIGH MODEL)',
        'gpt-transcribe': 'GPT-TRANSCRIBE NEXT-GEN',
        'whisper-1': 'WHISPER-1 HIGH ACCURACY',
        'gpt-realtime-whisper': 'REALTIME WHISPER LOW LATENCY',
        'none': 'STT DISABLED'
      };
      if (elements.badgeSttTier) elements.badgeSttTier.textContent = sttTierMap[currentSttModel] || currentSttModel.toUpperCase();
    }

    // Fetch live architecture stats for Vector DB & Mongo in Settings Section 5
    fetch('/api/architecture').then(r => r.json()).then(arch => {
      if (arch && arch.vector_db) {
        if (elements.badgeQdrantStatus) {
          elements.badgeQdrantStatus.textContent = (arch.vector_db.status || 'ONLINE').toUpperCase();
        }
        if (elements.qdrantInfoText) {
          elements.qdrantInfoText.textContent = `Collection: ${arch.vector_db.collection || 'mintx_knowledge'} (${arch.vector_db.indexed_vectors || 12} items)`;
        }
      }
      if (arch && arch.database && elements.badgeMongoStatus) {
        elements.badgeMongoStatus.textContent = (arch.database.status || 'CONNECTED').toUpperCase();
      }
    }).catch(e => console.warn('[Settings] Error fetching arch diagnostics:', e));
    if (elements.sliderVadSilence) {
      elements.sliderVadSilence.value = currentVadSilenceMs;
      const desc = currentVadSilenceMs <= 300 ? 'Snappy' : currentVadSilenceMs <= 600 ? 'Conversational Balanced' : currentVadSilenceMs <= 900 ? 'Patient' : 'Extended Thoughtful';
      if (elements.labelVadSilenceVal) elements.labelVadSilenceVal.textContent = `${currentVadSilenceMs}ms (${desc})`;
    }
    if (elements.sliderVadThreshold) {
      elements.sliderVadThreshold.value = currentVadThreshold;
      const desc = currentVadThreshold <= 0.3 ? 'Very Sensitive' : currentVadThreshold <= 0.55 ? 'Standard' : 'Noisy Environment';
      if (elements.labelVadThresholdVal) elements.labelVadThresholdVal.textContent = `${currentVadThreshold.toFixed(2)} (${desc})`;
    }
    if (elements.sliderVadPrefix) {
      elements.sliderVadPrefix.value = currentVadPrefixMs;
      const desc = currentVadPrefixMs <= 150 ? 'Snappy' : currentVadPrefixMs <= 350 ? 'Crisp Syllables' : 'Extended Buffer';
      if (elements.labelVadPrefixVal) elements.labelVadPrefixVal.textContent = `${currentVadPrefixMs}ms (${desc})`;
    }
    if (elements.chkMicEcho) elements.chkMicEcho.checked = currentMicEcho;
    if (elements.chkBargeIn) elements.chkBargeIn.checked = currentBargeIn;
    if (elements.chkMicNoise) elements.chkMicNoise.checked = currentMicNoise;
    if (elements.chkMicGain) elements.chkMicGain.checked = currentMicGain;

    // ── Populate TTS Controls ──
    if (elements.sliderTtsSpeed) {
      elements.sliderTtsSpeed.value = currentTtsSpeed;
      const desc = currentTtsSpeed <= 0.85 ? 'Deliberate' : currentTtsSpeed <= 1.05 ? 'Natural Speed' : currentTtsSpeed <= 1.25 ? 'Brisk' : 'Fast';
      if (elements.labelTtsSpeedVal) elements.labelTtsSpeedVal.textContent = `${currentTtsSpeed.toFixed(2)}x (${desc})`;
    }
    if (elements.sliderTtsVolume) {
      elements.sliderTtsVolume.value = currentTtsVolume;
      const desc = currentTtsVolume <= 50 ? 'Soft Whisper' : currentTtsVolume <= 110 ? 'Standard Output' : 'Boosted Loud';
      if (elements.labelTtsVolumeVal) elements.labelTtsVolumeVal.textContent = `${currentTtsVolume}% (${desc})`;
    }
    if (elements.selectTtsModality) elements.selectTtsModality.value = currentTtsModality;
    if (elements.selectTtsDelivery) elements.selectTtsDelivery.value = currentTtsDelivery;

    // Instruction Presets & Textarea
    renderSettingsPresets();
    if (elements.selectInstructionPreset) {
      elements.selectInstructionPreset.value = activeInstructionId;
    }
    if (elements.customInstructions) {
      elements.customInstructions.value = activeInstruction ? activeInstruction.instructions : '';
    }

    // Show backdrop & modal
    elements.settingsBackdrop.style.display = 'flex';
    requestAnimationFrame(() => {
      elements.settingsBackdrop.classList.add('open');
      elements.settingsModal.classList.add('open');
    });
  }

  function closeSettingsModal() {
    elements.settingsBackdrop.classList.remove('open');
    elements.settingsModal.classList.remove('open');
    setTimeout(() => {
      elements.settingsBackdrop.style.display = 'none';
    }, 250);
  }

  function sendTextMessage() {
    const text = elements.inputMessage.value.trim();
    if (!text) return;
    elements.inputMessage.value = '';

    if (!isConnected) {
      pendingTextMessage = text;
      elements.btnMasterCall.click();
      return;
    }

    transmitUserText(text);
  }

  function transmitUserText(text) {
    if (!text || !text.trim()) return;

    addMessageToHistory('user', text);
    elements.liveSpeakerTag.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
      You (Text)
    `;
    elements.liveTranscriptText.textContent = `"${text}"`;

    // Engage echo shield so room noise does not create a secondary VAD response
    isAiSpeaking = true;

    sendWsEvent({
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: text }],
      },
    });

    sendWsEvent({
      type: 'response.create',
    });

    setAppState('thinking');
  }

  // ==========================================================================
  // Conversation Audio Recording & Text File Formatter System
  // ==========================================================================

  /**
   * Encodes recorded Float32Array PCM chunks into standard 16-bit PCM WAV Blob.
   * Compatible with all players (Windows Media Player, VLC, QuickTime, Android, iOS).
   */
  function encodeWAV(chunks, sampleRate) {
    let totalSamples = 0;
    for (let i = 0; i < chunks.length; i++) {
      totalSamples += chunks[i].length;
    }

    // Merge chunks
    const merged = new Float32Array(totalSamples);
    let offset = 0;
    for (let i = 0; i < chunks.length; i++) {
      merged.set(chunks[i], offset);
      offset += chunks[i].length;
    }

    // Convert to 16-bit signed PCM
    const pcm16 = new Int16Array(totalSamples);
    for (let i = 0; i < totalSamples; i++) {
      let s = Math.max(-1, Math.min(1, merged[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);
    const dataByteLen = pcm16.byteLength;

    // Standard 44-byte WAV header
    const headerBuffer = new ArrayBuffer(44);
    const view = new DataView(headerBuffer);

    function writeStr(view, offset, str) {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    }

    writeStr(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataByteLen, true);
    writeStr(view, 8, 'WAVE');
    writeStr(view, 12, 'fmt ');
    view.setUint32(16, 16, true);            // Subchunk1Size for PCM
    view.setUint16(20, 1, true);             // AudioFormat 1 = PCM
    view.setUint16(22, numChannels, true);   // 1 Channel (Mono)
    view.setUint32(24, sampleRate, true);    // Sample Rate
    view.setUint32(28, byteRate, true);      // Byte Rate
    view.setUint16(32, blockAlign, true);    // Block Align
    view.setUint16(34, bitsPerSample, true); // Bits per Sample
    writeStr(view, 36, 'data');
    view.setUint32(40, dataByteLen, true);   // Subchunk2Size (data size)

    return new Blob([headerBuffer, pcm16.buffer], { type: 'audio/wav' });
  }

  /**
   * Generates a structured, human-readable conversation transcript text file (.txt)
   */
  function formatTranscriptText(sessionId, durationStr, history) {
    const now = new Date();
    const dateStr = now.toISOString().replace('T', ' ').slice(0, 19);
    const personaName = activeInstruction
      ? `${activeInstruction.name} [${activeInstruction.category || 'Custom'}]`
      : 'Mintx Solutions (Sales)';
    const voiceName = VOICE_INFO[currentVoice]
      ? `${VOICE_INFO[currentVoice].name} (${VOICE_INFO[currentVoice].tone})`
      : currentVoice;

    let turnsCaller = 0;
    let turnsAssistant = 0;
    const dialogueLines = [];

    if (!history || history.length === 0) {
      dialogueLines.push('  (No spoken dialogue captured in this session)');
    } else {
      history.forEach((item, idx) => {
        const isUser = item.role === 'user';
        if (isUser) turnsCaller++;
        else turnsAssistant++;
        const speaker = isUser ? 'CALLER / YOU' : 'MINTX AI ASSISTANT';
        const timeBadge = item.time ? `[${item.time}]` : `[Turn ${idx + 1}]`;
        dialogueLines.push(`${timeBadge} ${speaker}:`);
        dialogueLines.push(`"${item.text.trim()}"`);
        dialogueLines.push('');
      });
    }

    const totalTurns = turnsCaller + turnsAssistant;
    const banner = '='.repeat(80);
    const subBanner = '-'.repeat(80);

    return [
      banner,
      'MINTX SOLUTIONS AI — CONVERSATION TRANSCRIPT & CALL LOG',
      banner,
      `Session ID         : ${sessionId}`,
      `Date & Time        : ${dateStr}`,
      `Call Duration      : ${durationStr}`,
      `Audio File Ref     : call_audio_${sessionId}.wav`,
      `AI Voice           : ${voiceName}`,
      `Realtime Model     : ${currentModel}`,
      `Active Persona     : ${personaName}`,
      `Total Dialogue     : ${totalTurns} turns (${turnsCaller} Caller, ${turnsAssistant} Assistant)`,
      banner,
      '',
      subBanner,
      'CONVERSATION DIALOGUE',
      subBanner,
      '',
      ...dialogueLines,
      banner,
      'END OF TRANSCRIPT • MINTX SOLUTIONS REALTIME VOICE STUDIO',
      banner,
    ].join('\r\n');
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 2000);
  }

  function downloadTextFile(text, filename) {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    downloadBlob(blob, filename);
  }

  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
    const secs = (seconds % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
  }

  function startCallRecording() {
    recordedPcmChunks = [];
    isActivelyRecording = true;
    recordingStartTime = Date.now();

    const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
    lastRecordedSessionId = `call_${timestamp}`;

    if (elements.recStatusPill) elements.recStatusPill.style.display = 'inline-flex';
    if (elements.recPillTimer) elements.recPillTimer.textContent = '00:00';
    if (elements.btnToggleRecord) elements.btnToggleRecord.classList.add('rec-active');

    if (recordingTimerInterval) clearInterval(recordingTimerInterval);
    recordingTimerInterval = setInterval(() => {
      if (!isActivelyRecording) return;
      const elapsedSec = Math.floor((Date.now() - recordingStartTime) / 1000);
      if (elements.recPillTimer) {
        elements.recPillTimer.textContent = formatTime(elapsedSec);
      }
    }, 1000);

    console.log(`[Recording] Started 2-way call recording session: ${lastRecordedSessionId}`);
  }

  function stopCallRecording() {
    if (!isActivelyRecording && recordedPcmChunks.length === 0) return;

    isActivelyRecording = false;
    if (recordingTimerInterval) {
      clearInterval(recordingTimerInterval);
      recordingTimerInterval = null;
    }

    const elapsedSec = Math.max(1, Math.floor((Date.now() - recordingStartTime) / 1000));
    const durationFormatted = formatTime(elapsedSec);

    if (elements.recStatusPill) elements.recStatusPill.style.display = 'none';

    // Only process if we actually have recorded audio chunks
    if (recordedPcmChunks.length > 0) {
      const sampleRate = audioContext ? audioContext.sampleRate : 24000;
      lastRecordedAudioBlob = encodeWAV(recordedPcmChunks, sampleRate);
      lastRecordedTranscriptText = formatTranscriptText(lastRecordedSessionId, durationFormatted, conversationHistory);

      console.log(`[Recording] Audio compiled: ${lastRecordedAudioBlob.size} bytes. Transcript: ${lastRecordedTranscriptText.length} chars.`);

      // Upload to server archive asynchronously
      uploadRecordingToServer(lastRecordedSessionId, lastRecordedAudioBlob, lastRecordedTranscriptText);

      // Open Recording Ready Modal for user preview and instant download
      openRecModal(durationFormatted, conversationHistory.length);
    }
  }

  async function uploadRecordingToServer(sessionId, audioBlob, transcriptText) {
    if (elements.recServerMsg) {
      elements.recServerMsg.textContent = 'Auto-saving audio (.wav) and transcript (.txt) to server recordings/ folder...';
    }
    try {
      const formData = new FormData();
      formData.append('audio_file', audioBlob, `call_audio_${sessionId}.wav`);
      formData.append('transcript_text', transcriptText);
      formData.append('session_id', sessionId);
      formData.append('voice', currentVoice);
      formData.append('model', currentModel);
      formData.append('persona', activeInstruction ? activeInstruction.name : 'Mintx Solutions');

      const res = await fetch('/api/recordings/save', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok && data.status === 'ok') {
        if (elements.recServerMsg) {
          elements.recServerMsg.textContent = `✅ Saved to server recordings/: ${data.audio_filename} & ${data.transcript_filename}`;
        }
        console.log('[Recording Upload Success]:', data);
      } else {
        if (elements.recServerMsg) {
          elements.recServerMsg.textContent = 'Note: Local files ready for download (server save returned warning).';
        }
      }
    } catch (e) {
      console.warn('[Recording Server Upload Notice]:', e);
      if (elements.recServerMsg) {
        elements.recServerMsg.textContent = 'Note: Audio and transcript ready for direct local download.';
      }
    }
  }

  function openRecModal(durationFormatted, turnsCount) {
    if (!elements.recModal) return;

    if (elements.recSummaryDuration) elements.recSummaryDuration.textContent = durationFormatted;
    if (elements.recSummaryTurns) elements.recSummaryTurns.textContent = `${turnsCount} turns`;
    if (elements.recSummaryVoice) {
      const v = VOICE_INFO[currentVoice];
      elements.recSummaryVoice.textContent = v ? v.name : currentVoice;
    }

    if (lastRecordedAudioBlob && elements.recAudioElement) {
      if (lastRecordedAudioUrl) {
        URL.revokeObjectURL(lastRecordedAudioUrl);
      }
      lastRecordedAudioUrl = URL.createObjectURL(lastRecordedAudioBlob);
      elements.recAudioElement.src = lastRecordedAudioUrl;

      const sizeKb = (lastRecordedAudioBlob.size / 1024).toFixed(1);
      if (elements.recAudioSize) elements.recAudioSize.textContent = `${sizeKb} KB`;
    }

    if (lastRecordedTranscriptText && elements.recTranscriptPre) {
      elements.recTranscriptPre.textContent = lastRecordedTranscriptText;
      const txtKb = (new Blob([lastRecordedTranscriptText]).size / 1024).toFixed(1);
      if (elements.recTxtSize) elements.recTxtSize.textContent = `${txtKb} KB`;
    }

    elements.recModal.style.display = 'flex';
    if (elements.recModalBackdrop) elements.recModalBackdrop.style.display = 'block';
    requestAnimationFrame(() => {
      elements.recModal.classList.add('open');
      if (elements.recModalBackdrop) elements.recModalBackdrop.classList.add('open');
    });
  }

  function closeRecModal() {
    if (!elements.recModal) return;
    if (elements.recAudioElement) {
      elements.recAudioElement.pause();
    }
    elements.recModal.classList.remove('open');
    if (elements.recModalBackdrop) elements.recModalBackdrop.classList.remove('open');
    setTimeout(() => {
      elements.recModal.style.display = 'none';
      if (elements.recModalBackdrop) elements.recModalBackdrop.style.display = 'none';
    }, 200);
  }

  function setupRecordingHandlers() {
    // 1. Manual Recording Toggle in Control Island
    if (elements.btnToggleRecord) {
      elements.btnToggleRecord.addEventListener('click', () => {
        if (isActivelyRecording) {
          stopCallRecording();
          isConversationRecordingEnabled = false;
          elements.btnToggleRecord.classList.remove('rec-active');
          elements.btnToggleRecord.classList.add('rec-disabled');
          elements.btnToggleRecord.setAttribute('data-tooltip', 'Recording: OFF (Click to turn ON)');
          showToast('⏹️ Call recording paused.');
        } else {
          isConversationRecordingEnabled = true;
          elements.btnToggleRecord.classList.remove('rec-disabled');
          elements.btnToggleRecord.classList.add('rec-active');
          elements.btnToggleRecord.setAttribute('data-tooltip', 'Recording: ON (Click to pause)');
          if (isConnected) {
            startCallRecording();
            showToast('🔴 Recording 2-way conversation audio...');
          } else {
            showToast('Recording armed: Will record when call starts.');
          }
        }
      });
    }

    // 2. Download Transcript .txt from History Drawer
    if (elements.btnDownloadTxt) {
      elements.btnDownloadTxt.addEventListener('click', () => {
        if (!conversationHistory || conversationHistory.length === 0) {
          showToast('No dialogue to download yet. Start talking first!');
          return;
        }
        const sid = lastRecordedSessionId || 'transcript_' + new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
        const txt = formatTranscriptText(sid, 'History Drawer Export', conversationHistory);
        downloadTextFile(txt, `mintx_call_transcript_${sid}.txt`);
        showToast('📄 Downloaded formatted transcript (.txt)');
      });
    }

    // 3. Recording Modal Download Buttons
    if (elements.btnDownloadAudioFile) {
      elements.btnDownloadAudioFile.addEventListener('click', () => {
        if (!lastRecordedAudioBlob) {
          showToast('No audio recording available to download.');
          return;
        }
        downloadBlob(lastRecordedAudioBlob, `mintx_call_audio_${lastRecordedSessionId}.wav`);
        showToast('🎵 Downloaded conversation audio (.wav)');
      });
    }

    if (elements.btnDownloadTxtFile) {
      elements.btnDownloadTxtFile.addEventListener('click', () => {
        if (!lastRecordedTranscriptText) {
          showToast('No transcript text available to download.');
          return;
        }
        downloadTextFile(lastRecordedTranscriptText, `mintx_call_transcript_${lastRecordedSessionId}.txt`);
        showToast('📄 Downloaded formatted transcript (.txt)');
      });
    }

    if (elements.btnDownloadBoth) {
      elements.btnDownloadBoth.addEventListener('click', () => {
        if (lastRecordedAudioBlob) {
          downloadBlob(lastRecordedAudioBlob, `mintx_call_audio_${lastRecordedSessionId}.wav`);
        }
        if (lastRecordedTranscriptText) {
          setTimeout(() => {
            downloadTextFile(lastRecordedTranscriptText, `mintx_call_transcript_${lastRecordedSessionId}.txt`);
          }, 350);
        }
        showToast('📦 Downloading both Audio (.wav) and Transcript (.txt)!');
      });
    }

    // 4. Modal Close Handlers
    if (elements.btnCloseRecModal) {
      elements.btnCloseRecModal.addEventListener('click', closeRecModal);
    }
    if (elements.btnRecModalClose) {
      elements.btnRecModalClose.addEventListener('click', closeRecModal);
    }
    if (elements.recModalBackdrop) {
      elements.recModalBackdrop.addEventListener('click', closeRecModal);
    }
  }

  // ── Instructions & Personas Manager Controller ────────────────────────────

  const STARTER_TEMPLATES = {
    mintx: `You are the AI Phone Assistant for **Mintx Solutions**, a professional software development company.

About Mintx Solutions:
- Services: Custom Websites, Mobile Apps (Android/iOS), Web Applications, E-Commerce, UI/UX Design, Cloud Hosting, Maintenance.
- Technologies: React, Next.js, Flutter, Python, Node.js, MongoDB, AWS.
- Website: www.mintxsolutions.com

Language Rules:
1. Speak fluently in **English** and **Tamil**.
2. If the customer speaks Tamil or Tanglish (e.g. "website pannanum", "enakku oru app venum", "price evvalavu"), reply in polite, natural Tamil.
3. If the customer speaks English, reply in English.
4. Keep responses concise (1 to 2 short sentences), natural, and clear for real-time voice speech over the phone.

Customer Requirement Gathering Flow:
1. Greet politely as Mintx Solutions assistant.
2. Ask what they want to build (Website, Mobile App, E-commerce, etc.).
3. Ask about their business / project purpose.
4. Ask what key features they need (e.g., payment gateway, user login, admin panel).
5. Ask for their preferred timeline or budget.
6. Ask for their name and phone number to send the project proposal.
7. Confirm the requirements and reassure them: "Our team will contact you within 24 hours with a detailed proposal."`,

    support: `You are a warm, polite, and efficient Customer Support & Appointment Booking Assistant.

Core Responsibilities:
1. Greet callers warmly and listen carefully to their queries or issues.
2. If the caller wants to book an appointment or callback, collect their name, preferred date/time, phone number, and purpose.
3. Answer common questions clearly in 1 to 2 concise sentences suitable for voice phone calls.
4. If a problem requires technical escalation, confirm: "I have recorded your request and our support team will follow up promptly."
5. Speak fluently in English and Tamil. If the user speaks Tamil, reply respectfully in Tamil.`,

    tamil: `You are the primary Tamil AI Voice Assistant for Mintx Solutions (Tamil Voice Assistant).

Rules:
1. Always respond in polite, natural spoken Tamil or Tanglish.
2. Patiently listen to customer requirements for Website, Mobile App, or Software.
3. Keep answers concise in 1 or 2 simple sentences.
4. Obtain the customer's name and contact number, and assure them our technical team will reach out promptly.`,

    concise: `You are an ultra-concise AI Phone Assistant.
Rules:
1. Limit every response to exactly 1 or 2 short sentences.
2. Do not ramble, use bullet points, or list items.
3. Keep the conversation dynamic, snappy, and clear for phone audio.
4. Respond in English or Tamil depending on the caller's language.`,

    blank: `You are a helpful AI assistant.
Rules:
1. Greet the user politely.
2. Keep responses concise (1 to 2 short sentences).
3. Be clear, natural, and helpful.`
  };

  async function loadMainServerSettings(forceSync = false) {
    try {
      let res = await fetch('/api/models/settings');
      let data;
      if (res.ok) {
        data = await res.json();
      } else {
        res = await fetch('/api/voice/config');
        if (res.ok) data = await res.json();
      }
      if (data && data.status === 'ok') {
        if (data.reasoning_model && (forceSync || !localStorage.getItem('preferred_reasoning_model'))) {
          currentReasoningModel = data.reasoning_model;
          localStorage.setItem('preferred_reasoning_model', currentReasoningModel);
        }
        const serverStt = data.transcription_model || data.stt_model;
        if (serverStt) {
          const storedStt = localStorage.getItem('preferred_stt_model');
          const keepSaaras = String(storedStt || '').startsWith('saaras');
          if (!keepSaaras && (forceSync || !storedStt || storedStt === 'gpt-realtime-whisper' || storedStt === 'whisper-1' || storedStt === 'gpt-transcribe')) {
            currentSttModel = serverStt;
            localStorage.setItem('preferred_stt_model', currentSttModel);
          }
        }
        if (data.sarvam) {
          if (data.sarvam.speaker && (forceSync || !localStorage.getItem('preferred_sarvam_speaker'))) {
            currentSarvamSpeaker = data.sarvam.speaker;
            localStorage.setItem('preferred_sarvam_speaker', currentSarvamSpeaker);
          }
          if (data.sarvam.model && (forceSync || !localStorage.getItem('preferred_sarvam_model'))) {
            currentSarvamModel = data.sarvam.model;
            localStorage.setItem('preferred_sarvam_model', currentSarvamModel);
          }
          if (data.sarvam.pace && (forceSync || !localStorage.getItem('preferred_sarvam_pace'))) {
            currentSarvamPace = parseFloat(data.sarvam.pace);
            localStorage.setItem('preferred_sarvam_pace', currentSarvamPace);
          }
          if (data.sarvam.language_code && (forceSync || !localStorage.getItem('preferred_sarvam_language'))) {
            currentSarvamLanguage = data.sarvam.language_code;
            localStorage.setItem('preferred_sarvam_language', currentSarvamLanguage);
          }
        }
        currentVoice = toLiveVoice(currentVoice);
        localStorage.setItem('preferred_voice', currentVoice);

        if (data.embedding_model && (forceSync || !localStorage.getItem('preferred_embedding_model'))) {
          currentEmbeddingModel = data.embedding_model;
          localStorage.setItem('preferred_embedding_model', currentEmbeddingModel);
        }
        if (data.vector_db && (forceSync || !localStorage.getItem('preferred_vector_db'))) {
          currentVectorDb = data.vector_db;
          localStorage.setItem('preferred_vector_db', currentVectorDb);
        }

        syncVoiceUI(currentVoice);
        updateArchStrip();
        if (elements.badgeReasoningVal) elements.badgeReasoningVal.textContent = `🧠 ${currentReasoningModel}`;
        if (elements.modelBadge) elements.modelBadge.textContent = currentModel;
        if (elements.summaryReasoningModel) elements.summaryReasoningModel.textContent = currentReasoningModel;
        if (elements.summaryRealtimeModel) elements.summaryRealtimeModel.textContent = currentModel;
        if (elements.summarySttModel) elements.summarySttModel.textContent = 'GPT-Live (built-in)';
        if (elements.summaryTtsEngine) elements.summaryTtsEngine.textContent = `GPT-Live voice (${currentVoice})`;

        if (forceSync) {
          showToast('✅ Synced UI with server .env main settings!');
          openSettingsModal();
        }
      }
    } catch (e) {
      console.warn('[Config Sync] Failed to fetch server voice config:', e);
    }
  }

  async function loadInstructions() {
    try {
      const res = await fetch('/api/instructions');
      if (!res.ok) throw new Error('Failed to load instructions');
      const data = await res.json();
      instructionsList = data.items || [];
      activeInstructionId = data.active_id || '';
      activeInstruction = data.active || (instructionsList.find(i => i.id === activeInstructionId) || instructionsList[0]);

      updateActiveInstructionUI();
      renderInstructionsGrid();
      renderSettingsPresets();
    } catch (e) {
      console.warn('[Instructions] Error loading instructions:', e);
    }
  }

  function updateActiveInstructionUI() {
    if (!activeInstruction) return;
    if (elements.badgeInstructionVal) {
      elements.badgeInstructionVal.textContent = activeInstruction.name;
    }
    if (elements.customInstructions) {
      elements.customInstructions.value = activeInstruction.instructions;
    }
    if (elements.selectInstructionPreset) {
      elements.selectInstructionPreset.value = activeInstruction.id;
    }
    if (elements.stageName) {
      elements.stageName.textContent = activeInstruction.name;
    }
    if (elements.stageCategory) {
      elements.stageCategory.textContent = activeInstruction.category || 'Custom';
    }
    if (elements.stagePreview) {
      elements.stagePreview.textContent = activeInstruction.instructions;
    }

    const greetingText = activeInstruction.starting_greeting || 'Vanakkam! MintX Solutions-ku welcome. Enna service venum nu sollunga?';
    if (elements.greetingPhraseDisplay) {
      elements.greetingPhraseDisplay.textContent = `"${greetingText}"`;
    }
    if (elements.greetingInputField) {
      elements.greetingInputField.value = greetingText;
    }
  }

  async function saveStartingGreeting(phrase) {
    if (!phrase || !phrase.trim()) return;
    phrase = phrase.trim();
    try {
      showToast('Updating starting greeting...', false);
      const res = await fetch('/api/instructions/starting-greeting', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ greeting: phrase, instruction_id: activeInstructionId })
      });
      if (!res.ok) throw new Error('Failed to update starting greeting');
      const data = await res.json();
      const updatedGreeting = data.starting_greeting || phrase;

      if (elements.greetingPhraseDisplay) {
        elements.greetingPhraseDisplay.textContent = `"${updatedGreeting}"`;
      }
      if (elements.greetingInputField) {
        elements.greetingInputField.value = updatedGreeting;
      }
      if (elements.greetingEditDrawer) {
        elements.greetingEditDrawer.style.display = 'none';
      }

      // Hot reload realtime session via WebSocket
      if (ws && ws.readyState === WebSocket.OPEN) {
        sendWsEvent({
          type: 'client.set_starting_greeting',
          greeting: updatedGreeting,
          instruction_id: activeInstructionId
        });
      }

      showToast(`Starting greeting updated!`, false);
      // Reload instructions list in background to sync state
      loadInstructions();
    } catch (err) {
      console.error('[Save Greeting Error]:', err);
      showToast(`Failed: ${err.message}`, true);
    }
  }

  function testStartingGreeting() {
    const greeting = (elements.greetingInputField && elements.greetingInputField.value.trim()) ||
                     (activeInstruction && activeInstruction.starting_greeting) ||
                     'Vanakkam! MintX Solutions-ku welcome. Enna service venum nu sollunga?';

    if (ws && ws.readyState === WebSocket.OPEN) {
      sendWsEvent({
        type: 'client.test_greeting',
        greeting: greeting
      });
      showToast('AI speaking starting words...', false);
    } else {
      // Offline / browser TTS test
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(greeting);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
        showToast('Speaking test greeting in browser...', false);
      } else {
        showToast(`Greeting: "${greeting}"`, false);
      }
    }
  }

  function renderSettingsPresets() {
    if (!elements.selectInstructionPreset) return;
    elements.selectInstructionPreset.innerHTML = '';
    instructionsList.forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.id;
      opt.textContent = `${item.name} (${item.category || 'Custom'})`;
      if (item.id === activeInstructionId) {
        opt.selected = true;
      }
      elements.selectInstructionPreset.appendChild(opt);
    });
  }

  function renderInstructionsGrid() {
    if (!elements.instructionsGrid) return;
    if (elements.instructionCount) {
      elements.instructionCount.textContent = instructionsList.length;
    }
    elements.instructionsGrid.innerHTML = '';

    instructionsList.forEach((item) => {
      const isActive = item.id === activeInstructionId;
      const card = document.createElement('div');
      card.className = `instruction-card ${isActive ? 'active' : ''}`;
      card.dataset.id = item.id;

      card.innerHTML = `
        <div class="card-header-row">
          <div class="card-title-group">
            <h4 class="card-title">${escapeHtml(item.name)}</h4>
            <div class="card-meta">
              <span class="card-category-badge">${escapeHtml(item.category || 'Custom')}</span>
              ${isActive ? '<span class="card-active-pill"><span class="dot"></span>ACTIVE</span>' : ''}
            </div>
          </div>
        </div>
        <p class="card-desc">${escapeHtml(item.description || 'No description provided.')}</p>
        <div class="card-preview">${escapeHtml(item.instructions)}</div>
        <div class="card-actions">
          <div class="card-actions-left">
            <button type="button" class="btn-card-action btn-card-edit" title="Modify Instruction">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
              Modify
            </button>
            ${`
              <button type="button" class="btn-card-action btn-card-delete" title="Delete Instruction">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                Delete
              </button>
            `}
          </div>
          <button type="button" class="btn-card-action btn-card-activate ${isActive ? 'is-active' : ''}">
            ${isActive ? '✓ Active' : 'Set Active'}
          </button>
        </div>
      `;

      // Event listeners for card buttons
      const editBtn = card.querySelector('.btn-card-edit');
      if (editBtn) {
        editBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          openInstructionEditor(item);
        });
      }

      const deleteBtn = card.querySelector('.btn-card-delete');
      if (deleteBtn) {
        deleteBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          deleteInstructionItem(item.id, item.name);
        });
      }

      const activateBtn = card.querySelector('.btn-card-activate');
      if (activateBtn && !isActive) {
        activateBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          activateInstruction(item.id);
        });
      }

      elements.instructionsGrid.appendChild(card);
    });
  }

  function setupInstructionHandlers() {
    // Open/close instructions modal
    if (elements.btnOpenInstructions) {
      elements.btnOpenInstructions.addEventListener('click', openInstructionsModal);
    }
    if (elements.btnCloseInstructions) {
      elements.btnCloseInstructions.addEventListener('click', closeInstructionsModal);
    }
    if (elements.instructionsBackdrop) {
      elements.instructionsBackdrop.addEventListener('click', (e) => {
        if (e.target === elements.instructionsBackdrop) {
          closeInstructionsModal();
        }
      });
    }

    // Global keyboard shortcut: Ctrl+I or Alt+I to toggle instructions modal
    window.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.altKey) && (e.key === 'i' || e.key === 'I')) {
        e.preventDefault();
        if (elements.instructionsBackdrop && elements.instructionsBackdrop.classList.contains('open')) {
          closeInstructionsModal();
        } else {
          openInstructionsModal();
        }
      }
    });

    // Main stage instruction card actions
    if (elements.btnStageModify) {
      elements.btnStageModify.addEventListener('click', (e) => {
        e.stopPropagation();
        openInstructionEditor(activeInstruction);
        openInstructionsModal();
      });
    }
    if (elements.btnStageAdd) {
      elements.btnStageAdd.addEventListener('click', (e) => {
        e.stopPropagation();
        openInstructionEditor(null);
        openInstructionsModal();
      });
    }
    if (elements.stageCard) {
      elements.stageCard.addEventListener('click', () => {
        openInstructionsModal();
      });
    }

    // Link from Settings modal
    if (elements.btnSettingsManageInstructions) {
      elements.btnSettingsManageInstructions.addEventListener('click', () => {
        closeSettingsModal();
        openInstructionsModal();
      });
    }

    // Preset dropdown change in Settings modal
    if (elements.selectInstructionPreset) {
      elements.selectInstructionPreset.addEventListener('change', () => {
        const selectedId = elements.selectInstructionPreset.value;
        const found = instructionsList.find(i => i.id === selectedId);
        if (found && elements.customInstructions) {
          elements.customInstructions.value = found.instructions;
        }
      });
    }

    // Tab buttons
    if (elements.tabBtnList) {
      elements.tabBtnList.addEventListener('click', () => switchInstructionsTab('list'));
    }
    if (elements.tabBtnEditor) {
      elements.tabBtnEditor.addEventListener('click', () => {
        if (elements.tabBtnEditor.classList.contains('active')) return;
        openInstructionEditor(null); // Add mode
      });
    }

    // Toolbar "+ Add New"
    if (elements.btnToolbarAdd) {
      elements.btnToolbarAdd.addEventListener('click', () => openInstructionEditor(null));
    }

    // Editor form cancel
    if (elements.btnEditorCancel) {
      elements.btnEditorCancel.addEventListener('click', () => switchInstructionsTab('list'));
    }

    // Editor form save buttons
    if (elements.btnEditorSave) {
      elements.btnEditorSave.addEventListener('click', () => saveInstruction(false));
    }
    if (elements.btnEditorSaveActivate) {
      elements.btnEditorSaveActivate.addEventListener('click', () => saveInstruction(true));
    }

    // Live character counter for prompt textarea
    if (elements.editorPrompt) {
      elements.editorPrompt.addEventListener('input', updateCharCount);
    }

    // Starter template chip buttons
    if (elements.templateChips) {
      elements.templateChips.forEach((chip) => {
        chip.addEventListener('click', () => {
          const tplKey = chip.getAttribute('data-template');
          if (tplKey && STARTER_TEMPLATES[tplKey] !== undefined) {
            elements.editorPrompt.value = STARTER_TEMPLATES[tplKey];
            updateCharCount();
            if (!elements.editorName.value.trim()) {
              if (tplKey === 'mintx') elements.editorName.value = 'Mintx Solutions Sales';
              if (tplKey === 'support') elements.editorName.value = 'Customer Support Desk';
              if (tplKey === 'tamil') elements.editorName.value = 'Tamil Voice Assistant';
              if (tplKey === 'concise') elements.editorName.value = 'Fast Telephony Agent';
            }
            showToast(`Loaded ${chip.textContent.trim()} template`);
          }
        });
      });
    }

    // Starting Words & Greeting controls
    if (elements.btnEditGreeting) {
      elements.btnEditGreeting.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!elements.greetingEditDrawer) return;
        const isHidden = elements.greetingEditDrawer.style.display === 'none' || !elements.greetingEditDrawer.style.display;
        elements.greetingEditDrawer.style.display = isHidden ? 'flex' : 'none';
        if (isHidden && elements.greetingInputField) {
          elements.greetingInputField.focus();
          elements.greetingInputField.select();
        }
      });
    }

    if (elements.btnCancelGreeting) {
      elements.btnCancelGreeting.addEventListener('click', () => {
        if (elements.greetingEditDrawer) {
          elements.greetingEditDrawer.style.display = 'none';
        }
      });
    }

    if (elements.btnSaveGreeting) {
      elements.btnSaveGreeting.addEventListener('click', () => {
        if (elements.greetingInputField) {
          saveStartingGreeting(elements.greetingInputField.value);
        }
      });
    }

    if (elements.greetingInputField) {
      elements.greetingInputField.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          saveStartingGreeting(elements.greetingInputField.value);
        }
      });
    }

    if (elements.btnTestGreeting) {
      elements.btnTestGreeting.addEventListener('click', (e) => {
        e.stopPropagation();
        testStartingGreeting();
      });
    }

    if (elements.chipGreetingPresets) {
      elements.chipGreetingPresets.forEach((chip) => {
        chip.addEventListener('click', (e) => {
          e.stopPropagation();
          const phrase = chip.getAttribute('data-phrase');
          if (phrase) {
            saveStartingGreeting(phrase);
          }
        });
      });
    }
  }

  function updateCharCount() {
    if (elements.editorPrompt && elements.editorCharCount) {
      const len = elements.editorPrompt.value.length;
      elements.editorCharCount.textContent = `${len.toLocaleString()} chars`;
    }
  }

  function openInstructionsModal() {
    elements.instructionsBackdrop.style.display = 'flex';
    requestAnimationFrame(() => {
      elements.instructionsBackdrop.classList.add('open');
    });
  }

  function closeInstructionsModal() {
    elements.instructionsBackdrop.classList.remove('open');
    setTimeout(() => {
      elements.instructionsBackdrop.style.display = 'none';
      switchInstructionsTab('list');
    }, 250);
  }

  function switchInstructionsTab(tab) {
    if (tab === 'list') {
      elements.tabBtnList.classList.add('active');
      elements.tabBtnEditor.classList.remove('active');
      elements.paneInstructionsList.style.display = 'flex';
      elements.paneInstructionsEditor.style.display = 'none';
    } else {
      elements.tabBtnList.classList.remove('active');
      elements.tabBtnEditor.classList.add('active');
      elements.paneInstructionsList.style.display = 'none';
      elements.paneInstructionsEditor.style.display = 'flex';
      elements.editorName.focus();
    }
  }

  function openInstructionEditor(item = null) {
    if (item) {
      // Modify Mode
      elements.editorInstructionId.value = item.id;
      elements.editorName.value = item.name;
      elements.editorCategory.value = item.category || 'Custom';
      elements.editorDesc.value = item.description || '';
      elements.editorPrompt.value = item.instructions;
      elements.tabEditorTitle.textContent = `✏️ Modify: ${item.name.substring(0, 16)}...`;
    } else {
      // Add Mode
      elements.editorInstructionId.value = '';
      elements.editorName.value = '';
      elements.editorCategory.value = 'Custom';
      elements.editorDesc.value = '';
      elements.editorPrompt.value = '';
      elements.tabEditorTitle.textContent = '➕ Add New Instruction';
    }
    updateCharCount();
    switchInstructionsTab('editor');
  }

  async function saveInstruction(setActivate = false) {
    const id = elements.editorInstructionId.value.trim();
    const name = elements.editorName.value.trim();
    const category = elements.editorCategory.value;
    const description = elements.editorDesc.value.trim();
    const promptText = elements.editorPrompt.value.trim();

    if (!name) {
      alert('Please enter a name for the instruction persona.');
      elements.editorName.focus();
      return;
    }
    if (!promptText) {
      alert('Please enter prompt instructions.');
      elements.editorPrompt.focus();
      return;
    }

    const payload = {
      name,
      category,
      description,
      instructions: promptText,
      set_active: setActivate,
    };

    try {
      let res;
      if (id) {
        // Update / Modify existing
        res = await fetch(`/api/instructions/${encodeURIComponent(id)}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      } else {
        // Create / Add new
        res = await fetch('/api/instructions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      }

      const result = await res.json();
      if (!res.ok || result.status !== 'ok') {
        throw new Error(result.detail || 'Failed to save instruction');
      }

      await loadInstructions();

      if (setActivate || result.active_id === id) {
        if (isConnected) {
          sendWsEvent({
            type: 'session.update',
            voice: currentVoice,
            model: currentModel,
            audio_mode: audioMode,
            instructions: promptText,
          });
        }
        showToast(`Saved and activated "${name}"!`);
      } else {
        showToast(id ? `Updated "${name}" successfully!` : `Created "${name}" successfully!`);
      }

      switchInstructionsTab('list');
    } catch (e) {
      console.error('[Instructions] Error saving:', e);
      alert(`Error saving instruction: ${e.message}`);
    }
  }

  async function activateInstruction(id, notify = true) {
    try {
      const res = await fetch('/api/instructions/active', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      });

      const result = await res.json();
      if (!res.ok || result.status !== 'ok') {
        throw new Error(result.detail || 'Failed to activate instruction');
      }

      activeInstructionId = id;
      activeInstruction = result.active;
      updateActiveInstructionUI();
      renderInstructionsGrid();
      renderSettingsPresets();

      // If connected to OpenAI Realtime live session, dynamically update live session!
      if (isConnected && activeInstruction) {
        sendWsEvent({
          type: 'session.update',
          voice: currentVoice,
          model: currentModel,
          audio_mode: audioMode,
          instructions: activeInstruction.instructions,
        });
      }

      if (notify) {
        showToast(`Activated instruction: "${activeInstruction.name}"`);
      }
    } catch (e) {
      console.error('[Instructions] Error activating:', e);
      alert(`Could not activate instruction: ${e.message}`);
    }
  }

  async function updateActiveInstructionPrompt(newPrompt) {
    if (!newPrompt || !newPrompt.trim()) return;
    newPrompt = newPrompt.trim();
    try {
      if (activeInstruction && activeInstructionId) {
        const payload = {
          name: activeInstruction.name || 'Custom Persona',
          category: activeInstruction.category || 'Custom',
          description: activeInstruction.description || '',
          instructions: newPrompt,
          set_active: true,
        };
        const res = await fetch(`/api/instructions/${encodeURIComponent(activeInstructionId)}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const result = await res.json();
        if (res.ok && result.status === 'ok') {
          activeInstruction.instructions = newPrompt;
          updateActiveInstructionUI();
          showToast(`✅ Active prompt updated directly from UI!`);
          return;
        }
      }

      // If no active instruction exists, create and activate a custom one
      const res = await fetch('/api/instructions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: 'Custom UI Persona',
          category: 'Custom',
          description: 'Created directly from UI settings',
          instructions: newPrompt,
          set_active: true,
        }),
      });
      const result = await res.json();
      if (res.ok && result.status === 'ok') {
        activeInstructionId = result.item.id;
        activeInstruction = result.item;
        updateActiveInstructionUI();
        showToast(`✅ Active prompt created and saved from UI!`);
      }
    } catch (e) {
      console.warn('[Instructions] Failed to persist prompt update:', e);
    }
  }

  async function deleteInstructionItem(id, name) {
    if (!confirm(`Are you sure you want to delete "${name}"?`)) return;

    try {
      const res = await fetch(`/api/instructions/${encodeURIComponent(id)}`, {
        method: 'DELETE',
      });
      const result = await res.json();
      if (!res.ok || result.status !== 'ok') {
        throw new Error(result.detail || 'Failed to delete');
      }

      showToast(`Deleted instruction "${name}"`);
      await loadInstructions();
    } catch (e) {
      console.error('[Instructions] Error deleting:', e);
      alert(`Could not delete instruction: ${e.message}`);
    }
  }

  // ==========================================================================
  // Voice Training & Interactive Test Studio Implementation
  // ==========================================================================

  function setOperatingMode(mode) {
    currentOperatingMode = mode;
    if (elements.modeBtnLive && elements.modeBtnTest) {
      if (mode === 'test') {
        elements.modeBtnTest.classList.add('active');
        elements.modeBtnLive.classList.remove('active');
        if (elements.voiceTrainConsole) elements.voiceTrainConsole.style.display = 'flex';
      } else {
        elements.modeBtnLive.classList.add('active');
        elements.modeBtnTest.classList.remove('active');
        if (elements.voiceTrainConsole) elements.voiceTrainConsole.style.display = 'none';
      }
    }
  }

  function setupVoiceTrainingHandlers() {
    // Mode Switcher
    if (elements.modeBtnLive) {
      elements.modeBtnLive.addEventListener('click', () => {
        setOperatingMode('live');
        showToast('Switched to Live Call Mode 📞');
      });
    }
    if (elements.modeBtnTest) {
      elements.modeBtnTest.addEventListener('click', () => {
        setOperatingMode('test');
        showToast('Switched to 🧪 Test & Voice Train Mode');
      });
    }

    // Main Teach Mic Button
    if (elements.btnVoiceTeach) {
      elements.btnVoiceTeach.addEventListener('click', () => {
        if (!isRecordingVoiceCorrection) {
          startVoiceTeachRecording();
        } else {
          stopVoiceTeachRecording();
        }
      });
    }

    // Quick Action: Cancel / Apply manual edit input
    if (elements.btnCancelVoiceCorrection) {
      elements.btnCancelVoiceCorrection.addEventListener('click', () => {
        if (elements.teachQuickActions) elements.teachQuickActions.style.display = 'none';
        elements.teachLiveText.innerHTML = 'Click the button to speak your training correction.';
      });
    }

    if (elements.btnApplyVoiceCorrection) {
      elements.btnApplyVoiceCorrection.addEventListener('click', () => {
        const text = elements.teachEditInput ? elements.teachEditInput.value.trim() : '';
        if (text) {
          applyVoiceCorrectionText(text);
          if (elements.teachQuickActions) elements.teachQuickActions.style.display = 'none';
        }
      });
    }

    // Training Preset Chips
    if (elements.trainChipBtns) {
      elements.trainChipBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
          const correction = btn.getAttribute('data-correction');
          if (correction) {
            elements.teachLiveText.innerHTML = `Applying preset: <em>"${escapeHtml(correction)}"</em>...`;
            applyVoiceCorrectionText(correction);
          }
        });
      });
    }

    // Undo Correction Button
    if (elements.btnUndoCorrection) {
      elements.btnUndoCorrection.addEventListener('click', async () => {
        await undoVoiceCorrection();
      });
    }

    // Close Training Alert Banner
    if (elements.btnCloseTrainAlert) {
      elements.btnCloseTrainAlert.addEventListener('click', () => {
        if (elements.trainAlertBanner) elements.trainAlertBanner.style.display = 'none';
      });
    }

    // History Modal Open / Close
    if (elements.btnViewTrainingHistory) {
      elements.btnViewTrainingHistory.addEventListener('click', () => {
        openTrainingHistoryModal();
      });
    }
    if (elements.btnCloseTrainHistory) {
      elements.btnCloseTrainHistory.addEventListener('click', closeTrainingHistoryModal);
    }
    if (elements.btnCloseTrainHistoryFooter) {
      elements.btnCloseTrainHistoryFooter.addEventListener('click', closeTrainingHistoryModal);
    }
    if (elements.trainHistoryBackdrop) {
      elements.trainHistoryBackdrop.addEventListener('click', closeTrainingHistoryModal);
    }

    // ── Bridge Interactive Tell Correction Studio ──
    if (elements.btnBridgeApplyCorrection) {
      elements.btnBridgeApplyCorrection.addEventListener('click', () => {
        const text = elements.inputBridgeCorrection ? elements.inputBridgeCorrection.value.trim() : '';
        if (text) {
          elements.teachLiveText.innerHTML = `Applying correction: <em>"${escapeHtml(text)}"</em>...`;
          applyVoiceCorrectionText(text);
          if (elements.inputBridgeCorrection) elements.inputBridgeCorrection.value = '';
        } else {
          showToast('Please type or speak a correction rule.', true);
        }
      });
    }

    if (elements.inputBridgeCorrection) {
      elements.inputBridgeCorrection.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          if (elements.btnBridgeApplyCorrection) elements.btnBridgeApplyCorrection.click();
        }
      });
    }

    // Bridge Speak Correction Button
    if (elements.btnBridgeSpeakCorrection) {
      elements.btnBridgeSpeakCorrection.addEventListener('click', () => {
        if (!isRecordingVoiceCorrection) {
          startVoiceTeachRecording();
          if (elements.bridgeMicIcon) elements.bridgeMicIcon.textContent = '⏹️';
          if (elements.bridgeMicText) elements.bridgeMicText.textContent = 'Stop & Apply';
          elements.btnBridgeSpeakCorrection.classList.add('recording');
        } else {
          stopVoiceTeachRecording();
          if (elements.bridgeMicIcon) elements.bridgeMicIcon.textContent = '🎙️';
          if (elements.bridgeMicText) elements.bridgeMicText.textContent = 'Speak Correction';
          elements.btnBridgeSpeakCorrection.classList.remove('recording');
        }
      });
    }

    // Bridge Refresh Rules Button
    if (elements.btnBridgeRefreshRules) {
      elements.btnBridgeRefreshRules.addEventListener('click', () => {
        loadTrainingHistory();
        showToast('Refreshed active rules from instructions.json');
      });
    }

    // Stage Persona Rules Toggle & Direct Add
    if (elements.btnToggleStageRules) {
      elements.btnToggleStageRules.addEventListener('click', (e) => {
        e.stopPropagation();
        if (elements.stageRulesDrawer) {
          const isHidden = elements.stageRulesDrawer.style.display === 'none';
          elements.stageRulesDrawer.style.display = isHidden ? 'block' : 'none';
        }
      });
    }

    if (elements.btnStageAddRule) {
      elements.btnStageAddRule.addEventListener('click', async (e) => {
        e.stopPropagation();
        const rule = prompt('Enter a new rule or correction to save in instructions.json:\n(e.g., "Always ask customer for WhatsApp number" or "Website price Rs.15,000 nu sollu")');
        if (rule && rule.trim()) {
          await addCustomTrainingRuleDirect(rule.trim());
        }
      });
    }

    // Modal Add Custom Rule Form
    if (elements.btnSubmitNewRule) {
      elements.btnSubmitNewRule.addEventListener('click', async () => {
        const rule = elements.inputNewCustomRule ? elements.inputNewCustomRule.value.trim() : '';
        if (!rule) {
          showToast('Please type a rule to add.', true);
          return;
        }
        await addCustomTrainingRuleDirect(rule);
        if (elements.inputNewCustomRule) elements.inputNewCustomRule.value = '';
      });
    }

    if (elements.inputNewCustomRule) {
      elements.inputNewCustomRule.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          if (elements.btnSubmitNewRule) elements.btnSubmitNewRule.click();
        }
      });
    }

    // Live Response Correction Trigger (from live transcript ribbon)
    if (elements.btnCorrectLive) {
      elements.btnCorrectLive.addEventListener('click', () => {
        const liveText = elements.liveTranscriptText ? elements.liveTranscriptText.textContent.trim() : '';
        openResponseCorrectionModal(liveText);
      });
    }

    // Response Correction Modal Controls
    if (elements.btnCloseResponseCorrect) {
      elements.btnCloseResponseCorrect.addEventListener('click', closeResponseCorrectionModal);
    }
    if (elements.btnModalCancelCorrect) {
      elements.btnModalCancelCorrect.addEventListener('click', closeResponseCorrectionModal);
    }
    if (elements.responseCorrectBackdrop) {
      elements.responseCorrectBackdrop.addEventListener('click', closeResponseCorrectionModal);
    }
    if (elements.btnModalSubmitCorrect) {
      elements.btnModalSubmitCorrect.addEventListener('click', () => {
        const text = elements.modalCorrectionInput ? elements.modalCorrectionInput.value.trim() : '';
        if (!text) {
          showToast('Please speak or type a correction.', true);
          return;
        }
        applyVoiceCorrectionText(text, activeCorrectionTarget.responseText, activeCorrectionTarget.context);
        closeResponseCorrectionModal();
      });
    }
    if (elements.btnModalMic) {
      elements.btnModalMic.addEventListener('click', () => {
        toggleModalVoiceRecording();
      });
    }
  }

  // ── Voice Recording & Transcription Engine ───────────────────────────

  async function startVoiceTeachRecording() {
    try {
      lastRecognizedTeachSpeech = '';
      correctionAlreadyAppliedForCurrentRecording = false;

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      voiceTeachAudioChunks = [];

      const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
      voiceTeachMediaRecorder = new MediaRecorder(stream, { mimeType });

      voiceTeachMediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          voiceTeachAudioChunks.push(e.data);
        }
      };

      voiceTeachMediaRecorder.onstop = async () => {
        const audioBlob = new Blob(voiceTeachAudioChunks, { type: mimeType });
        stream.getTracks().forEach((t) => t.stop());
        if (!correctionAlreadyAppliedForCurrentRecording) {
          await processVoiceTeachAudio(audioBlob);
        }
      };

      voiceTeachMediaRecorder.start(250);
      isRecordingVoiceCorrection = true;

      elements.btnVoiceTeach.classList.add('recording');
      elements.teachBtnText.textContent = '⏹️ Listening... Click to Apply';
      elements.teachLiveText.className = 'teach-live-text active-recording';
      elements.teachLiveText.textContent = '🎙️ Listening... Speak your rule or correction!';

      startLiveSpeechCaptioning();
    } catch (err) {
      console.error('[Voice Recording Error]:', err);
      showToast('Microphone access needed for voice training.', true);
      elements.teachLiveText.textContent = 'Microphone access denied. You can type corrections below.';
      if (elements.teachQuickActions) elements.teachQuickActions.style.display = 'flex';
    }
  }

  function stopVoiceTeachRecording() {
    if (voiceTeachMediaRecorder && voiceTeachMediaRecorder.state !== 'inactive') {
      voiceTeachMediaRecorder.stop();
    }
    stopLiveSpeechCaptioning();

    isRecordingVoiceCorrection = false;
    elements.btnVoiceTeach.classList.remove('recording');
    elements.teachBtnText.textContent = '🎙️ Speak Correction';
    elements.teachLiveText.className = 'teach-live-text';

    // Instant fast-path: if speech was captured by live recognizer, apply immediately with 0ms delay!
    const capturedText = (lastRecognizedTeachSpeech || '').trim();
    if (capturedText) {
      correctionAlreadyAppliedForCurrentRecording = true;
      elements.teachLiveText.innerHTML = `⚡ <strong>Optimizing & learning rule...</strong>`;
      applyVoiceCorrectionText(capturedText);
    } else {
      elements.teachLiveText.textContent = 'Transcribing with Whisper and optimizing prompt...';
    }
  }

  // Live Speech Recognition for Instantaneous (0ms) Voice Correction
  function startLiveSpeechCaptioning() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) return;
    try {
      speechRecognizer = new SpeechRec();
      speechRecognizer.continuous = true;
      speechRecognizer.interimResults = true;
      speechRecognizer.lang = 'ta-IN'; // Recognizes both Tamil and English words

      speechRecognizer.onresult = (e) => {
        let transcript = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
          transcript += e.results[i][0].transcript;
        }
        if (transcript.trim()) {
          lastRecognizedTeachSpeech = transcript.trim();
          if (isRecordingVoiceCorrection) {
            elements.teachLiveText.innerHTML = `🎙️ Listening... <em>Captured spoken instruction</em>`;
          }
          if (elements.teachEditInput) {
            elements.teachEditInput.value = transcript.trim();
          }
        }
      };

      speechRecognizer.start();
    } catch (e) {
      // Non-fatal fallback to Whisper audio recorder
    }
  }

  function stopLiveSpeechCaptioning() {
    if (speechRecognizer) {
      try { speechRecognizer.stop(); } catch (e) {}
      speechRecognizer = null;
    }
  }

  async function processVoiceTeachAudio(audioBlob) {
    if (correctionAlreadyAppliedForCurrentRecording) return;

    const formData = new FormData();
    formData.append('audio', audioBlob, 'voice_correction.webm');
    formData.append('instruction_id', activeInstructionId || 'default-mintx');
    formData.append('last_response', currentAssistantTranscript);
    formData.append('context', conversationHistory.slice(-4).map((m) => `${m.role}: ${m.text}`).join('\n'));
    formData.append('auto_correct', 'true');

    try {
      showToast('Processing spoken instruction with Whisper...');
      const res = await fetch('/api/instructions/voice-transcribe', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok || data.status !== 'ok') {
        throw new Error(data.detail || 'Voice transcription failed');
      }

      const spokenText = data.transcript || '';
      if (!spokenText.trim()) {
        elements.teachLiveText.textContent = 'No clear speech detected. Please click to speak again.';
        return;
      }

      elements.teachLiveText.innerHTML = `⚡ <strong>Optimizing & learning rule with AI...</strong>`;
      if (data.correction && data.correction.status === 'ok') {
        handleCorrectionApplied(data.correction, spokenText);
      } else {
        await applyVoiceCorrectionText(spokenText);
      }
    } catch (err) {
      console.error('[Voice Processing Error]:', err);
      showToast(`Error: ${err.message}`, true);
      elements.teachLiveText.textContent = `Could not process voice: ${err.message}`;
    }
  }

  async function applyVoiceCorrectionText(feedbackText, targetResponse = '', context = '') {
    try {
      elements.teachLiveText.innerHTML = `⚡ <strong>Optimizing & learning rule with AI...</strong>`;
      showToast('Tuning prompt rules with AI...');

      const res = await fetch('/api/instructions/voice-correct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instruction_id: activeInstructionId,
          feedback: feedbackText,
          last_response: targetResponse || currentAssistantTranscript,
          context: context || conversationHistory.slice(-4).map((m) => `${m.role}: ${m.text}`).join('\n'),
        }),
      });

      const data = await res.json();
      if (!res.ok || data.status !== 'ok') {
        throw new Error(data.detail || 'Failed to refine prompt');
      }

      handleCorrectionApplied(data, feedbackText);
    } catch (e) {
      console.error('[Apply Correction Error]:', e);
      showToast(`Error: ${e.message}`, true);
      elements.teachLiveText.textContent = `Error: ${e.message}`;
    }
  }

  function handleCorrectionApplied(data, spokenFeedback, fromServer = false) {
    let summaryEn = data.summary_en || 'Instruction prompt updated.';
    if (/[\u0B80-\u0BFF]/.test(summaryEn)) {
      summaryEn = 'Instruction prompt updated.';
    }
    let diff = data.rule_diff || summaryEn;
    if (/[\u0B80-\u0BFF]/.test(diff)) {
      diff = summaryEn;
    }
    const newInstructions = data.new_instructions || (data.item && data.item.instructions);
    const verbalAck = data.verbal_ack || `Understood! I've updated the rule: ${diff}`;
    const startingGreeting = data.starting_greeting || (data.item && data.item.starting_greeting);

    // Update starting greeting UI card if returned
    if (startingGreeting) {
      if (elements.greetingPhraseDisplay) {
        elements.greetingPhraseDisplay.textContent = `"${startingGreeting}"`;
      }
      if (elements.greetingInputField) {
        elements.greetingInputField.value = startingGreeting;
      }
    }

    // 1. Hot-reload active Realtime WebSocket session (only if not already done on server via tool call)
    if (!fromServer && ws && ws.readyState === WebSocket.OPEN) {
      if (newInstructions) {
        sendWsEvent({
          type: 'session.update',
          voice: currentVoice,
          model: currentModel,
          audio_mode: audioMode,
          instructions: newInstructions,
        });
        console.log('[WebSocket] Hot-reloaded Realtime session with updated instructions!');
      }

      if (startingGreeting) {
        sendWsEvent({
          type: 'client.set_starting_greeting',
          greeting: startingGreeting,
          instruction_id: activeInstructionId
        });
        console.log('[WebSocket] Set active starting greeting on session:', startingGreeting);
      }

      // 2. Trigger Real-time AI Verbal Acknowledgment aloud over WebSocket!
      if (verbalAck) {
        sendWsEvent({
          type: 'client.speak_ack',
          text: verbalAck,
        });
        console.log('[WebSocket] Triggered AI real-time verbal acknowledgment:', verbalAck);
      }
    }

    // 3. Display Verbal Acknowledgment Banner on AI Assistant Side
    if (elements.verbalAckBanner && elements.verbalAckQuote) {
      elements.verbalAckQuote.textContent = `"${verbalAck}"`;
      elements.verbalAckBanner.style.display = 'flex';
      setTimeout(() => {
        if (elements.verbalAckBanner) {
          elements.verbalAckBanner.style.display = 'none';
        }
      }, 8000);
    }

    // 4. Refresh instructions data & active UI cards
    loadInstructions();
    loadTrainingHistory();

    // 5. Show flash alert banner in Console
    if (elements.trainAlertBanner) {
      elements.trainAlertBanner.style.display = 'flex';
      elements.trainAlertTitle.textContent = summaryEn;
      elements.trainAlertDiff.textContent = `New rule: "${diff}"`;
    }

    // 6. Update status display and enable Undo button
    elements.teachLiveText.innerHTML = `✅ <strong>Learned!</strong> Rule: <em>"${escapeHtml(diff)}"</em>`;
    if (elements.btnUndoCorrection) {
      elements.btnUndoCorrection.style.display = 'inline-flex';
    }

    showToast(`🎯 AI Rule Updated & Verbal Acknowledgment Spoken!`);
  }

  async function undoVoiceCorrection() {
    try {
      showToast('Reverting last voice correction...');
      const res = await fetch('/api/instructions/undo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction_id: activeInstructionId }),
      });
      const data = await res.json();
      if (!res.ok || data.status !== 'ok') {
        throw new Error(data.detail || 'Could not revert');
      }

      const revertedInstructions = data.item ? data.item.instructions : '';
      if (ws && ws.readyState === WebSocket.OPEN && revertedInstructions) {
        sendWsEvent({
          type: 'session.update',
          voice: currentVoice,
          model: currentModel,
          audio_mode: audioMode,
          instructions: revertedInstructions,
        });
      }

      loadInstructions();
      loadTrainingHistory();
      if (elements.btnUndoCorrection) elements.btnUndoCorrection.style.display = 'none';
      if (elements.trainAlertBanner) elements.trainAlertBanner.style.display = 'none';
      elements.teachLiveText.innerHTML = '⏪ Reverted to previous instruction revision.';
      showToast('Reverted to previous rule successfully.');
    } catch (e) {
      console.error('[Undo Error]:', e);
      showToast(`Undo failed: ${e.message}`, true);
    }
  }

  // ── Training History Log Drawer / Modal ──────────────────────────────

  async function loadTrainingHistory() {
    try {
      const res = await fetch('/api/instructions/history');
      const data = await res.json();
      if (data.status === 'ok' && Array.isArray(data.history)) {
        if (elements.trainHistoryCount) {
          elements.trainHistoryCount.textContent = data.history.length;
        }
        renderTrainingHistory(data.history);
      }
    } catch (e) {
      console.warn('[Training History] Error loading:', e);
    }
  }

  async function addCustomTrainingRuleDirect(ruleText) {
    if (!ruleText || !ruleText.trim()) return;
    try {
      showToast('Customizing rule in instructions.json...', false);
      const res = await fetch('/api/instructions/rules', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rule: ruleText.trim(),
          instruction_id: activeInstructionId,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to add rule');
      showToast('✅ Rule customized & saved into instructions.json!');
      await loadInstructions();
      await loadTrainingHistory();

      if (isConnected && data.active_prompt) {
        sendWsEvent({
          type: 'session.update',
          voice: currentVoice,
          model: currentModel,
          audio_mode: audioMode,
          instructions: data.active_prompt,
        });
      }
      if (isConnected && data.entry && data.entry.verbal_ack) {
        sendWsEvent({
          type: 'client.speak_ack',
          text: data.entry.verbal_ack,
        });
      }
    } catch (err) {
      console.error('[Add Rule Error]:', err);
      alert(`Error saving rule: ${err.message}`);
    }
  }

  function renderTrainingHistory(historyItems) {
    // 1. Update count indicators
    if (elements.trainHistoryCount) {
      elements.trainHistoryCount.textContent = historyItems.length;
    }
    if (elements.bridgeRulesCountBadge) {
      elements.bridgeRulesCountBadge.textContent = `${historyItems.length} active rules in instructions.json`;
    }
    if (elements.stageRulesCount) {
      elements.stageRulesCount.textContent = historyItems.length;
    }

    // 2. Render Modal List
    if (elements.trainHistoryList) {
      if (historyItems.length === 0) {
        elements.trainHistoryList.innerHTML = `
          <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
            No voice training corrections recorded yet. Click "Speak Correction" or add a rule above to train the AI!
          </div>
        `;
      } else {
        elements.trainHistoryList.innerHTML = historyItems.map((item) => {
          const summary = item.summary_en || 'Applied voice training rule';
          const rule = item.rule_diff || summary;
          const feedback = item.feedback || summary;
          return `
          <div class="train-history-card" id="train-card-${item.id}" style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1rem; margin-bottom: 0.85rem;">
            <div class="train-history-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
              <strong style="color: #c084fc; font-size: 0.82rem;">${escapeHtml(item.instruction_name || 'Instruction')}</strong>
              <span class="train-history-time" style="font-size: 0.75rem; color: var(--text-muted);">${escapeHtml(item.timestamp || '')}</span>
            </div>
            <div class="train-history-feedback" style="font-size: 0.83rem; color: #e2e8f0; margin-bottom: 0.4rem;">
              <span style="color: #94a3b8;">🗣️ Spoken / Input:</span> "${escapeHtml(feedback)}"
            </div>
            <div class="train-history-diff" id="rule-display-${item.id}" style="font-size: 0.84rem; color: #38bdf8; background: rgba(56, 189, 248, 0.07); padding: 0.5rem 0.75rem; border-radius: 8px; border-left: 3px solid #38bdf8;">
              ⚡ <strong style="color: #7dd3fc;">Active Rule:</strong> <span class="rule-text-val">${escapeHtml(rule)}</span>
            </div>

            <!-- Inline Edit Form -->
            <div class="train-rule-edit-form" id="rule-edit-form-${item.id}" style="display: none; margin-top: 0.6rem;">
              <label style="font-size: 0.75rem; color: #c084fc; font-weight: 600; display: block; margin-bottom: 0.25rem;">Customize Rule in instructions.json:</label>
              <textarea class="setting-textarea" id="rule-edit-input-${item.id}" rows="2" style="font-size: 0.85rem; width: 100%; box-sizing: border-box; background: rgba(0,0,0,0.4); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 8px; color: #fff; padding: 0.5rem;">${escapeHtml(rule)}</textarea>
              <div style="display: flex; gap: 0.5rem; justify-content: flex-end; margin-top: 0.4rem;">
                <button type="button" class="btn-pill-action cancel btn-cancel-edit-rule" data-id="${item.id}" style="padding: 0.3rem 0.75rem; font-size: 0.78rem;">Cancel</button>
                <button type="button" class="btn-pill-action apply btn-save-edit-rule" data-id="${item.id}" style="padding: 0.3rem 0.85rem; font-size: 0.78rem;">Save to Instructions 💾</button>
              </div>
            </div>

            <!-- Action Bar -->
            <div class="train-card-actions" id="card-actions-${item.id}" style="display: flex; gap: 0.5rem; justify-content: flex-end; margin-top: 0.6rem;">
              <button type="button" class="btn-pill-action btn-edit-rule-trigger" data-id="${item.id}" title="Edit and customize this rule in instructions.json" style="background: rgba(168, 85, 247, 0.15); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.3); padding: 0.3rem 0.75rem; font-size: 0.78rem; border-radius: 6px; cursor: pointer;">
                ✏️ Modify Rule
              </button>
              <button type="button" class="btn-pill-action cancel btn-delete-rule-trigger" data-id="${item.id}" title="Delete this rule from instructions.json" style="background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); padding: 0.3rem 0.75rem; font-size: 0.78rem; border-radius: 6px; cursor: pointer;">
                🗑️ Delete
              </button>
            </div>
          </div>
        `;
        }).join('');
      }
    }

    // 3. Render On-Screen Bridge Rules List
    if (elements.bridgeRulesList) {
      if (historyItems.length === 0) {
        elements.bridgeRulesList.innerHTML = `
          <div style="font-size: 0.78rem; color: #94a3b8; padding: 0.5rem 0; font-style: italic;">
            No custom rules active in instructions.json yet. Tell a correction or click a chip below!
          </div>
        `;
      } else {
        elements.bridgeRulesList.innerHTML = historyItems.map((item) => {
          const rule = item.rule_diff || item.summary_en || 'Rule';
          return `
          <div class="rule-item-card" id="bridge-card-${item.id}">
            <div class="rule-item-main">
              <div class="rule-item-text">
                <span style="color: #a855f7; font-weight: 700; font-size: 0.76rem;">⚡ RULE:</span>
                <span id="bridge-rule-text-${item.id}" style="color: #f1f5f9; font-size: 0.82rem; margin-left: 0.25rem;">${escapeHtml(rule)}</span>
                <span style="font-size: 0.7rem; color: #64748b; margin-left: 0.5rem;">(${escapeHtml(item.timestamp || '')})</span>
              </div>
              <div class="rule-item-actions" id="bridge-actions-${item.id}">
                <button type="button" class="btn-rule-action modify btn-edit-rule-trigger" data-id="${item.id}" title="Modify this rule in instructions.json">
                  ✏️ Modify
                </button>
                <button type="button" class="btn-rule-action delete btn-delete-rule-trigger" data-id="${item.id}" title="Delete from instructions.json">
                  🗑️ Delete
                </button>
              </div>
            </div>

            <!-- Inline Editor in Bridge -->
            <div class="rule-inline-editor" id="bridge-edit-form-${item.id}" style="display: none;">
              <textarea class="rule-edit-textarea" id="bridge-edit-input-${item.id}" rows="2">${escapeHtml(rule)}</textarea>
              <div style="display: flex; gap: 0.4rem; justify-content: flex-end; margin-top: 0.35rem;">
                <button type="button" class="btn-rule-action btn-cancel-edit-rule" data-id="${item.id}" style="background: rgba(255,255,255,0.1); color: #cbd5e1;">Cancel</button>
                <button type="button" class="btn-rule-action modify btn-save-edit-rule" data-id="${item.id}" style="background: #10b981; color: #fff; font-weight: 600;">Save to Instructions 💾</button>
              </div>
            </div>
          </div>
        `;
        }).join('');
      }
    }

    // 4. Render Stage Rules Drawer
    if (elements.stageRulesList) {
      if (historyItems.length === 0) {
        elements.stageRulesList.innerHTML = `
          <div style="font-size: 0.75rem; color: #94a3b8; padding: 0.3rem 0;">
            No custom rules active. Click "+ Add Rule" to customize instructions.json.
          </div>
        `;
      } else {
        elements.stageRulesList.innerHTML = historyItems.map((item) => {
          const rule = item.rule_diff || item.summary_en || 'Rule';
          return `
          <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); padding: 0.35rem 0.6rem; border-radius: 6px;">
            <span style="font-size: 0.78rem; color: #e2e8f0; flex: 1;">⚡ ${escapeHtml(rule)}</span>
            <div style="display: flex; gap: 0.3rem;">
              <button type="button" class="btn-rule-action modify btn-edit-rule-trigger" data-id="${item.id}" style="padding: 0.15rem 0.45rem; font-size: 0.7rem;">✏️ Edit</button>
              <button type="button" class="btn-rule-action delete btn-delete-rule-trigger" data-id="${item.id}" style="padding: 0.15rem 0.45rem; font-size: 0.7rem;">🗑️</button>
            </div>
          </div>
        `;
        }).join('');
      }
    }

    // 5. Attach Universal Action Listeners (Works across Modal, Bridge, and Stage Card!)
    document.querySelectorAll('.btn-edit-rule-trigger').forEach((btn) => {
      btn.addEventListener('click', () => {
        const id = btn.getAttribute('data-id');
        // Toggle forms in modal
        const mDisplay = document.getElementById(`rule-display-${id}`);
        const mForm = document.getElementById(`rule-edit-form-${id}`);
        const mActions = document.getElementById(`card-actions-${id}`);
        if (mDisplay) mDisplay.style.display = 'none';
        if (mActions) mActions.style.display = 'none';
        if (mForm) {
          mForm.style.display = 'block';
          const input = document.getElementById(`rule-edit-input-${id}`);
          if (input) input.focus();
        }

        // Toggle forms in bridge
        const bForm = document.getElementById(`bridge-edit-form-${id}`);
        const bActions = document.getElementById(`bridge-actions-${id}`);
        if (bActions) bActions.style.display = 'none';
        if (bForm) {
          bForm.style.display = 'block';
          const input = document.getElementById(`bridge-edit-input-${id}`);
          if (input) input.focus();
        }
      });
    });

    document.querySelectorAll('.btn-cancel-edit-rule').forEach((btn) => {
      btn.addEventListener('click', () => {
        const id = btn.getAttribute('data-id');
        // In modal
        const mDisplay = document.getElementById(`rule-display-${id}`);
        const mForm = document.getElementById(`rule-edit-form-${id}`);
        const mActions = document.getElementById(`card-actions-${id}`);
        if (mForm) mForm.style.display = 'none';
        if (mDisplay) mDisplay.style.display = 'block';
        if (mActions) mActions.style.display = 'flex';

        // In bridge
        const bForm = document.getElementById(`bridge-edit-form-${id}`);
        const bActions = document.getElementById(`bridge-actions-${id}`);
        if (bForm) bForm.style.display = 'none';
        if (bActions) bActions.style.display = 'flex';
      });
    });

    document.querySelectorAll('.btn-save-edit-rule').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const id = btn.getAttribute('data-id');
        const mInput = document.getElementById(`rule-edit-input-${id}`);
        const bInput = document.getElementById(`bridge-edit-input-${id}`);
        const newRule = (bInput && bInput.value.trim()) || (mInput && mInput.value.trim()) || '';
        if (!newRule) return;

        btn.disabled = true;
        btn.textContent = 'Saving...';
        try {
          const res = await fetch(`/api/instructions/history/${encodeURIComponent(id)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rule_diff: newRule }),
          });
          const result = await res.json();
          if (!res.ok) throw new Error(result.detail || 'Failed to update rule');
          showToast('✅ Rule modified in instructions.json and active on AI!');
          await loadInstructions();
          await loadTrainingHistory();
          if (isConnected && result.active_prompt) {
            sendWsEvent({
              type: 'session.update',
              voice: currentVoice,
              model: currentModel,
              audio_mode: audioMode,
              instructions: result.active_prompt,
            });
          }
        } catch (err) {
          alert(`Error modifying rule: ${err.message}`);
          btn.disabled = false;
          btn.textContent = 'Save to Instructions 💾';
        }
      });
    });

    document.querySelectorAll('.btn-delete-rule-trigger').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const id = btn.getAttribute('data-id');
        if (!confirm('Are you sure you want to delete this rule from instructions.json?')) return;
        try {
          const res = await fetch(`/api/instructions/history/${encodeURIComponent(id)}`, {
            method: 'DELETE',
          });
          const result = await res.json();
          if (!res.ok) throw new Error(result.detail || 'Failed to delete rule');
          showToast('🗑️ Rule removed from instructions.json');
          await loadInstructions();
          await loadTrainingHistory();
          if (isConnected && result.active_prompt) {
            sendWsEvent({
              type: 'session.update',
              voice: currentVoice,
              model: currentModel,
              audio_mode: audioMode,
              instructions: result.active_prompt,
            });
          }
        } catch (err) {
          alert(`Error deleting rule: ${err.message}`);
        }
      });
    });
  }

  function openTrainingHistoryModal() {
    if (elements.trainHistoryModal && elements.trainHistoryBackdrop) {
      elements.trainHistoryBackdrop.style.display = 'block';
      elements.trainHistoryModal.style.display = 'block';
      loadTrainingHistory();
    }
  }

  function closeTrainingHistoryModal() {
    if (elements.trainHistoryModal && elements.trainHistoryBackdrop) {
      elements.trainHistoryBackdrop.style.display = 'none';
      elements.trainHistoryModal.style.display = 'none';
    }
  }

  // ── Response-level Quick Voice Correction Modal ──────────────────────

  function openResponseCorrectionModal(responseText) {
    activeCorrectionTarget.responseText = responseText || '';
    activeCorrectionTarget.context = conversationHistory.slice(-4).map((m) => `${m.role}: ${m.text}`).join('\n');

    if (elements.modalTargetAiResponse) {
      elements.modalTargetAiResponse.textContent = responseText ? `"${responseText}"` : '(Current AI response)';
    }
    if (elements.modalCorrectionInput) {
      elements.modalCorrectionInput.value = '';
    }
    if (elements.modalMicText) {
      elements.modalMicText.textContent = '🎙️ Click to Speak';
    }
    if (elements.btnModalMic) {
      elements.btnModalMic.classList.remove('recording');
    }

    if (elements.responseCorrectModal && elements.responseCorrectBackdrop) {
      elements.responseCorrectBackdrop.style.display = 'block';
      elements.responseCorrectModal.style.display = 'block';
    }
  }

  function closeResponseCorrectionModal() {
    if (isModalRecording) {
      stopModalVoiceRecording();
    }
    if (elements.responseCorrectModal && elements.responseCorrectBackdrop) {
      elements.responseCorrectBackdrop.style.display = 'none';
      elements.responseCorrectModal.style.display = 'none';
    }
  }

  async function toggleModalVoiceRecording() {
    if (!isModalRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        modalAudioChunks = [];
        const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
        modalMediaRecorder = new MediaRecorder(stream, { mimeType });

        modalMediaRecorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) modalAudioChunks.push(e.data);
        };

        modalMediaRecorder.onstop = async () => {
          const audioBlob = new Blob(modalAudioChunks, { type: mimeType });
          stream.getTracks().forEach((t) => t.stop());
          await transcribeModalAudio(audioBlob);
        };

        modalMediaRecorder.start(250);
        isModalRecording = true;
        if (elements.btnModalMic) elements.btnModalMic.classList.add('recording');
        if (elements.modalMicText) elements.modalMicText.textContent = '⏹️ Stop Recording';
        if (elements.modalMicHint) elements.modalMicHint.textContent = 'Listening... Speak your correction now!';
      } catch (err) {
        showToast('Microphone permission required.', true);
      }
    } else {
      stopModalVoiceRecording();
    }
  }

  function stopModalVoiceRecording() {
    if (modalMediaRecorder && modalMediaRecorder.state !== 'inactive') {
      modalMediaRecorder.stop();
    }
    isModalRecording = false;
    if (elements.btnModalMic) elements.btnModalMic.classList.remove('recording');
    if (elements.modalMicText) elements.modalMicText.textContent = '🎙️ Click to Speak';
    if (elements.modalMicHint) elements.modalMicHint.textContent = 'Transcribing with Whisper...';
  }

  async function transcribeModalAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'modal_correction.webm');

    try {
      showToast('Transcribing spoken correction...');
      const res = await fetch('/api/instructions/voice-transcribe', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok && data.status === 'ok') {
        const text = data.transcript || '';
        if (elements.modalCorrectionInput) {
          elements.modalCorrectionInput.value = text;
        }
        if (elements.modalMicHint) {
          elements.modalMicHint.textContent = 'Transcription ready. Click Apply to update rule.';
        }
      }
    } catch (e) {
      if (elements.modalMicHint) {
        elements.modalMicHint.textContent = 'Transcription failed. Please type manually.';
      }
    }
  }

  function showToast(message, isError = false) {
    if (!elements.appToast) return;
    elements.appToast.textContent = message;
    elements.appToast.style.borderColor = isError ? 'var(--accent-rose)' : 'var(--accent-cyan)';
    elements.appToast.style.boxShadow = isError
      ? '0 10px 30px rgba(0, 0, 0, 0.5), 0 0 20px rgba(244, 63, 94, 0.3)'
      : '0 10px 30px rgba(0, 0, 0, 0.5), 0 0 20px rgba(6, 182, 212, 0.25)';
    elements.appToast.classList.add('show');

    setTimeout(() => {
      elements.appToast.classList.remove('show');
    }, 3200);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
})();
