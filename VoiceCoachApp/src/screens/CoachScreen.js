import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Modal,
  Animated,
  Easing,
  Pressable,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import * as Speech from 'expo-speech';
import * as Haptics from 'expo-haptics';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  ExpoSpeechRecognitionModule,
  useSpeechRecognitionEvent,
} from 'expo-speech-recognition';
import {
  API_BASE_URL,
  COLORS,
  PHASE_COLORS,
  WEEKS,
  STORAGE_KEYS,
} from '../constants';

// Orb visual states: idle | listening | thinking | speaking
export default function CoachScreen() {
  const insets = useSafeAreaInsets();

  const [weekIndex, setWeekIndex] = useState(0);
  const [muted, setMuted] = useState(false);
  const [orbState, setOrbState] = useState('idle');
  const [orbStatus, setOrbStatus] = useState('Tap to begin');
  const [coachReply, setCoachReply] = useState(
    'Welcome. Tap the orb to start your session.'
  );
  const [transcript, setTranscript] = useState('…');
  const [exchanges, setExchanges] = useState(0);
  const [weekModal, setWeekModal] = useState(false);
  const [started, setStarted] = useState(false);
  const [busy, setBusy] = useState(false);

  const messagesRef = useRef([]);
  const listeningRef = useRef(false);
  const finalTranscriptRef = useRef('');

  const week = WEEKS[weekIndex];
  const accent = PHASE_COLORS[week.phase];

  // ---- Animations ----
  const pulse = useRef(new Animated.Value(0)).current;
  const spin = useRef(new Animated.Value(0)).current;
  const animRef = useRef(null);

  useEffect(() => {
    if (animRef.current) animRef.current.stop();
    pulse.setValue(0);
    spin.setValue(0);

    if (orbState === 'listening') {
      animRef.current = Animated.loop(
        Animated.sequence([
          Animated.timing(pulse, { toValue: 1, duration: 600, useNativeDriver: true }),
          Animated.timing(pulse, { toValue: 0, duration: 600, useNativeDriver: true }),
        ])
      );
      animRef.current.start();
    } else if (orbState === 'thinking') {
      animRef.current = Animated.loop(
        Animated.timing(spin, {
          toValue: 1,
          duration: 1300,
          easing: Easing.linear,
          useNativeDriver: true,
        })
      );
      animRef.current.start();
    } else if (orbState === 'speaking' || orbState === 'idle') {
      animRef.current = Animated.loop(
        Animated.sequence([
          Animated.timing(pulse, { toValue: 1, duration: 1400, useNativeDriver: true }),
          Animated.timing(pulse, { toValue: 0, duration: 1400, useNativeDriver: true }),
        ])
      );
      animRef.current.start();
    }
    return () => animRef.current && animRef.current.stop();
  }, [orbState, pulse, spin]);

  const orbScale = pulse.interpolate({
    inputRange: [0, 1],
    outputRange: orbState === 'speaking' ? [1, 1.08] : [1, orbState === 'listening' ? 1.12 : 1.03],
  });
  const spinDeg = spin.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });

  // ---- Load persisted state ----
  useEffect(() => {
    (async () => {
      try {
        const w = await AsyncStorage.getItem(STORAGE_KEYS.week);
        const m = await AsyncStorage.getItem(STORAGE_KEYS.mute);
        if (w != null) setWeekIndex(Math.max(0, Math.min(WEEKS.length - 1, parseInt(w, 10) || 0)));
        if (m === '1') setMuted(true);
      } catch (e) {}
    })();
  }, []);

  // ---- TTS ----
  const speak = useCallback(
    (text) => {
      if (muted || !text) {
        setOrbState('idle');
        setOrbStatus('Tap to respond');
        return;
      }
      setOrbState('speaking');
      setOrbStatus('Speaking');
      Speech.stop();
      Speech.speak(text, {
        rate: 1.0,
        pitch: 1.0,
        onDone: () => {
          setOrbState('idle');
          setOrbStatus('Tap to respond');
        },
        onStopped: () => setOrbState('idle'),
        onError: () => {
          setOrbState('idle');
          setOrbStatus('Tap to respond');
        },
      });
    },
    [muted]
  );

  // ---- Speech recognition events ----
  useSpeechRecognitionEvent('result', (event) => {
    const t = event.results?.[0]?.transcript ?? '';
    if (t) setTranscript(t);
    if (event.isFinal) finalTranscriptRef.current = t;
  });
  useSpeechRecognitionEvent('end', () => {
    listeningRef.current = false;
    const said = (finalTranscriptRef.current || '').trim();
    if (said) {
      sendMessage(said);
    } else {
      setOrbState('idle');
      setOrbStatus('Tap to respond');
    }
  });
  useSpeechRecognitionEvent('error', () => {
    listeningRef.current = false;
    setOrbState('idle');
    setOrbStatus('Tap to respond');
  });

  // ---- Networking ----
  const startSession = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setStarted(true);
    messagesRef.current = [];
    setExchanges(0);
    setOrbState('thinking');
    setOrbStatus('Connecting');
    setCoachReply('…');
    try {
      const r = await fetch(`${API_BASE_URL}/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ weekIndex }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Failed to start session');
      setCoachReply(data.reply);
      messagesRef.current.push({ role: 'assistant', content: data.reply });
      speak(data.reply);
    } catch (e) {
      setCoachReply('Could not reach the coach. Check API_BASE_URL and your connection.');
      setOrbState('idle');
      setOrbStatus('Tap to retry');
    } finally {
      setBusy(false);
    }
  }, [busy, weekIndex, speak]);

  const sendMessage = useCallback(
    async (userText) => {
      if (!userText || busy) return;
      setBusy(true);
      setTranscript(userText);
      messagesRef.current.push({ role: 'user', content: userText });
      setOrbState('thinking');
      setOrbStatus('Thinking');
      setCoachReply('…');
      try {
        const r = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ weekIndex, messages: messagesRef.current }),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.error || 'Chat failed');
        setCoachReply(data.reply);
        messagesRef.current.push({ role: 'assistant', content: data.reply });
        setExchanges((n) => n + 1);
        speak(data.reply);
      } catch (e) {
        setCoachReply('Something went wrong reaching the coach. Try again.');
        setOrbState('idle');
        setOrbStatus('Tap to retry');
      } finally {
        setBusy(false);
      }
    },
    [busy, weekIndex, speak]
  );

  // ---- STT control ----
  const startListening = useCallback(async () => {
    if (listeningRef.current) return;
    try {
      const perm = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
      if (!perm.granted) {
        setOrbStatus('Microphone permission needed');
        return;
      }
      finalTranscriptRef.current = '';
      setTranscript('…');
      listeningRef.current = true;
      setOrbState('listening');
      setOrbStatus('Listening');
      ExpoSpeechRecognitionModule.start({
        lang: 'en-US',
        interimResults: true,
        continuous: false,
      });
    } catch (e) {
      listeningRef.current = false;
      setOrbState('idle');
      setOrbStatus('Tap to respond');
    }
  }, []);

  // ---- Orb tap ----
  const onOrbPress = useCallback(() => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    if (busy) return;
    if (!started) {
      startSession();
      return;
    }
    if (listeningRef.current) {
      ExpoSpeechRecognitionModule.stop();
    } else {
      Speech.stop();
      startListening();
    }
  }, [busy, started, startSession, startListening]);

  // ---- Week selection ----
  const chooseWeek = useCallback(
    (i) => {
      Haptics.selectionAsync();
      setWeekIndex(i);
      AsyncStorage.setItem(STORAGE_KEYS.week, String(i));
      setStarted(false);
      messagesRef.current = [];
      setExchanges(0);
      setCoachReply('Week set. Tap the orb to begin this session.');
      setTranscript('…');
      setOrbState('idle');
      setOrbStatus('Tap to begin');
      setWeekModal(false);
    },
    []
  );

  // ---- Mute ----
  const toggleMute = useCallback(() => {
    Haptics.selectionAsync();
    setMuted((m) => {
      const next = !m;
      AsyncStorage.setItem(STORAGE_KEYS.mute, next ? '1' : '0');
      if (next) Speech.stop();
      return next;
    });
  }, []);

  const restart = useCallback(() => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    Speech.stop();
    startSession();
  }, [startSession]);

  const nextWeek = useCallback(() => {
    if (weekIndex < WEEKS.length - 1) chooseWeek(weekIndex + 1);
  }, [weekIndex, chooseWeek]);

  const orbIcon = { idle: '🎙', listening: '👂', thinking: '✦', speaking: '🔊' }[orbState];

  return (
    <View style={[styles.root, { paddingTop: insets.top + 8 }]}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.top}>
          <Text style={styles.brand}>STRATEGIC VOICE COACH</Text>
          <View style={styles.topBtns}>
            <TouchableOpacity
              style={[styles.iconBtn, !muted && { borderColor: accent }]}
              onPress={toggleMute}
            >
              <Text style={styles.iconTxt}>{muted ? '🔇' : '🔊'}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.iconBtn} onPress={restart}>
              <Text style={styles.iconTxt}>↻</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Week card */}
        <TouchableOpacity
          activeOpacity={0.85}
          style={[styles.weekCard, { borderLeftColor: accent }]}
          onPress={() => setWeekModal(true)}
        >
          <Text style={[styles.weekNum, { color: accent }]}>{week.week}</Text>
          <Text style={[styles.phaseTag, { color: accent }]}>{week.phase.toUpperCase()}</Text>
          <Text style={styles.weekTitle}>{week.title}</Text>
          <Text style={styles.weekTag}>{week.tag}</Text>
        </TouchableOpacity>

        {/* Orb */}
        <View style={styles.orbZone}>
          <Pressable onPress={onOrbPress}>
            <Animated.View
              style={[
                styles.orb,
                {
                  backgroundColor: accent,
                  shadowColor: accent,
                  transform: [
                    { scale: orbScale },
                    ...(orbState === 'thinking' ? [{ rotate: spinDeg }] : []),
                  ],
                },
              ]}
            >
              <View style={styles.orbCore}>
                <Text style={styles.orbIcon}>{orbIcon}</Text>
              </View>
            </Animated.View>
          </Pressable>
          <Text style={styles.orbStatus}>{orbStatus}</Text>
        </View>

        {/* Coach panel */}
        <View style={styles.panel}>
          <Text style={styles.panelLabel}>COACH</Text>
          <Text style={styles.coachText}>{coachReply}</Text>
        </View>

        {/* Transcript panel */}
        <View style={styles.panel}>
          <Text style={styles.panelLabel}>YOU</Text>
          <Text style={styles.transcriptText}>{transcript}</Text>
        </View>

        {/* Stats */}
        <View style={styles.stats}>
          <Stat num={exchanges} label="EXCHANGES" accent={accent} />
          <Stat num={week.week} label="WEEK" accent={accent} />
          <Stat num={week.phase.slice(0, 6)} label="PHASE" accent={accent} small />
        </View>

        {/* Next week */}
        <TouchableOpacity style={styles.nextBtn} onPress={nextWeek} disabled={weekIndex >= WEEKS.length - 1}>
          <Text style={[styles.nextTxt, weekIndex >= WEEKS.length - 1 && { opacity: 0.3 }]}>
            Next Week →
          </Text>
        </TouchableOpacity>
      </ScrollView>

      {/* Week picker modal */}
      <Modal visible={weekModal} animationType="slide" transparent onRequestClose={() => setWeekModal(false)}>
        <Pressable style={styles.modalBg} onPress={() => setWeekModal(false)}>
          <Pressable style={styles.modal} onPress={(e) => e.stopPropagation()}>
            <View style={styles.modalHead}>
              <Text style={styles.modalTitle}>Choose a Week</Text>
              <TouchableOpacity style={styles.iconBtn} onPress={() => setWeekModal(false)}>
                <Text style={styles.iconTxt}>✕</Text>
              </TouchableOpacity>
            </View>
            <ScrollView style={{ maxHeight: 460 }}>
              {WEEKS.map((w, i) => (
                <TouchableOpacity
                  key={w.week}
                  style={[styles.weekItem, i === weekIndex && styles.weekItemCurrent]}
                  onPress={() => chooseWeek(i)}
                >
                  <Text style={[styles.wiNum, { color: PHASE_COLORS[w.phase] }]}>{w.week}</Text>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.wiTitle}>{w.title}</Text>
                    <Text style={[styles.wiPhase, { color: PHASE_COLORS[w.phase] }]}>
                      {w.phase.toUpperCase()}
                    </Text>
                  </View>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
}

function Stat({ num, label, accent, small }) {
  return (
    <View style={styles.stat}>
      <Text style={[styles.statNum, { color: accent }, small && { fontSize: 14 }]}>{num}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.bg },
  scroll: { paddingHorizontal: 20, paddingBottom: 36 },
  top: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 },
  brand: { color: COLORS.inkDim, fontSize: 11, letterSpacing: 2, fontWeight: '500' },
  topBtns: { flexDirection: 'row', gap: 8 },
  iconBtn: {
    width: 38, height: 38, borderRadius: 11, borderWidth: 1, borderColor: COLORS.panelBorder,
    backgroundColor: COLORS.panel, alignItems: 'center', justifyContent: 'center',
  },
  iconTxt: { color: COLORS.ink, fontSize: 16 },

  weekCard: {
    backgroundColor: COLORS.panel, borderWidth: 1, borderColor: COLORS.panelBorder,
    borderLeftWidth: 4, borderRadius: 18, padding: 18, marginBottom: 22, overflow: 'hidden',
  },
  weekNum: { position: 'absolute', right: 16, top: 6, fontSize: 56, fontWeight: '800', opacity: 0.15 },
  phaseTag: { fontSize: 10, letterSpacing: 2, fontWeight: '600' },
  weekTitle: { color: COLORS.ink, fontSize: 26, fontWeight: '700', marginVertical: 5 },
  weekTag: { color: COLORS.inkDim, fontSize: 13, lineHeight: 18 },

  orbZone: { alignItems: 'center', marginVertical: 10, marginBottom: 24 },
  orb: {
    width: 168, height: 168, borderRadius: 84, alignItems: 'center', justifyContent: 'center',
    shadowOpacity: 0.6, shadowRadius: 30, shadowOffset: { width: 0, height: 10 }, elevation: 12,
  },
  orbCore: {
    width: 54, height: 54, borderRadius: 27, backgroundColor: 'rgba(7,7,13,0.55)',
    alignItems: 'center', justifyContent: 'center',
  },
  orbIcon: { fontSize: 24 },
  orbStatus: { color: COLORS.inkDim, fontSize: 11, letterSpacing: 1.5, marginTop: 14, fontWeight: '500' },

  panel: {
    backgroundColor: COLORS.panel, borderWidth: 1, borderColor: COLORS.panelBorder,
    borderRadius: 16, padding: 16, marginBottom: 14,
  },
  panelLabel: { color: COLORS.inkDim, fontSize: 10, letterSpacing: 2, fontWeight: '600', marginBottom: 8 },
  coachText: { color: COLORS.ink, fontSize: 17, lineHeight: 25 },
  transcriptText: { color: COLORS.inkDim, fontSize: 14, lineHeight: 21, fontStyle: 'italic' },

  stats: { flexDirection: 'row', gap: 10, marginTop: 4 },
  stat: {
    flex: 1, backgroundColor: COLORS.panel, borderWidth: 1, borderColor: COLORS.panelBorder,
    borderRadius: 13, paddingVertical: 12, alignItems: 'center',
  },
  statNum: { fontSize: 22, fontWeight: '800' },
  statLabel: { color: COLORS.inkDim, fontSize: 9, letterSpacing: 1, marginTop: 3, fontWeight: '500' },

  nextBtn: { alignItems: 'center', marginTop: 20 },
  nextTxt: { color: COLORS.accent, fontSize: 14, letterSpacing: 1, fontWeight: '500' },

  modalBg: { flex: 1, backgroundColor: 'rgba(3,3,8,0.78)', justifyContent: 'flex-end' },
  modal: {
    backgroundColor: '#0e0e16', borderTopLeftRadius: 22, borderTopRightRadius: 22,
    borderWidth: 1, borderColor: COLORS.panelBorder, padding: 20, paddingBottom: 34,
  },
  modalHead: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 },
  modalTitle: { color: COLORS.ink, fontSize: 22, fontWeight: '700' },
  weekItem: {
    flexDirection: 'row', alignItems: 'center', gap: 14, padding: 12, borderRadius: 13,
    borderWidth: 1, borderColor: 'transparent',
  },
  weekItemCurrent: { backgroundColor: COLORS.panel, borderColor: COLORS.panelBorder },
  wiNum: { fontSize: 22, fontWeight: '800', width: 30, textAlign: 'center' },
  wiTitle: { color: COLORS.ink, fontSize: 15, fontWeight: '500' },
  wiPhase: { fontSize: 10, letterSpacing: 1, fontWeight: '600', marginTop: 2 },
});
