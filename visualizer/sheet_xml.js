// JS port of evaluate_muster.triplets_to_musicxml: convert score notes
// ({t, d, p} in 10ms bins on the 0.5s beat grid) to single-part MusicXML.
// Without a meter argument it mirrors the Python element-for-element (exact
// token grid, 4/4): same greedy voice assignment, same measure splitting with
// ties, same forward/backup shifts — diffable against the Python reference
// output. With a meter block (from scores_xml.js) it instead snaps notes to a
// 1/12-beat grid and typesets them in the piece's REAL time signature and key,
// with barlines on the real downbeat phase — the annotated beat is not always
// a 4/4 quarter note (e.g. 6/8 pieces are annotated at the dotted quarter, so
// a 16th is 1/6 beat = ~8 bins, unrepresentable as any clean 4/4 value).
// Used by visualizer.html's engraved sheet view to typeset the predicted
// rollout with OSMD. Loadable in the browser (window global) and in node
// (module.exports).
(function (root) {
  "use strict";

  function esc(text) {
    return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  const STEP_NAMES = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"];
  const STEP_ALTERS = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0];
  const FLAT_NAMES = ["C", "D", "D", "E", "E", "F", "G", "G", "A", "A", "B", "B"];
  const FLAT_ALTERS = [0, -1, 0, -1, 0, 0, -1, 0, -1, 0, -1, 0];

  // notes: [{t, d, p}] (nulls filtered by caller); beatSeconds matches the grid.
  // meter (optional, from scores_xml.js): {beats, beat_type, beats_per_measure,
  // offset_bins, fifths, measure_number_from} — the window's real notated meter.
  function tripletsToMusicXML(notes, beatSeconds, meter) {
    beatSeconds = beatSeconds || 0.5;
    const binsPerQuarter = Math.round(100 * beatSeconds);
    if (binsPerQuarter <= 0) return null;

    // All positions/durations below are in "units": raw bins on the 4/4
    // exact-grid path, snapped 1/12-beat grid units on the real-meter path
    // (divisions is defined per path so a unit is always 1 division).
    let toOnset, toDur, unitsPerMeasure, divisions, tsBeats, tsBeatType, fifths, numberFrom;
    if (meter) {
      divisions = (3 * meter.beats_per_measure * meter.beat_type) / meter.beats;
      if (!Number.isInteger(divisions) || divisions <= 0) return null;
      const scale = 12 / binsPerQuarter;  // grid beat (50 bins) -> 12 units
      unitsPerMeasure = 12 * meter.beats_per_measure;
      toOnset = t => Math.round((t + meter.offset_bins) * scale);
      toDur = d => Math.max(1, Math.round(d * scale));
      tsBeats = meter.beats;
      tsBeatType = meter.beat_type;
      fifths = meter.fifths || 0;
      numberFrom = meter.measure_number_from || 1;
    } else {
      divisions = binsPerQuarter;
      unitsPerMeasure = binsPerQuarter * 4;
      toOnset = t => Math.trunc(t);
      toDur = d => Math.max(1, Math.trunc(d));
      tsBeats = 4;
      tsBeatType = 4;
      fifths = 0;
      numberFrom = 1;
    }
    const stepNames = fifths < 0 ? FLAT_NAMES : STEP_NAMES;
    const stepAlters = fifths < 0 ? FLAT_ALTERS : STEP_ALTERS;

    // group by (onset, dur) -> chord pitches
    const grouped = new Map();
    for (const n of notes) {
      if (!n || n.p == null || n.p < 0 || n.p > 127) continue;
      const onset = toOnset(n.t);
      const dur = toDur(n.d);
      const key = onset + "_" + dur;
      if (!grouped.has(key)) grouped.set(key, { onset, dur, pitches: [] });
      grouped.get(key).pitches.push(Math.trunc(n.p));
    }
    if (!grouped.size) return null;

    const chordEvents = Array.from(grouped.values());
    chordEvents.forEach(ev => ev.pitches.sort((a, b) => a - b));
    chordEvents.sort((a, b) =>
      a.onset - b.onset || a.dur - b.dur || comparePitchLists(a.pitches, b.pitches));

    // greedy voice assignment: first voice free at the event's onset
    const voices = [];
    const voiceEnd = [];
    for (const ev of chordEvents) {
      let assigned = null;
      for (let v = 0; v < voiceEnd.length; v++) {
        if (voiceEnd[v] <= ev.onset) { assigned = v; break; }
      }
      if (assigned === null) {
        assigned = voices.length;
        voices.push([]);
        voiceEnd.push(0);
      }
      voices[assigned].push(ev);
      voiceEnd[assigned] = ev.onset + ev.dur;
    }

    const totalUnits = Math.max(...chordEvents.map(ev => ev.onset + ev.dur));
    const numMeasures = Math.max(1, Math.ceil(totalUnits / unitsPerMeasure));

    // split events at barlines into tied segments
    const voiceMeasureEvents = voices.map(() =>
      Array.from({ length: numMeasures }, () => []));
    voices.forEach((voiceEvents, v) => {
      for (const ev of voiceEvents) {
        let segOnset = ev.onset;
        let remaining = ev.dur;
        let first = true;
        while (remaining > 0) {
          const mIdx = Math.min(Math.floor(segOnset / unitsPerMeasure), numMeasures - 1);
          const localOnset = segOnset - mIdx * unitsPerMeasure;
          const segDur = Math.min(remaining, unitsPerMeasure - localOnset);
          voiceMeasureEvents[v][mIdx].push({
            onset: localOnset,
            dur: segDur,
            pitches: ev.pitches,
            tieStart: remaining > segDur,
            tieStop: !first,
          });
          segOnset += segDur;
          remaining -= segDur;
          first = false;
        }
      }
    });

    const lines = [];
    lines.push('<?xml version="1.0" encoding="UTF-8"?>');
    lines.push('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">');
    lines.push('<score-partwise version="3.0">');
    lines.push('<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>');
    lines.push('<part id="P1">');

    for (let mIdx = 0; mIdx < numMeasures; mIdx++) {
      lines.push(`<measure number="${numberFrom + mIdx}">`);
      if (mIdx === 0) {
        lines.push("<attributes>");
        lines.push(`<divisions>${divisions}</divisions>`);
        lines.push(`<key><fifths>${fifths}</fifths></key>`);
        lines.push(`<time><beats>${tsBeats}</beats><beat-type>${tsBeatType}</beat-type></time>`);
        lines.push("<clef><sign>G</sign><line>2</line></clef>");
        lines.push("</attributes>");
      }

      const activeVoices = [];
      for (let v = 0; v < voiceMeasureEvents.length; v++) {
        if (voiceMeasureEvents[v][mIdx].length) activeVoices.push(v);
      }
      if (!activeVoices.length) {
        lines.push(`<note><rest/><duration>${unitsPerMeasure}</duration><voice>1</voice><staff>1</staff></note>`);
        lines.push("</measure>");
        continue;
      }

      activeVoices.forEach((v, activeIdx) => {
        let cursor = 0;
        const events = voiceMeasureEvents[v][mIdx].slice().sort((a, b) =>
          a.onset - b.onset || a.dur - b.dur || comparePitchLists(a.pitches, b.pitches));
        for (const ev of events) {
          if (ev.onset > cursor) {
            lines.push(`<forward><duration>${ev.onset - cursor}</duration></forward>`);
            cursor = ev.onset;
          }
          ev.pitches.forEach((pitch, chordIdx) => {
            const pc = ((pitch % 12) + 12) % 12;
            const octave = Math.floor(pitch / 12) - 1;
            let s = "<note>";
            if (chordIdx > 0) s += "<chord/>";
            s += "<pitch>";
            s += `<step>${stepNames[pc]}</step>`;
            if (stepAlters[pc]) s += `<alter>${stepAlters[pc]}</alter>`;
            s += `<octave>${octave}</octave>`;
            s += "</pitch>";
            if (ev.tieStop) s += '<tie type="stop"/>';
            if (ev.tieStart) s += '<tie type="start"/>';
            s += `<duration>${ev.dur}</duration>`;
            s += `<voice>${v + 1}</voice>`;
            s += "<staff>1</staff>";
            if (ev.tieStart || ev.tieStop) {
              s += "<notations>";
              if (ev.tieStop) s += '<tied type="stop"/>';
              if (ev.tieStart) s += '<tied type="start"/>';
              s += "</notations>";
            }
            s += "</note>";
            lines.push(s);
          });
          cursor = Math.max(cursor, ev.onset + ev.dur);
        }
        if (activeIdx < activeVoices.length - 1) {
          lines.push(`<backup><duration>${cursor}</duration></backup>`);
        }
      });
      lines.push("</measure>");
    }

    lines.push("</part>");
    lines.push("</score-partwise>");
    return lines.join("\n");
  }

  function comparePitchLists(a, b) {
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      if (a[i] !== b[i]) return a[i] - b[i];
    }
    return a.length - b.length;
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { tripletsToMusicXML };
  }
  root.tripletsToMusicXML = tripletsToMusicXML;
})(typeof window !== "undefined" ? window : globalThis);
