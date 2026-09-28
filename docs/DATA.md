# Data

Neither dataset is redistributed here. `asrshift download` fetches both at pinned revisions.

| | Eka Medical ASR Evaluation Dataset | PriMock57 |
|---|---|---|
| Role | source (calibration) | target (shift) |
| Source | [ekacare/eka-medical-asr-evaluation-dataset](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset) (English config) | [babylonhealth/primock57](https://github.com/babylonhealth/primock57) |
| Revision | `433e66037c8aebb8f396b938624a5ab6afc4ca36` | `cd2ac707ad03cb4d2531f4ec6b90c659bf4357c5` |
| Licence | MIT | CC BY 4.0 |
| Content | 3,619 English clips, 8.4 h, 16 kHz MP3 in parquet; reference text; annotated medical entities (drugs, advice, diagnostics, clinical findings) | 57 mock GP consultations, 8.6 h; doctor and patient on separate 16 kHz WAV channels; utterance-level TextGrid transcripts; clinician notes |
| Speech | 2,206 isolated entity narrations (median 3.7 s), 1,303 narrated sentences (median 15 s), 110 conversation clips; Indian-English speakers, many Indian drug brand names | spontaneous role-played consultations (UK clinicians, staff acting as patients), verbatim transcripts |

## Units

**Eka:** one clip = one unit.

**PriMock57 turns:** one annotated utterance, cut from its speaker's own channel with 0.1 s of padding
(6,554 units with at least one scorable word).

**PriMock57 windows:** the two channels are summed into one track (peak-limited) and cut into windows
of at most 30 s. A cut may only fall between two annotated utterances that do not overlap in time, and
at most 0.3 s of padding is added on each side without reaching a neighbouring window or pushing the
window past 30 s. 32 blocks of continuously overlapping speech are longer than 30 s and form their own
windows. 1,240 windows, median 26 s, 4.8 speaker turns on average.

## Splits

**Eka**, speaker-disjoint, stratified by recording context (calibration 50% / validation 20% / test 30%
of clips). Session IDs are nearly one per clip and some prompts are read by two speakers, so after the
speaker split, 200 clips whose session or prompt text would cross splits are dropped. The final split is
disjoint in speakers, sessions and prompt texts (`tests/test_data.py` checks this on the real data).
Unit-to-split assignments are in `results/splits/`. Speaker IDs are hashed there because some are
e-mail addresses.

**PriMock57**, by consultation: 10 recalibration consultations (two per recording day, drawn with seed
598 and listed in `configs/default.yaml`) and 47 test consultations. The consultations were run by 7
clinicians, and no clinician ID is released, so recalibration and test share clinicians.

## Transcriber tags (PriMock57)

| Tag | Count | Handling |
|---|---:|---|
| `<UNSURE>…</UNSURE>` | 1,132 | words kept |
| `<UNIN/>` | 1,359 | wildcard: absorbs any hypothesis words at zero cost |
| `<INAUDIBLE_SPEECH/>` | 814 | wildcard, as above |

The number of words each wildcard absorbs is recorded. The sensitivity label `err_c3` charges every
absorbed word beyond three as an error.

## Known quirks

* Eka references occasionally contain typos ("associted") or non-English words. Whisper is charged
  for them, as in any benchmark.
* PriMock57 references are verbatim (restarts, repeated words, fillers) while Whisper writes clean
  text. Fillers are removed by the normaliser, and immediate repetitions of up to four words are
  collapsed on both sides. Long repetition loops are kept, so hallucination loops still count.
* PriMock57 audio was recorded from video calls (Opus). Eka audio is MP3.
