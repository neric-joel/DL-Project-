# Study protocol

This file fixes every analysis choice that shapes a reported number. It was written before any
test-split label was computed. Choices made afterwards are listed under *Amendments* with the
reason and date, so a reader can tell which results were planned and which were exploratory.

## Question

Does an ASR review policy that is calibrated on short medical speech stay reliable when it is
applied, unchanged, to longer conversations between doctors and patients?

A **review policy** reads confidence signals that Whisper produces anyway, turns them into a risk
score for each transcript, and sends the riskiest transcripts to a human. Everything else is
accepted automatically. Whisper's weights are never changed.

## Data and units

| Role | Data | Unit | Split |
|---|---|---|---|
| Source | Eka Medical ASR Evaluation (English, MIT) | one clip | speaker-disjoint: calibration 50% / validation 20% / test 30% of clips, stratified by recording context |
| Target | PriMock57 (CC BY 4.0) | **window**: both channels mixed, ≤ 30 s (except 32 longer blocks of overlapping speech, Amendment 1.5), cut only between annotated utterances | by consultation: 10 recalibration (2 per day, seed 598) / 47 test |
| Target (control) | PriMock57 | **turn**: one utterance from its speaker's channel | same consultation split |

*Why speaker-disjoint instead of session-disjoint (the proposal's wording):* Eka session IDs are
almost one per clip, while each speaker records dozens of sessions, and some prompt texts are read
by two speakers. We split by speaker and then drop the 200 clips whose session or prompt text would
cross splits. The result is disjoint in speakers, sessions and texts.

*Why two PriMock unit types:* windows carry everything the hypothesis names (speaker turns, pauses,
longer context, overlapping speech). Turns are conversational but single-speaker and short, like
Eka clips. Comparing the two separates "conversational speech" from "long multi-speaker audio".

## ASR

whisper-small through faster-whisper 1.2.1 (CTranslate2 float16 on GPU), with the library's
default decoding settings except `language="en"` and `word_timestamps=True`: beam 5, temperature
fallback 0.0–1.0, compression-ratio threshold 2.4, log-prob threshold −1.0, no-speech threshold
0.6, conditioning on previous text, no VAD. Each unit is decoded independently.

## Scoring

Reference and hypothesis go through the same normalisation: Whisper's `EnglishTextNormalizer`,
one spelling for dosage units (milligram → mg, ...), and collapsing of immediate repetitions of
1–4 words ("it's a bit it's a bit" → "it's a bit"). The last step matters because PriMock57
references are verbatim (restarts, repeated words) while Whisper writes clean text, so without it
the target domain would be charged for disfluencies nobody would correct. It is applied to both
sides, so it cannot hide a real word error.

PriMock57 transcriber tags: words inside `<UNSURE>…</UNSURE>` are kept; `<UNIN/>` (unintelligible)
becomes a wildcard that absorbs any run of hypothesis words at no cost. Units with no scorable
reference words are excluded.

Per unit we record reference length, substitutions, deletions, insertions and WER.

**Clinically critical tokens**, scored on both datasets from the same alignment:
*numbers* (any normalised token containing a digit: doses, durations, frequencies) and
*negation cues* (no, not, never, nothing, none, nobody, neither, nor, without, nil, deny/denies/denied).
A reference token is recognised if the alignment marks it correct.

**Medical entities (Eka):** the dataset's annotated spans (drugs, advice, diagnostics, clinical
findings). An entity is correct if every one of its normalised tokens is aligned as correct.

## Labels

A transcript is **erroneous** (`err = 1`) if its WER is above **0.10**: more than one error per ten
reference words, i.e. it would need correcting before it could go into a record. For a clip of up
to nine words this means any error at all.

Secondary labels, reported as sensitivity analyses: any error (WER > 0), severe error (WER > 0.30),
and for Eka any medical-entity error.

Decision rule fixed in advance: keep 0.10 unless the base rate on Eka calibration or PriMock
recalibration windows falls outside 10–90%, in which case move to the nearest of {0.05, 0.20, 0.30}
that brings both inside; the move is recorded as an amendment. Only calibration and recalibration
labels may be inspected for this.

*Outcome:* base rates of 51.0% (Eka calibration) and 50.5% (PriMock57 recalibration windows), so the
threshold stays at 0.10.

## Confidence features

All computed from the decoder output, identically in both domains:

* `conf` = exp(token-weighted mean segment log-probability); Whisper's own sequence confidence
* lowest segment log-probability
* word probabilities: mean, minimum, 10th percentile, share below 0.5
* largest no-speech probability; largest compression ratio; share of repeated word trigrams
* whether temperature fallback was used
* log number of hypothesis words, log audio duration, hypothesis words per second, number of segments

## Review policies

| # | Policy | Risk score | Fitted on | Thresholds set on |
|---|---|---|---|---|
| P1 | Raw confidence | 1 − `conf` | none | Eka validation |
| P2 | No-speech + repetition rules | max(no-speech / 0.6, compression ratio / 2.4); review if ≥ 1 is Whisper's own rule | none | Eka validation (for budgets) |
| P3 | Eka-calibrated, frozen | logistic regression on all features → P(err) | Eka calibration | Eka validation |
| P4 | Recalibrated on PriMock57 | P3's logit refitted with two parameters (Platt): σ(a·logit + b) | PriMock recalibration (same unit type) | PriMock recalibration |

Variants: P3 with confidence features only (drops length and rate features); P4 with the logistic
regression refitted from scratch on the recalibration set. Reference ceiling: an in-domain model
cross-fitted on the PriMock test set by consultation (not deployable, shows what is achievable).

Regularisation strength for the logistic regressions is chosen by 5-fold cross-validation grouped
by speaker/session (Eka) or consultation (PriMock), minimising log loss.

## Operating points

1. **Review budget** b ∈ {1%, 5%, 10%}: review the units whose risk is above the (1−b) quantile of
   risk on the policy's tuning set. The threshold is then frozen. On the evaluation set we report the
   share actually sent to review, the share of erroneous transcripts caught, the error rate among
   auto-accepted transcripts, and the pooled WER of auto-accepted transcripts.
2. **Risk target** α ∈ {0.10, 0.20}: accept a transcript if its predicted P(err) ≤ τ, where τ is the
   largest threshold whose accepted-error rate on the tuning set is ≤ α. On the evaluation set we
   report the realised accepted-error rate against α.

## Metrics

* ASR: WER, substitution / deletion / insertion rates, number and negation recognition rates,
  entity accuracy and entity error rate by type (Eka).
* Calibration of P(err): expected calibration error (15 equal-width bins), Brier score, mean
  predicted risk minus observed error rate (negative = overconfident), reliability diagrams.
* Raw word confidence: word-level ECE of Whisper's word probabilities against word correctness.
* Ranking: AUROC for err, area under the risk–coverage curve (AURC), risk–coverage curves.
* Share of errors made with high confidence: erroneous transcripts whose raw confidence is above
  the Eka-tuned 10%-budget threshold of P1, i.e. those the raw policy would accept.

Uncertainty: 95% percentile intervals from 2,000 cluster-bootstrap replicates that resample
speakers/sessions (Eka) or consultations (PriMock). Differences between P3 and P4 use a paired
bootstrap over the same consultations. The effect of the recalibration set size is measured by
redrawing k ∈ {2, 5, 10, 20} recalibration consultations 50 times each and scoring on the rest.

## Hypotheses

* **H1** P3 is overconfident on PriMock windows: mean predicted risk below the observed error rate,
  and ECE above its Eka-test value.
* **H2** At the threshold that sends 10% of Eka validation to review, P3 auto-accepts a larger share
  of erroneous PriMock windows than of erroneous Eka test clips.
* **H3** Recalibrating on 10 PriMock consultations lowers ECE on PriMock test windows and brings the
  realised accepted-error rate closer to the risk target.
* **Exploratory** Whether turns behave like Eka or like windows; which features carry the shift.

## Deviations from the proposal

* **Eka split by speaker, not only by session.** See *Data and units*. Stricter than proposed.
* **PriMock57 medical entities.** The proposal made this optional and named a fallback: full entity
  scoring on Eka, transcript-level transfer on PriMock57. We take that fallback for the confirmatory
  analysis. On both datasets, numbers and negations are scored as pre-specified clinically critical
  tokens (Amendment 1). Any PriMock57 term lexicon or NER pass is exploratory.
* **Whisper large-v3-turbo** is not part of the confirmatory analysis (optional in the proposal). If
  it is run, it will be declared here before its labels exist.
* **Operating points beyond the proposal** (20% and 30% budgets, α = 0.30, the plug-in rule of
  Amendment 1) are additions. The 1% budget is kept but is **exploratory**: its threshold rests on
  about 7 Eka validation clips and 2 PriMock recalibration windows. The headline operating point is
  the 10% budget.

## Amendments

### Amendment 1: 2026-09-28, before any PriMock57 unit was decoded or scored

An independent critique of this protocol (three reviewers, each point checked by a skeptic) raised
the issues below. When this amendment was written, no PriMock57 hypothesis or label existed. A
smoke test of the scorer had already produced labels for 1,329 Eka clips across all three splits and
printed aggregate Eka WER and error rates (including test). None of the changes below depend on those
numbers. All are deterministic functions of the planned alignments, and all add analyses rather than
remove any.

1. **The primary label depends on unit length.** `WER > 0.10` means "any error" for a 3-word Eka
   clip but "7 or more errors" for a 66-word window. So the frozen model can look miscalibrated on
   windows with no change in Whisper's word-level confidence, and a single dose error in a window
   (the proposal's own example) leaves the label at 0. Added:
   * a **length-stratified** view: base rate, mean predicted risk, calibration-in-the-large and
     accepted-error rate in reference-length bins ≤9, 10–30, 31–50, >50 words. Pre-registered
     reading: 31–50 is the one bin where Eka (narrated sentences) and PriMock57 overlap. If P3 is
     overconfident inside it, the effect is not a label-length artifact. Results in >50 are
     extrapolation and are reported as such;
   * a **length-matched contrast**: Eka narrated sentences vs PriMock57 turns and windows, all with
     25–41 reference words;
   * **word-level calibration** (length-free) promoted to hypothesis H4.
2. **Critical tokens.** *Numbers*: a token containing a digit (but not an ordinal like "1st", which
   the normaliser makes out of "first"), or one of *once, twice, thrice, half, od, bd, tds, qds, prn*,
   or "one" followed by a unit, dose-form or time word (the normaliser writes "1" as "one").
   *Negations*: unchanged. Both are now scored on **both sides**: reference tokens not recognised
   (missed), and hypothesis tokens that are substituted or inserted (false alarms, such as a
   hallucinated "not" or a wrong dose). New label **`crit_err`** = any missed or false critical token.
   It is reported as a secondary label and, more importantly, as a **residual metric of every policy**:
   the share of auto-accepted transcripts with a critical error, and the share of critical errors
   sent to review. The old digit-only count is kept in the tables for transparency.
3. **What recalibration can and cannot change.** Platt scaling is monotone, so P4 ranks exactly like
   P3, and for the two empirical threshold rules (budget quantiles, empirical risk target) P4's
   decisions equal *P3's score with thresholds re-tuned on PriMock57*. Tables label them that way.
   H3 is split:
   * **H3a** (Platt): recalibration lowers ECE, Brier and |calibration-in-the-large| on PriMock
     test windows.
   * **H3b** (threshold re-tuning): re-tuned thresholds bring the realised accepted-error rate closer
     to the target.
   * **H3c** (decisions that use the probabilities): a **plug-in rule** accepts the largest set of
     transcripts whose *mean predicted* P(err) is ≤ α. It is computed on the unlabelled evaluation
     stream, so it needs no target labels. With frozen P3 its realised accepted-error rate exceeds α
     on PriMock windows; with P4 it is closer to α.
   In the recalibration-size curve, intercept-only recalibration (one parameter) is reported next to
   Platt. If a Platt fit gives a slope ≤ 0 (possible with 2 consultations), it falls back to
   intercept-only, and draws where no threshold meets the target are counted, not dropped.
4. **Feature extrapolation.** P3 uses length and segment-count features. On Eka these are nearly
   constant within a recording context, and PriMock windows lie far outside them. P3-conf
   (confidence features only) is therefore co-reported with P3 for H1 and H2. Pre-stated reading:
   if P3 is overconfident on windows but P3-conf is not, the gap is due to P3's length features
   extrapolating, not to Whisper's confidence.
5. **Windows.** Padding is capped so that a window never exceeds 30 s unless one block of
   continuously overlapping speech is longer than 30 s. That happens for 32 windows (26 test, 6
   recalibration; 30.0–55.1 s). They are kept, and a sensitivity analysis drops them. Within a unit,
   faster-whisper can take more than one 30 s pass, each conditioned on the previous text.
   Window cuts come from the gold utterance times, which act as a perfect voice-activity detector. A
   deployed scribe would cut less cleanly, so target WER is likely optimistic.
6. **Deployment view** (secondary): per PriMock57 test consultation, the minutes of audio sent to
   review, the share of consultations with at least one auto-accepted erroneous window, and
   auto-accepted critical-token errors per consultation. A **label-free drift alarm** fires if the
   frozen 10%-budget threshold sends a share of PriMock57 windows to review that falls outside the
   Eka-test 95% interval, or if mean confidence or mean P3 risk shifts by more than 0.5 Eka standard
   deviations.
7. **Word-level confident errors**: the share of wrong hypothesis words with Whisper probability
   ≥ 0.9, and the same for number and negation tokens. Deleted critical tokens have no confidence and
   are reported separately.
8. **Eka entity label**: `ent_err` is evaluated on Eka clips that contain at least one entity (P1–P3).

**What the contrasts separate.** Eka → PriMock57 windows is a *bundled* deployment shift: speaker
population and accent, read prompts vs spontaneous role-play, recording channel, clean vs verbatim
references, length, and several speakers. H1–H3 are claims about this bundle. Turns vs windows holds
population, channel and reference style fixed and varies only length, the number of speakers and
overlap. Eka vs turns carries everything else.

**Hypothesis H4** (word level): Whisper's word probabilities are more overconfident (higher word ECE,
more wrong words with p ≥ 0.9) on PriMock57 than on Eka.

### Amendment 2: 2026-09-28, still before any PriMock57 unit was scored

The second half of the same critique concerned scoring details and the statistics of the tests.
Inference on PriMock57 was running, but no PriMock57 transcript had been scored or looked at.

1. **Transcriber tags.** PriMock57 also uses `<INAUDIBLE_SPEECH/>` (814 times), which the first draft
   silently dropped, so Whisper would have been charged for any words written there. It is now a
   wildcard like `<UNIN/>`. Any other tag stops the scorer with an error instead of passing silently.
2. **Spelling variants.** "OK"/"O.K." → "okay", "alright" → "all right", dotted acronyms joined
   ("B.P." → "bp"), applied to both sides. PriMock57 references write "OK" and Whisper writes "Okay";
   1,861 reference lines contain it.
3. **Wildcards can hide hallucinations.** A wildcard absorbs any number of hypothesis words at zero
   cost. The number absorbed is now recorded, and a sensitivity label `err_c3` charges every absorbed
   word beyond three per wildcard as an insertion. That is an upper bound on a capped wildcard, and
   exact when nothing is capped.
4. **Repetition loops.** Collapsing repeats could also remove a Whisper repetition loop ("thank you
   thank you thank you …"), a known hallucination. Runs whose collapse would remove more than six
   tokens are now left alone and count as errors.
5. **P2.** The raw compression ratio grows with text length, so ranking by it mostly measures how long
   a transcript is. P2's risk is now max(no-speech / 0.6, compression ratio / 2.4 if it exceeds 2.4,
   share of repeated trigrams / 0.2), with raw confidence breaking ties. It still flags exactly what
   Whisper's own thresholds flag, plus looping text. The version first registered is reported as
   `P2-rawCR`.
6. **Risk-target rule.** Choosing the best-looking cut on a small tuning set is optimistic. A cut must
   now accept at least 30 tuning units. A second, conservative version (`riskcp`) uses fixed-sequence
   testing with a one-sided 95% Clopper–Pearson bound. If no cut qualifies, everything goes to review,
   and the table says so (`feasible = False`). Both rules are applied to every policy.
7. **Decision rules and multiplicity.** The confirmatory family is H1, H2, H3a, H3b, H3c and H4 at
   α_target = 0.20 (for H3b/c), all on PriMock57 test windows with label `err`. Each is tested with a
   one-sided cluster-bootstrap p-value (2,000 replicates), and p-values are Holm-adjusted across the
   family. A hypothesis is *supported* if its adjusted p < 0.05. A statistic that is undefined in more
   than 5% of replicates (for example an infeasible risk target) is reported as *not testable*.
   * H1: calibration-in-the-large of P3 on windows < 0.
   * H2: share of erroneous transcripts that P3 auto-accepts at the Eka 10%-budget threshold is higher
     on windows than on Eka test.
   * H3a: ECE(P4) − ECE(P3) < 0. H3b and H3c: |realised accepted error − 0.20| is smaller for P4 than
     for P3 under the empirical and the plug-in rule respectively.
   * H4: word-level ECE is higher on windows than on Eka test.
   Everything else (turns, other labels, other operating points, sensitivity analyses) is descriptive.
8. **Null baseline for the length confound.** 100 simulations in which every hypothesis word is wrong
   with probability 1 − (its Whisper probability), so word confidence is perfectly calibrated in both
   domains by construction. P3 is refitted on the simulated Eka labels and the H1/H2 statistics are
   recomputed. The observed statistics are read against this distribution. It has no deletions, so it
   is a guide, not an exact null.
9. **Feature support.** Label-free: the share of PriMock57 units outside Eka's 1st–99th percentile for
   each feature, a domain classifier (Eka vs PriMock57 windows; AUROC), and the shift of P3's mean
   logit broken down by feature (coefficient × shift of the standardised mean).
10. **Robustness of the source thresholds.** The 10%-budget thresholds of P1 and P3 are also tuned on
    Eka calibration + validation (out-of-fold P3 scores for calibration clips), because the validation
    split's entity narrations come from one speaker.
11. **Overlapping speech.** Window references order utterances by start time, which can penalise
    Whisper for a different but valid order of overlapping speech. Sensitivity analysis: windows with no
    overlapping speech only.
12. **Shared clinicians.** PriMock57's 57 consultations were run by 7 clinicians, and no clinician ID
    is released. The recalibration and test consultations therefore share clinicians, so P4 measures
    same-clinic recalibration, not transfer to new clinicians. PriMock57 intervals are conditional on
    these clinicians. H1 and H2 (frozen P3) are not affected.
13. **Wording.** "Share of errors made with high confidence" at transcript level is simply
    1 − (errors caught) at the 10% budget and is reported for every policy. The word-level version
    (Amendment 1.7) is the one that answers the proposal's question about confident errors.
14. **Privacy.** Some Eka speaker IDs are e-mail addresses. Only a hash of them is written to any
    output in the repository.

### Amendment 3: 2026-09-28, still before any PriMock57 unit was scored (exploratory analysis)

**Shared medical vocabulary.** The proposal's optional PriMock57 entity step ("a simple lexicon or
entity recognition tool, and check a sample by hand") is implemented as follows. Every medical entity
annotated in Eka is normalised; pure numbers and doses, and single words that are usually not clinical
in conversation ("back", "rest", ...), are removed. The terms that also occur in PriMock57 reference
transcripts form the lexicon (270 terms, 2,468 occurrences). The same greedy longest-match matcher is
run on the references of *both* datasets. A term is correct if all its words are aligned as correct.
This gives term accuracy on an identical vocabulary in both domains, with no brand names. The
lexicon uses only human annotations and reference text, never ASR output or labels. It is committed
(`results/terms/`), together with a random sample of 100 matches for a manual precision check by the
team. A small local LLM (qwen3.5:2b through Ollama) was tried first as the entity tool and rejected,
because on a test chunk it copied the prompt's examples and missed most findings. All term results are
exploratory.

### Amendment 4: 2026-09-28, after inference finished and before any scoring of the full data

An adversarial code review (four reviewers; each defect was reproduced by a second agent before it
was accepted) found the scoring defects below. All were fixed before `asrshift score` was run on the
complete outputs. One verifier examined individual PriMock57 *recalibration* units to confirm the
number-fusion defect. That split is labelled data the protocol lets us use. No test-split result had
been computed.

1. **Numbers fused across boundaries.** Whisper's normaliser merges adjacent number words whatever
   the punctuation or speaker ("twenty six." + "Twenty six, OK." → "2626"; "Dolo 650 three times" →
   "653"). This mostly affected PriMock57 windows and so biased the comparison against the target
   domain. Text is now normalised piece by piece. A piece ends at an utterance boundary (window
   references now keep one utterance per line), at clause punctuation followed by a space, and between
   a digit and a spelled-out number. Both sides are treated the same way.
2. **Eka entities.** Duplicate annotations are counted once. Each entity is matched to the occurrence
   nearest its annotated character offset, and no occurrence is used twice. Entities that cannot be
   located in the normalised reference are no longer scored through a hypothesis-side fallback; they
   are counted separately (`n_ent_unmapped`).
3. **Unknown transcriber tags** that are well formed (e.g. `<LAUGH/>`) now stop the scorer, as
   Amendment 2.1 intended.
4. **"one" as a number** only before a unit, dose-form or time word, as Amendment 1.2 says ("or"/"to"
   removed).
5. **Sensitivity analyses** drop windows from the *test* set only, so the recalibration set, P4 and its
   thresholds stay identical to the primary analysis. The long-window cut-off is 30 s, which removes all
   32 over-length windows. The sensitivity table reports whether each operating point was feasible.
   Added: dropping windows in which a wildcard absorbed more than three words.
6. **Plug-in intervals** re-estimate the plug-in threshold in every bootstrap replicate, because it is
   computed from the evaluation stream itself.

### Amendment 5: 2026-09-28, before any large-v3-turbo output existed

The proposal's optional second model is run once the main pipeline was finished: **whisper-large-v3-turbo**
(`dropbox-dash/faster-whisper-large-v3-turbo`, revision `0a363e91`, float16), configured in
`configs/large-v3-turbo.yaml`. Everything else is identical: units, splits, decoding settings, scoring,
labels, policies (fitted on turbo's own Eka calibration outputs), operating points and tests. Its
results are **exploratory**. The confirmatory family (H1–H4) is whisper-small's.

### Amendment 6: 2026-09-28, *after* the first full results (post hoc)

While choosing an example transcript for the demo, we found that Whisper's normaliser turns the
interjection "Oh" into the digit **0**. PriMock57 transcribers write "Ohh"/"Ooh", so a harmless
interjection counted as a substitution *and* a false number alarm, in 5.2% of PriMock57 windows but 0.4%
of Eka clips. That biased the dose/number results against the target domain. A systematic list of the
most frequent substitutions (both domains) then showed more words written two ways: unit
abbreviations spelled out ("dl"/"deciliter", "l"/"liter", "mm"/"millimeter", "cm", "g"/"grams",
"g/dL"/"g per dL"), "e.g."/"for example", "Ltd"/"limited", "mum"/"mom". Also, the normaliser deletes "mm"
as a filler, so "5 mm" lost its unit.

All were fixed symmetrically. Interjections (oh, ooh, ah, er, erm, eh, huh, uh-huh, mm-hmm) are
dropped unless "oh" means zero ("oh seven", "point oh five"). The unit and spelling pairs are mapped
to one form. Pairs of *different* words ("yeah"/"yes", "the"/"a") were left alone. Everything was then
rescored and re-evaluated. Because the change came after test results existed, the numbers from
before and after it are both given below. The pre-change results are in the git history (commit
`9549ba9`, "Add whisper-small results").

| | before | after |
|---|---:|---:|
| WER, Eka test / PriMock57 windows / turns | 14.1% / 11.8% / 11.6% | 13.7% / 11.6% / 11.4% |
| Share with WER > 10%, Eka test / windows / turns | 55.5% / 53.0% / 37.8% | 54.8% / 51.7% / 36.9% |
| Units with a dose/negation error, windows | 28.8% | 24.4% |
| Number false-alarm rate, Eka test / windows / turns | 7.1% / 16.1% / 17.5% | 7.0% / 5.6% / 7.0% |
| H1: calibration-in-the-large of P3 on windows (p, Holm) | −0.079 (0.003) | −0.074 (0.003) |
| H2: extra share of errors auto-accepted (p, Holm) | 0.190 (0.003) | 0.200 (0.004) |
| H3a: ΔECE, P4 − P3 (p, Holm) | −0.028 (0.26) | −0.022 (0.41) |
| **H3b**: Δ\|accepted error − 0.20\|, empirical rule (p, Holm) | −0.049 (**0.20**) | −0.101 (**0.006**) |
| H3c: Δ\|accepted error − 0.20\|, plug-in rule (p, Holm) | −0.044 (0.26) | −0.029 (0.41) |
| H4: Δ word-level ECE (p, Holm) | 0.035 (0.003) | 0.034 (0.003) |

The verdicts on H1, H2, H4 (supported) and H3a, H3c (not supported) do not depend on the change.
**H3b does:** it is not supported before the correction and supported after it. Its statistic hangs on
a threshold picked from 194 recalibration windows, and a small change in their labels moves the cut.
We therefore do **not** count H3b as a confirmatory finding. We report it as unstable, which is itself
a deployment lesson about tuning thresholds on a few consultations. The large change in the number
false-alarm rate on PriMock57 (16% → 6%) shows that the earlier gap between domains was an artifact of
"oh" → "0", not an ASR behaviour.
