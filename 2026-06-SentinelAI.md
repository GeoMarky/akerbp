# Case Interview – SentinelAI: Decision-Making Under Uncertainty

## Background

The purpose of this case is for you to demonstrate your knowledge, experience, and your quality-and-safety mindset as an AI/ML practitioner — with a particular focus on **uncertainty** and how it shapes **risk exposure in high-consequence decisions**.

You are encouraged to use AI tools during your preparation. However, the goal is not a perfect solution, but to enable a structured and practical discussion around your approach. What we want to explore is your **reasoning**, not your prep.

- Feel free to make reasonable assumptions where needed.
- Focus on the areas where you have strong understanding.
- Your discussion should explicitly address how to balance **decision quality, safety, and the cost of being wrong**.
- We'd prefer you talk us through **markdown notes alongside code in a repo** rather than build slides — a polished `.ppt` is not expected (or wanted).
- This case spans many modalities and responsibilities **on purpose** — you are **not** expected to cover all of it. See **Choosing your focus** below and go deep where you're strongest.

### Working with code

We like to talk **around** code rather than through slides. You're welcome to use a codebase to ground the discussion — your own, an open-source project, or a small synthetic scaffold you put together. It does **not** need to be polished, complete, or built specifically for this case: a rough notebook or a few modules you can point at is plenty. Markdown files are perfectly fine for structuring your points. We're interested in how you reason about the code and the decisions behind it — not in a finished deliverable or a rehearsed presentation.

- ✅ **Use AI agents/tools freely** while preparing and during the session — to explore a repo, draft tests, sketch a design, or check yourself.
- ✅ Bring an **existing codebase** if it helps you illustrate a point.
- ❌ But don't bring **proprietary code or data** — yours from a current/previous employer, Aker BP's, or any third party's. Any codebase you show must be **your own, open-source, or synthetic**, and don't paste confidential material into any AI tool.
- ❗ Be ready to **defend and modify** anything an AI agent produced. "The agent wrote it" is not an explanation.

### Suggested format (rough timing, not a strict script)

Skew the time toward your focus area (see **Choosing your focus**) — the split below is a default, not a requirement.

- **~5-10 min** — Walk us through your mental model: how uncertainty flows from a raw reading to the final action (Part 1). Point at code where it helps.
- **~10-15 min** — Your sharpest critiques of the design, how you'd adapt it when the world changes, and how you would make it trustworthy in production (Parts 2–4). We may ask you to do this partly as a code/design review or mentoring conversation with the interviewer.
- **Remaining time** — Open discussion. We'll dig into the areas you went deep on.

## System Overview

**SentinelAI** is an internal AI decision-support system that continuously monitors a **safety-critical asset** and helps decide, in near-real-time, whether a hazardous condition is developing.

For each incoming reading it produces a score and converts that score into one of three actions:

- **Nominal** — log, do nothing.
- **Advisory** — surface to a human operator for attention.
- **Alarm** — trigger escalation and a (possibly automated) protective action, e.g. a shutdown/trip.

When the score is elevated, an LLM-based reasoning layer retrieves relevant context — design specifications, operating-envelope limits, procedures, and **historical events with the decisions and outcomes that followed** — and produces a recommended action plus a written rationale shown to the operator.

> **Modality.** Pick the lens that fits your background — the reasoning is identical:
>
> - **Tabular / time-series** (default): streaming sensor readings (pressure, temperature, vibration, flow…).
> - **Signal / acoustics:** a continuous acoustic or vibration stream; the "reading" is a windowed feature frame.
> - **Computer vision:** periodic inspection imagery; the "reading" is a frame or detection.
> - **Language processing:** operator logs / maintenance reports / alarms-as-text; the "reading" is a document.

## Business Goal

The business wants to roll out SentinelAI quickly to:

- catch developing hazards earlier
- reduce operator workload and alarm fatigue

However, they want to ensure that the solution:

- does not introduce unacceptable safety, environmental, or integrity risk
- makes high-consequence decisions that are calibrated, defensible, and auditable

## Choosing your focus

This case is deliberately broader than any single role — it spans **modalities** (computer vision, language, time-series, acoustics) and **responsibilities** (modeling, infrastructure, software & CI/CD, MLOps, deployment, monitoring, governance, collaboration, mentoring). **You are not expected to cover all of it.**

Please pick:

1. **One modality lens** (see _System Overview_) — the data type you're most fluent in.
2. **One or two depth areas** from the table below — your home turf.

Go deep where you're strong. For everything else, a sentence on how you'd approach it — or who you'd partner with — is plenty. We'd much rather see real depth in your domain than shallow coverage of all of it. **Tell us your pick at the start** so we can steer the discussion there.

| If your home turf is…                                                    | Go deep on…                                                                                                  | A light touch is fine on…      |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| Modeling, uncertainty, statistics (data scientist / principal)           | Parts 1 & 2 — calibration, correlation, thresholds, decision theory                                          | Infra / CI-CD specifics        |
| A modality specialism (CV / acoustics & signal / language / time-series) | Parts 1 & 2 through your modality lens — representation, noise/SNR, domain shift, detector calibration       | Deployment plumbing            |
| LLM / language / RAG                                                     | Part 1 (retrieval & generation uncertainty) + Part 2 LLM bullets + Part 4 LLM/RAG evals                      | Numeric-model internals        |
| Infrastructure, software & CI/CD, integration                            | Part 4 — training/serving parity, IaC, versioning of model + prompt + index, deployment; Part 3 system seams | Uncertainty math               |
| MLOps / lifecycle                                                        | Part 4 — retraining triggers, validation gates, rollback, reproducibility                                    | Modality internals             |
| Monitoring / observability / reliability                                 | Part 4 — drift, calibration-over-time, alarm rates, faithfulness monitoring, auditability                    | Low-level modeling derivations |
| Product / safety / governance                                            | Part 3 (cost & policy adaptation) + Part 4 governance & human accountability                                 | Implementation detail          |
| Technical leadership / mentoring                                         | Parts 2–4 through a collaboration lens — code/design review, documentation, stakeholder alignment, mentoring | Modality internals             |

**One thing we ask of everyone, whatever your focus:** be able to speak to the _core safety idea_ — how uncertainty in a reading should shape a high-consequence decision, why you'd never fully hand the trip to an unmonitored model, and how you would help others work safely with that risk. How deep you go beyond that is up to your role.

## Your Tasks

Treat the four technical parts as a **menu, not a checklist** — weight your time toward your focus area, and feel free to skim the rest.

For Parts 2–4, we may ask you to present part of your reasoning as if you were helping a teammate improve the system. That could mean walking through a code or design review, explaining trade-offs to a less experienced engineer, or collaborating with the interviewer as a stand-in for a platform, domain, product, or safety stakeholder.

### Part 1 — Explain: trace the uncertainty

Walk us through how uncertainty enters and flows from a **raw reading to the final action**. Where does it originate — sensor noise, sampling, labelling, representation, the model, the threshold? Distinguish the kinds of uncertainty that matter and where each lives. What does the score "0.5" actually represent by the time it reaches the decision layer — and what does it _not_ tell you?

Then extend the trace through the **retrieval + LLM layer**: there are now two coupled stochastic systems. Where does **retrieval uncertainty** live (did we fetch the right, current context?) and where does **generation uncertainty** live (sampling, hallucination, faithfulness of the rationale)? How does the LLM's uncertainty _compose_ with the numeric model's — dampen or amplify?

### Part 2 — Critique: the modeling choices

Pull apart the design (see **Current Controls** below). We'll be especially interested in:

- **Calibration** — is a raw model score a probability? How would you check, and why does calibration matter _specifically_ when a score feeds a cost-weighted decision?
- **Correlation / dependence** — channels (or frames, or features) are treated as independent. When is that wrong, and what's the consequence for the decision?
- **Thresholds** — what's wrong with a single fixed threshold chosen on F1 for a rare, high-consequence event? What would you optimise instead?
- **Escalation** — is a binary auto-trip the right policy? What would you add (hysteresis, dwell time, human-in-the-loop, abstention)?
- **Retrieval (RAG) quality** — naive chunking + fixed top-k. How would you measure retrieval recall/precision, and guarantee retrieved limits are the _current_ revision?
- **LLM calibration & faithfulness** — is "90% confident" meaningful? The rationale reads convincingly but may not reflect why the score was high, and may cite limits that aren't in any source. How do you enforce provenance and detect ungrounded claims?
- **Automation bias & poisoned context** — learning from past dispositions inherits past human error; ingested free-text is untrusted and could carry injected instructions. How do you defend against both?

### Part 3 — Adapt: when the world changes

The framework has to survive change. Reason through how your design responds when:

- the **consequence cost shifts** — during a production-critical window a false shutdown suddenly costs far more, or a regulator tightens the acceptable miss rate;
- the **safety policy changes** — auto-trip is no longer permitted and every Alarm must be human-confirmed (so the LLM rationale becomes load-bearing); or a new asset type is added with **no labelled history and nothing to retrieve**.

_(We may introduce one of these live and ask you to revise on the spot.)_

### Part 4 — Production concerns

How would you make this trustworthy in operation? Cover:

- **MLOps** — training/serving parity, retraining triggers, versioning of the model, the prompt, _and_ the document index together.
- **Validation** — what you'd test beyond accuracy: calibration, robustness, slice performance, backtesting on rare events; plus **LLM/RAG evals** (retrieval metrics, groundedness/faithfulness, regression suites, red-teaming for injection).
- **Observability** — input/score/decision drift, alarm rates, latency; retrieval-quality and faithfulness monitoring.
- **Auditability** — reconstructing _why_ a given Alarm fired months later. Note the rationale + cited sources _are_ the audit trail for the LLM layer.
- **Governance** — sign-off, change control, human accountability, fail-safe behaviour, and the bigger question of when a generative model is allowed to be advisory-only vs allowed to gate a protective action at all.

## Constraints

- The asset is **high-consequence**: a missed event can cause a safety, environmental, or integrity incident.
- Costs are **asymmetric**: a missed event is far more costly than a false alarm — but frequent false alarms cause **alarm fatigue** and erode trust until operators ignore the system.
- Hazardous events are **rare**, and historical labels were assigned **after the fact**.
- The platform team has **limited capacity** for manual review and bespoke monitoring.
- The system must remain **useful enough that operators don't bypass it**.
- Sensitive information could exist in **tickets, logs, and historical reports**.
- The system is expected to **add more data sources and tools** and **evolve toward semi-autonomous workflows** over time.

## Architecture Overview

SentinelAI consists of the following components:

### Data Ingestion

- Multiple **correlated** sensor/signal channels; light cleaning; gaps forward-filled.
- A per-window (or per-frame / per-document) feature representation.

### Detection Model

- Outputs a single score in [0,1], taken **directly from the final sigmoid/softmax** as "the probability of a hazardous condition."

### Retrieval + LLM Reasoning Layer

- Retrieves context by **embedding similarity (fixed top-k)** from a document store: design-basis / operating-envelope specs, equipment data sheets, procedures, and **historical events with their dispositions and outcomes**.
- An LLM reasons over the retrieved context plus current readings to produce a recommended action, a free-text rationale, and a **verbalized confidence**.

### Decision & Escalation

- A **single fixed threshold** on the numeric score (≥ 0.5 → Alarm; one fixed band → Advisory).
- The LLM's recommendation and rationale are shown to the operator and written to the escalation record.
- **Alarm fires the protective action automatically.**

### Data Sources

- Design documentation and procedures (SharePoint / OneDrive).
- Historical ticket / event data with past decisions and outcomes.

### Dev & Deployment

- Model code, prompt templates, and tool definitions stored in GitHub.
- Changes deployed via CI/CD using infrastructure-as-code.

## Current Controls

- Raw model score used directly as a probability.
- Single fixed threshold (0.5), chosen to maximise F1.
- Channels treated as independent.
- LLM verbalized confidence and rationale taken at face value.
- Retrieval index built once; **not** version-tracked against the live design revision.
- Basic logging; **no** calibration, drift, retrieval-quality, or faithfulness monitoring.
- Automatic protective action on Alarm.

> The controls above are **intentionally weak** — identifying, prioritising, and fixing them is part of the exercise.



