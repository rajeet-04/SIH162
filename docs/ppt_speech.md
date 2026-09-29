# THERMIS — PPT Speech

## Suggested duration

Approximately 5–7 minutes, followed by a short live demonstration and questions.

## Slide 1 — Title: THERMIS

Good morning/afternoon everyone.

Our project is **THERMIS**, an AI-assisted decision-support system for detecting and triaging industrial fires and persistent thermal sources from satellite observations.

The important word here is *triage*. THERMIS is not designed to replace fire services or make autonomous emergency decisions. It helps an analyst answer three practical questions quickly:

1. Is this hotspot likely to be an industrial fire?
2. Could it instead be persistent industrial heat or flare activity?
3. What evidence supports that assessment?

## Slide 2 — The problem

Satellite systems can detect thermal anomalies, but a hotspot alone does not explain what is happening on the ground.

A hotspot may be an industrial fire, a flare, a wildfire, agricultural burning, or a short-lived false alarm. Analysts must combine location, persistence, nearby facilities, weather, historical activity, and imagery before deciding whether an alert deserves attention.

That manual process is slow, fragmented, and vulnerable to false positives. In an industrial safety context, both missed events and unnecessary escalations are costly.

## Slide 3 — Our objective

THERMIS turns a raw thermal detection into an evidence-backed review queue.

Instead of presenting thousands of undifferentiated points, it ranks events by likely operational importance and shows the evidence behind each ranking.

The system is designed for analysts, emergency-planning teams, and decision-makers who need a fast, explainable first assessment rather than an opaque model score.

## Slide 4 — How the system works

The workflow has five stages.

First, THERMIS ingests NASA FIRMS thermal observations.

Second, it enriches each event with context such as nearby industrial or flare facilities, weather, OpenStreetMap information, and historical thermal persistence.

Third, a calibrated two-stage tabular model estimates the event’s likelihood and risk profile.

Fourth, a supporting image verifier checks available visual evidence.

Finally, a fusion policy combines the signals. When the tabular and image evidence disagree, THERMIS does not force a confident answer. It marks the event as uncertain and sends it for review.

## Slide 5 — What makes THERMIS different

Our main difference is that we do not treat temperature as the whole story.

We use persistence over time, proximity to industrial and flare infrastructure, environmental context, and evidence provenance. Features are strictly backward-looking, so future information does not leak into an earlier prediction.

The model training process also uses chronological splitting. The newest time period is held out as a ranking set rather than being used to tune the model.

This makes the system more realistic for operational monitoring and easier to audit.

## Slide 6 — The dashboard

The dashboard is designed as an analyst command console.

The live feed shows the highest-priority events first. The map provides geographic context. Selecting an event opens its dossier, including the predicted class, confidence, evidence sources, and thermal history.

The timeline view lets the analyst inspect whether the signal is a one-time spike or a persistent source. The evaluation view exposes model and data-quality information instead of hiding it behind a single score.

The application also has an offline-capable mode, so the core demonstration remains usable when external services are unavailable.

## Slide 7 — Evidence and current results

The current project contains 12,165 accepted event rows, including archival observations and live FIRMS detections from India. The system also includes a CPU-loadable image verifier and a working FastAPI service connected to the dashboard.

Our verification checkpoint includes passing API, frontend, offline, and browser rehearsals. The live service can ingest and deduplicate FIRMS observations and maintain local event history.

We are deliberately careful about the performance claim. The available ranking set does not yet have authoritative expert labels, so we have not presented unverified field accuracy as production performance.

The image verifier is supporting evidence only. On its sealed holdout it achieved 0.665 accuracy and 0.78 fire recall, but its no-fire recall was weaker and its confidence was over-calibrated. That is exactly why disagreement is routed to human review.

## Slide 8 — Live demonstration

Now I will show the operational flow.

First, we open the THERMIS console and confirm that the backend is healthy.

Next, we inspect the live event feed and select a high-priority hotspot.

The event dossier shows the predicted category, supporting evidence, confidence, and location.

Then we open the thermal timeline. A persistent signal is treated differently from a single isolated observation.

Finally, we compare the model output with the evidence panel. If the signals disagree, the system explicitly says that the event requires review rather than pretending the uncertainty does not exist.

## Slide 9 — Safety, limitations, and next steps

THERMIS is an advisory decision-support MVP. It is not an autonomous emergency system, a regulatory instrument, or a replacement for field verification.

The next important step is expert labeling of a temporally held-out ranking set. That will allow us to measure field performance honestly and decide whether the model is ready for promotion.

We also need broader regional and seasonal validation, improved geolocation verification, stronger image calibration, and continued monitoring of missing or delayed data sources.

## Slide 10 — Closing

To conclude, THERMIS connects satellite thermal detection with context, machine learning, evidence, and human review.

Its value is not simply that it detects a hot pixel. Its value is that it helps an analyst understand which events deserve attention, why they were prioritised, and where uncertainty remains.

That combination of rapid triage, explainability, offline resilience, and careful validation is the foundation for a safer industrial-fire monitoring workflow.

Thank you. I’m happy to take questions.

## Short answers for likely questions

### Is THERMIS fully autonomous?

No. It is an advisory system. It ranks events and presents evidence for human review.

### What data sources are used?

NASA FIRMS is the primary thermal-event source. The system can also use facility context, weather, OpenStreetMap information, historical persistence, and NASA Earthdata imagery when available.

### What happens without internet access?

The offline demo uses a checked-in derived event bundle and model artifacts. It can still demonstrate the dashboard and decision flow, while live ingestion and external evidence require connectivity.

### Why use both tabular and image models?

The tabular model captures structured context such as persistence and facility proximity. The image verifier provides supporting visual evidence. Their disagreement is useful because it identifies events that should receive closer human attention.

### Can you claim production accuracy today?

Not yet. The current ranking set does not have authoritative expert labels. The responsible next step is to label a temporally held-out set and rerun the evaluation before making a production claim.

