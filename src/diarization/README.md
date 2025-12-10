# Speaker Diarization

Separates patient speech from interviewer speech in ADReSS audio recordings.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

### Diarize Audio

```bash
python -m src.diarization.diarize
```

Parses `.cha` transcription files and extracts patient (PAR) segments from audio. Outputs patient-only audio to `data/patient_audio/`.

### Audio Preprocessing

```bash
python -m src.diarization.audio_prep
```

Standardizes audio files to 16kHz mono format.

## Modules

- `diarize.py` - Main diarization logic using `.cha` file timestamps
- `audio_prep.py` - Audio standardization (sample rate, channels)
- `role_heuristics.py` - Speaker role identification heuristics

## Output

- `data/patient_audio/` - Patient-only audio segments
- `data/diarization/` - RTTM-like diarization files

