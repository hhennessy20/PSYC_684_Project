from typing import List, Dict


def assign_roles_by_duration(segments: List[Dict]) -> List[Dict]:
    """
    segments: list of dicts:
        {
            "start": float,
            "end": float,
            "speaker": "SPEAKER_00" | ...
        }

    Returns same list with "role" key added: "PATIENT" or "TESTER".
    """
    duration_per_speaker = {}
    for seg in segments:
        dur = seg["end"] - seg["start"]
        duration_per_speaker[seg["speaker"]] = (
            duration_per_speaker.get(seg["speaker"], 0.0) + dur
        )

    if not duration_per_speaker:
        return segments

    # Speaker with max total duration is considered PATIENT
    patient_speaker = max(duration_per_speaker, key=duration_per_speaker.get)

    for seg in segments:
        seg["role"] = "PATIENT" if seg["speaker"] == patient_speaker else "TESTER"

    return segments
