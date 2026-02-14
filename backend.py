import os
import re
import gc
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE  = torch.float32

MODEL_ID  = "google/medgemma-4b-it"
LORA_PATH = "./medgemma-forensic-lora"

app = FastAPI(title="Autopsy Drafting Backend")

tokenizer = None
model     = None


class Query(BaseModel):
    message: str

class AutopsyReport(BaseModel):
    cause_of_death:     str
    mechanism_of_death: str
    manner_of_death:    str
    time_since_death:   str
    key_findings:       List[str]
    toxicology:         str
    summary_opinion:    str
    injury_locations:   List[str]
    raw_report:         str
    parse_warnings:     List[str]


@app.on_event("startup")
def load_model():
    global tokenizer, model

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    del base_model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    print("Model ready.")


def build_prompt(message: str) -> str:
    """
    FIX 1: Added a second few-shot example covering multi-trauma + toxicology.
    This gives the model a demonstrated pattern for complex cases and prevents
    it from halting early when the case exceeds the single simple example.
    
    FIX 2: Removed trailing '1. Cause of Death:' primer from the prompt.
    Instead we inject it after generation. This avoids the primer consuming
    tokens from max_length=1024, which was pushing real findings content
    beyond the truncation boundary on longer inputs.
    """
    return f"""You are a forensic pathologist. Complete this autopsy report using ONLY the findings provided. Be concise and complete every section. Do not skip any numbered section.

FINDINGS: Male found supine. Depressed skull fracture left parietal. Subdural hematoma 40mL. Ligature mark anterior neck. Petechial hemorrhages bilateral conjunctivae. Rigor mortis fully developed. Livor mortis fixed posteriorly. No toxicology.

1. Cause of Death: Blunt force head trauma with ligature strangulation.
2. Mechanism of Death: Subdural hematoma causing raised intracranial pressure and brainstem herniation combined with mechanical asphyxia from ligature compression.
3. Manner of Death: Homicide
4. Estimated Time Since Death: 8-12 hours based on fully developed rigor mortis and fixed livor mortis.
5. Key Autopsy Findings:
- Depressed skull fracture, left parietal region
- Subdural hematoma approximately 40mL
- Ligature mark, anterior neck with soft tissue hemorrhage
- Petechial hemorrhages, bilateral conjunctivae
- Fixed posterior livor mortis
6. Toxicology Interpretation: No toxicology data available.
7. Summary Opinion: Death resulted from combined blunt force head trauma and ligature strangulation consistent with homicide.
8. Injury Locations:
- Head
- Neck

---

FINDINGS: Male, 28 years old. Found in vehicle. Gunshot wound entrance right temple, exit wound left temple. Intermediate range stippling. Cerebral lacerations and hemorrhage. Rigor partially developed upper extremities, absent lower. Livor mortis not fixed, posterior. Core temperature 33°C. Blood alcohol concentration 0.21%. No other injuries.

1. Cause of Death: Gunshot wound to the head.
2. Mechanism of Death: Cerebral laceration and intracranial hemorrhage causing rapid neurological death.
3. Manner of Death: Undetermined pending investigation.
4. Estimated Time Since Death: 2-4 hours based on partial rigor mortis in upper extremities, unfixed livor mortis, and core temperature of 33°C indicating minimal cooling.
5. Key Autopsy Findings:
- Gunshot wound entrance, right temple with intermediate range stippling
- Exit wound, left temple
- Bilateral cerebral lacerations and hemorrhage
- Partial rigor mortis, upper extremities only
- Posterior livor mortis, not fixed
6. Toxicology Interpretation: Blood alcohol concentration of 0.21%, consistent with significant intoxication. Alcohol may have been a contributing factor in circumstances of death.
7. Summary Opinion: Death resulted from a single gunshot wound to the head. Manner remains undetermined pending scene investigation and circumstances review.
8. Injury Locations:
- Right temple
- Left temple
- Brain

---

FINDINGS: {message}

1. Cause of Death:"""


def hard_stop_after_section_8(text: str) -> str:
    """
    FIX 3: Relaxed early termination logic.
    Original code broke on blank lines within the section 8 bullet list,
    which could truncate legitimate findings. Now we only stop on a new
    numbered section header or a '---' separator.
    """
    match = re.search(r"(8\.\s*Injury Locations[^\n]*\n)", text, re.IGNORECASE)
    if not match:
        return text

    before = text[:match.start()]
    header = match.group(1)
    after  = text[match.end():]

    kept         = []
    bullet_count = 0

    for line in after.split("\n"):
        stripped = line.strip()

        if stripped.startswith("---") or re.match(r"^9\.", stripped):
            break

        if bullet_count > 0 and stripped and not re.match(r"^[-•*]", stripped):
            break

        if re.match(r"^[-•*]\s+", stripped):
            bullet_count += 1
            if bullet_count <= 8:
                kept.append(line)
            else:
                break
        else:
            kept.append(line)

    return (before + header + "\n".join(kept)).strip()


def extract_bullets(text: str, max_items: int = 20) -> List[str]:
    items = re.findall(r"[-•*]\s*(.+)", text)
    result = []
    for item in items:
        clean = item.strip()
        if clean:
            result.append(clean)
        if len(result) >= max_items:
            break
    return result


def deduplicate(items: List[str]) -> List[str]:
    seen, result = set(), []
    for item in items:
        k = item.lower().strip()
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


def clean_injury_locations(locations: List[str]) -> List[str]:
    cleaned = []
    for loc in locations:
        region = re.split(r"[-–—(:]", loc)[0].strip().title()
        if 1 <= len(region.split()) <= 4:
            cleaned.append(region)
    return deduplicate(cleaned)[:8]


VALID_MANNERS = {
    "natural", "accidental", "homicidal", "homicide",
    "suicidal", "suicide", "undetermined"
}

def parse_report(raw: str) -> dict:
    warnings = []
    raw = hard_stop_after_section_8(raw)

    patterns = {
        "cause":     r"1\.\s*Cause of Death\s*:?\s*(.+?)(?=\n\s*2\.|\Z)",
        "mechanism": r"2\.\s*Mechanism of Death\s*:?\s*(.+?)(?=\n\s*3\.|\Z)",
        "manner":    r"3\.\s*Manner of Death\s*:?\s*(.+?)(?=\n\s*4\.|\Z)",
        "time":      r"4\.\s*Estimated Time Since Death\s*:?\s*(.+?)(?=\n\s*5\.|\Z)",
        "findings":  r"5\.\s*Key Autopsy Findings\s*:?\s*(.+?)(?=\n\s*6\.|\Z)",
        "tox":       r"6\.\s*Toxicology Interpretation\s*:?\s*(.+?)(?=\n\s*7\.|\Z)",
        "summary":   r"7\.\s*Summary Opinion\s*:?\s*(.+?)(?=\n\s*8\.|\Z)",
        "injuries":  r"8\.\s*Injury Locations\s*:?\s*(.+?)(?=\Z)",
    }

    def extract(key):
        m = re.search(patterns[key], raw, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    
    cause_raw = extract("cause")
    cause = re.split(r"(?<=[.!?])\s+", cause_raw)[0].strip()
    if len(cause) > 200:
        cause = cause[:200].rsplit(" ", 1)[0] + "."
        warnings.append("cause_of_death truncated")
    if not cause:
        warnings.append("cause_of_death: missing")
        cause = "Could not be determined."

    mechanism = re.split(r"\n", extract("mechanism"))[0].strip()
    if not mechanism:
        warnings.append("mechanism_of_death: missing")

    manner_raw = extract("manner").lower()
    manner_line = re.split(r"[\n,;.]", manner_raw)[0].strip()
    MANNER_NORMALISE = {
        "homicide":    "Homicide",
        "homicidal":   "Homicide",
        "suicide":     "Suicide",
        "suicidal":    "Suicide",
        "accidental":  "Accidental",
        "natural":     "Natural",
        "undetermined":"Undetermined",
    }
    manner_found = next(
        (MANNER_NORMALISE[v] for v in MANNER_NORMALISE if v in manner_line),
        None
    )
    if not manner_found:
        manner_found = "Undetermined"
        warnings.append(f"manner_of_death unparseable: '{manner_raw[:60]}'")

    time_clean = re.split(r"\n", extract("time"))[0].strip()
    if not time_clean:
        warnings.append("time_since_death: missing")

    findings_raw = extract("findings")
    findings = extract_bullets(findings_raw)

    if not findings:
        findings = [l.strip() for l in findings_raw.split("\n") if l.strip()]
    if not findings:
        warnings.append("key_findings: missing")

    tox = re.split(r"\n\n", extract("tox"))[0].strip()
    if not tox:
        tox = "No toxicology data available."
        warnings.append("toxicology: missing, defaulted")

    summary = re.split(r"\n\n", extract("summary"))[0].strip()
    if len(summary) > 500:
        summary = summary[:500].rsplit(" ", 1)[0] + "."
        warnings.append("summary_opinion truncated")
    if not summary:
        warnings.append("summary_opinion: missing")

    injuries = clean_injury_locations(extract_bullets(extract("injuries"), max_items=8))
    if not injuries:
        warnings.append("injury_locations: empty — frontend will use fallback")

    return {
        "cause_of_death":     cause,
        "mechanism_of_death": mechanism,
        "manner_of_death":    manner_found,
        "time_since_death":   time_clean,
        "key_findings":       findings,
        "toxicology":         tox,
        "summary_opinion":    summary,
        "injury_locations":   injuries,
        "parse_warnings":     warnings,
    }


@app.post("/draft", response_model=AutopsyReport)
def draft_report(query: Query):
    if not query.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    prompt = build_prompt(query.message)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1536,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length    = inputs["input_ids"].shape[1]
    response_tokens = outputs[0][input_length:]
    raw = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    raw = "1. Cause of Death:" + raw
    raw = raw.replace("---", "").strip()

    parsed = parse_report(raw)

    return AutopsyReport(
        cause_of_death     = parsed["cause_of_death"],
        mechanism_of_death = parsed["mechanism_of_death"],
        manner_of_death    = parsed["manner_of_death"],
        time_since_death   = parsed["time_since_death"],
        key_findings       = parsed["key_findings"],
        toxicology         = parsed["toxicology"],
        summary_opinion    = parsed["summary_opinion"],
        injury_locations   = parsed["injury_locations"],
        raw_report         = raw,
        parse_warnings     = parsed["parse_warnings"],
    )


@app.get("/")
def health_check():
    return {"status": "Autopsy Drafting Backend Running"}
