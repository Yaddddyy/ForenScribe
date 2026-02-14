import streamlit as st
import requests
import streamlit.components.v1 as components
import re

FALLBACK_KEYWORDS = {
    "Head":      ["skull fracture", "head trauma", "scalp", "cranial",
                  "temporal region", "parietal region", "occipital",
                  "frontal lobe", "subdural", "epidural", "intracranial",
                  "brain", "facial", "orbital", "mandible", "petechial"],
    "Neck":      ["neck", "cervical", "ligature", "larynx", "trachea",
                  "anterior neck", "strangulation", "hyoid"],
    "Chest":     ["chest", "thorax", "thoracic", "rib fracture", "sternum",
                  "pneumothorax", "hemothorax", "cardiac", "pulmonary",
                  "aorta", "pericardium"],
    "Abdomen":   ["abdomen", "abdominal", "liver laceration", "splenic",
                  "mesentery", "bowel", "retroperitoneal", "pancreas"],
    "Pelvis":    ["pelvis", "pelvic fracture", "hip fracture",
                  "pubic symphysis", "sacrum", "ilium"],
    "Spine":     ["vertebra", "spinal cord", "lumbar", "cervical spine",
                  "thoracic spine", "spinal fracture"],
    "Left Arm":  ["left arm", "left humer", "left radius", "left ulna",
                  "left forearm", "left wrist", "left hand", "left shoulder"],
    "Right Arm": ["right arm", "right humer", "right radius", "right ulna",
                  "right forearm", "right wrist", "right hand", "right shoulder"],
    "Left Leg":  ["left leg", "left femur", "left tibia", "left fibula",
                  "left knee", "left ankle", "left foot", "left thigh"],
    "Right Leg": ["right leg", "right femur", "right tibia", "right fibula",
                  "right knee", "right ankle", "right foot", "right thigh"],
}

def extract_zones_from_findings(findings_text: str) -> list:
    """
    Scan ONLY the original findings input for specific anatomical phrases.
    Avoids false positives from generic words in the report prose.
    """
    text_lower = findings_text.lower()
    found = set()
    for zone, keywords in FALLBACK_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.add(zone)
                break
    return list(found)

ZONE_KEYWORDS = {
    "Head":      ["head", "skull", "scalp", "face", "cranial", "temporal",
                  "parietal", "occipital", "frontal", "orbital", "mandible",
                  "petechial", "conjunctiv"],
    "Neck":      ["neck", "cervical", "larynx", "trachea", "ligature", "hyoid"],
    "Chest":     ["chest", "thorax", "thoracic", "rib", "sternum",
                  "clavicle", "lung", "heart", "cardiac", "pulmonary"],
    "Abdomen":   ["abdomen", "abdominal", "liver", "spleen", "stomach",
                  "bowel", "intestine", "kidney", "renal", "pancreas"],
    "Pelvis":    ["pelvis", "pelvic", "hip", "sacrum", "bladder", "groin"],
    "Spine":     ["spine", "spinal", "vertebra", "lumbar", "back"],
    "Left Arm":  ["left arm", "left upper", "left elbow", "left forearm",
                  "left wrist", "left hand", "left shoulder"],
    "Right Arm": ["right arm", "right upper", "right elbow", "right forearm",
                  "right wrist", "right hand", "right shoulder"],
    "Left Leg":  ["left leg", "left thigh", "left knee", "left shin",
                  "left ankle", "left foot"],
    "Right Leg": ["right leg", "right thigh", "right knee", "right shin",
                  "right ankle", "right foot"],
}

def map_regions_to_zones(regions: list) -> list:
    found = set()
    for region in regions:
        rl = region.lower()
        for zone, keywords in ZONE_KEYWORDS.items():
            for kw in keywords:
                if kw in rl:
                    found.add(zone)
                    break
    return list(found)

def render_3d_body(highlight_zones: list):
    zones_json = str(highlight_zones).replace("'", '"')

    html_code = f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 100%; height: 100%;
    background: #1a1a2e;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
  }}
  #hint {{
    position: fixed; top: 8px; left: 50%;
    transform: translateX(-50%);
    color: #888; font: 11px Arial, sans-serif;
  }}
  #legend {{
    position: fixed; bottom: 8px; left: 50%;
    transform: translateX(-50%);
    color: #fff; font: 12px Arial, sans-serif;
    display: flex; gap: 16px;
  }}
  .dot {{
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 4px; vertical-align: middle;
  }}
</style>
</head>
<body>
<div id="hint">Drag to rotate</div>
<div id="legend">
  <span><span class="dot" style="background:#c8a882"></span>Normal</span>
  <span><span class="dot" style="background:#ff2222"></span>Injured</span>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
  const highlightZones = {zones_json};

  const W = window.innerWidth;
  const H = window.innerHeight;

  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(W, H);
  renderer.shadowMap.enabled = true;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 1000);
  camera.position.set(0, 0.8, 6.5);
  camera.lookAt(0, 0.5, 0);

  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const sun = new THREE.DirectionalLight(0xffffff, 0.85);
  sun.position.set(3, 6, 5);
  sun.castShadow = true;
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x8899ff, 0.3);
  fill.position.set(-4, 2, -4);
  scene.add(fill);

  // Materials
  function mat(zone) {{
    return highlightZones.includes(zone)
      ? new THREE.MeshPhongMaterial({{ color: 0xff2222, emissive: 0x661111, shininess: 70 }})
      : new THREE.MeshPhongMaterial({{ color: 0xc8a882, shininess: 25 }});
  }}

  const body = new THREE.Group();

  function box(w, h, d, x, y, z, zone) {{
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat(zone));
    m.position.set(x, y, z); m.castShadow = true; body.add(m);
  }}
  function sphere(rx, ry, rz, x, y, z, zone) {{
    const m = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 20), mat(zone));
    m.scale.set(rx, ry, rz); m.position.set(x, y, z); body.add(m);
  }}
  function cyl(rt, rb, h, x, y, z, zone) {{
    const m = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, 14), mat(zone));
    m.position.set(x, y, z); m.castShadow = true; body.add(m);
  }}

  // ── Head & Neck ──────────────────────────
  sphere(0.50, 0.58, 0.50,   0,    3.50,  0,      "Head");
  cyl(0.17, 0.19, 0.44,      0,    3.06,  0,      "Neck");

  // ── Torso ────────────────────────────────
  box(1.28, 1.05, 0.58,      0,    2.22,  0,      "Chest");
  box(1.12, 0.75, 0.52,      0,    1.28,  0,      "Abdomen");
  box(1.18, 0.58, 0.52,      0,    0.64,  0,      "Pelvis");

  // ── Spine (thin strip behind) ─────────────
  box(0.14, 2.15, 0.10,      0,    1.72, -0.30,   "Spine");

  // ── Arms ─────────────────────────────────
  cyl(0.19, 0.17, 0.88,     -0.85, 2.28,  0,      "Left Arm");
  cyl(0.15, 0.13, 0.78,     -0.85, 1.38,  0,      "Left Arm");
  cyl(0.19, 0.17, 0.88,      0.85, 2.28,  0,      "Right Arm");
  cyl(0.15, 0.13, 0.78,      0.85, 1.38,  0,      "Right Arm");

  // ── Legs ─────────────────────────────────
  cyl(0.25, 0.21, 1.08,     -0.36, -0.18, 0,      "Left Leg");
  cyl(0.19, 0.15, 0.98,     -0.36, -1.30, 0,      "Left Leg");
  box(0.27, 0.17, 0.52,     -0.36, -1.90, 0.14,   "Left Leg");

  cyl(0.25, 0.21, 1.08,      0.36, -0.18, 0,      "Right Leg");
  cyl(0.19, 0.15, 0.98,      0.36, -1.30, 0,      "Right Leg");
  box(0.27, 0.17, 0.52,      0.36, -1.90, 0.14,   "Right Leg");

  scene.add(body);

  // Floor shadow
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(12, 12),
    new THREE.ShadowMaterial({{ opacity: 0.2 }})
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -2.15;
  floor.receiveShadow = true;
  scene.add(floor);

  // Mouse drag
  let dragging = false, prevX = 0, autoRotate = true;
  renderer.domElement.addEventListener("mousedown", e => {{
    dragging = true; autoRotate = false; prevX = e.clientX;
  }});
  window.addEventListener("mouseup",   () => dragging = false);
  window.addEventListener("mousemove", e => {{
    if (!dragging) return;
    body.rotation.y += (e.clientX - prevX) * 0.01;
    prevX = e.clientX;
  }});

  function animate() {{
    requestAnimationFrame(animate);
    if (autoRotate) body.rotation.y += 0.004;
    renderer.render(scene, camera);
  }}
  animate();
</script>
</body>
</html>"""

    components.html(html_code, height=540)

st.set_page_config(
    page_title="Autopsy Report Drafting Assistant",
    layout="wide"
)
st.title("Offline Autopsy Report Drafting Assistant")

findings = st.text_area("Enter Autopsy Findings:", height=250)

if st.button("Generate Report"):
    if not findings.strip():
        st.warning("Please enter autopsy findings.")
    else:
        with st.spinner("Generating structured report..."):
            try:
                resp = requests.post(
                    "http://127.0.0.1:8000/draft",
                    json={"message": findings},
                    timeout=600
                )

                if resp.status_code == 200:
                    data = resp.json()

                    col_report, col_3d = st.columns([1.2, 1])

                    with col_report:
                        st.subheader("Generated Autopsy Report")

                        st.markdown(f"**1. Cause of Death:** {data['cause_of_death']}")
                        st.markdown(f"**2. Mechanism of Death:** {data['mechanism_of_death']}")
                        st.markdown(f"**3. Manner of Death:** {data['manner_of_death']}")
                        st.markdown(f"**4. Estimated Time Since Death:** {data['time_since_death']}")

                        st.markdown("**5. Key Autopsy Findings:**")
                        for f in data["key_findings"]:
                            st.write("•", f)

                        st.markdown(f"**6. Toxicology Interpretation:** {data['toxicology']}")
                        st.markdown(f"**7. Summary Opinion:** {data['summary_opinion']}")

                        st.markdown("**8. Injury Locations:**")
                        for inj in data["injury_locations"]:
                            st.write("•", inj)

                        if data.get("parse_warnings"):
                            with st.expander("⚠️ Parse Warnings"):
                                for w in data["parse_warnings"]:
                                    st.write("•", w)

                    with col_3d:
                        st.subheader("Injury Map")
                    
                        zones = map_regions_to_zones(data["injury_locations"])
                        
                        if not zones:
                            zones = extract_zones_from_findings(findings)
                            if zones:
                                st.caption(
                                    f"ℹ️ Inferred from findings: {', '.join(sorted(zones))}"
                                )
                            else:
                                st.caption("No regions detected.")
                        else:
                            st.caption(f"Highlighted: {', '.join(sorted(zones))}")

                        render_3d_body(zones)

                else:
                    st.error(f"Backend error {resp.status_code}: {resp.text}")

            except Exception as e:
                st.error(f"Connection error: {e}")
