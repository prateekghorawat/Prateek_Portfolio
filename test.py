import streamlit as st
import os
import base64
import streamlit.components.v1 as components

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prateek Ghorawat – Portfolio",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def img_to_b64(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

def icon_html(folder: str) -> str:
    if not os.path.isdir(folder):
        return ""
    items = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
    ])
    parts = []
    for f in items:
        b64 = img_to_b64(os.path.join(folder, f))
        if b64:
            ext = f.rsplit(".", 1)[-1].lower()
            mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
            label = f.rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()
            parts.append(f"""
                <div class="icon-item">
                    <img src="data:{mime};base64,{b64}" alt="{label}" />
                    <span>{label}</span>
                </div>""")
    return "\n".join(parts)

def project_img_html(path: str) -> str:
    b64 = img_to_b64(path)
    if b64:
        ext = path.rsplit(".", 1)[-1].lower()
        mime = f"image/{ext}"
        return f'<img src="data:{mime};base64,{b64}" alt="project" class="proj-img" />'
    return '<div class="proj-img-placeholder">🔍</div>'

# ─────────────────────────────────────────────
# PROFILE IMAGE
# ─────────────────────────────────────────────
profile_b64 = img_to_b64("self_gen.png")
profile_src = f"data:image/png;base64,{profile_b64}" if profile_b64 else ""
profile_tag = (
    f'<img src="{profile_src}" class="profile-img" alt="Prateek Ghorawat" />'
    if profile_src
    else '<div class="profile-img profile-placeholder">👤</div>'
)

# ─────────────────────────────────────────────
# PROJECTS DATA
# ─────────────────────────────────────────────
projects = [
    {
        "title": "Multi-Agent AI System for Process Intelligence",
        "tech": "LangChain · LangGraph · MCP · Python · LLMs",
        "description": "Designed and deployed a multi-agent AI orchestration system at BMW Group enabling autonomous reasoning, coordination, and automation across complex business workflows. Integrated LangGraph for stateful agent control with real-time monitoring and structured output validation.",
        "link_url": "https://huggingface.co/spaces/pjpj4545/Prateek_portfolio",
        "link_text": "View Demo",
        "image_path": "projects_images/project_2.png"
    },
    {
        "title": "AI Prediction System for Trade & Risk Analytics",
        "tech": "Scikit-learn · ML Pipelines · Feature Engineering · Python",
        "description": "Built an end-to-end ML system achieving ~95% accuracy in anomaly and risk detection on large-scale operational datasets. Automated insight generation using LLM-based summarization, reducing analyst workload by 80%.",
        "link_url": "https://github.com/prateekghorawat",
        "link_text": "View Project",
        "image_path": "projects_images/project_1.png"
    },
    {
        "title": "Wasserstein GAN — Realistic Portrait Generation",
        "tech": "PyTorch · WGAN-GP · CelebA Dataset · NumPy · Matplotlib",
        "description": "Built a deep learning system generating realistic human portraits by training on 200,000+ celebrity face images. Resolved training instability and mode collapse via Wasserstein loss with gradient penalty (WGAN-GP).",
        "link_url": "https://github.com/prateekghorawat/Generative-AI/tree/main/CelebA-WGAN-Exploring-Advanced-Image-Generation-with-WGAN-and-CelebA-Dataset-main",
        "link_text": "View Project",
        "image_path": "projects_images/project_1.png"
    },
    {
        "title": "Sales Analytics Data Warehouse Modernization",
        "tech": "SQL Server · SSIS · Medallion Architecture · Star Schema · Power BI",
        "description": "Architected a modern data warehouse using Bronze → Silver → Gold layers for systematic data quality improvement. Reduced query latency by 50% with optimized star schema design and end-to-end ETL pipelines.",
        "link_url": "https://github.com/prateekghorawat/Data-Warehousing-/tree/main",
        "link_text": "View Project",
        "image_path": "projects_images/project_3.png"
    },
    {
        "title": "Fovea Detection & Eye Tracking System",
        "tech": "PyTorch · OpenCV · YOLO · CNN · CUDA · Medical Imaging",
        "description": "Medical imaging solution combining CNN architectures with YOLO object detection to identify and track the fovea in retinal photographs. Enables automated detection of retinal changes indicative of vision deterioration.",
        "link_url": "https://github.com/prateekghorawat/Computer-Vision-Projects",
        "link_text": "View Project",
        "image_path": "projects_images/project_4.png"
    },
]

# Build project cards HTML (alternating image side)
projects_html = ""
for i, p in enumerate(projects):
    img = project_img_html(p["image_path"])
    content = f"""
        <div class="proj-content">
            <span class="proj-tech">{p['tech']}</span>
            <h3 class="proj-title">{p['title']}</h3>
            <p class="proj-desc">{p['description']}</p>
            <a href="{p['link_url']}" class="proj-link" target="_blank">{p['link_text']} →</a>
        </div>
        <div class="proj-visual">{img}</div>
    """ if i % 2 == 0 else f"""
        <div class="proj-visual">{img}</div>
        <div class="proj-content">
            <span class="proj-tech">{p['tech']}</span>
            <h3 class="proj-title">{p['title']}</h3>
            <p class="proj-desc">{p['description']}</p>
            <a href="{p['link_url']}" class="proj-link" target="_blank">{p['link_text']} →</a>
        </div>
    """
    projects_html += f'<div class="proj-card">{content}</div>\n'

icons_markup = icon_html("icons")
if not icons_markup:
    icons_markup = '<p style="color:#64748b;font-size:0.9rem;">Add your tech icons to the <code>icons/</code> folder.</p>'

# ─────────────────────────────────────────────
# STRIP STREAMLIT CHROME
# ─────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FULL PAGE HTML
# ─────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
:root{{
  --bg:#080d18;--bg2:#0d1525;--bg3:#111e33;
  --border:#1a2a45;--border2:#243552;
  --text:#e2e8f0;--muted:#64748b;
  --accent:#3b82f6;--accent2:#60a5fa;
  --white:#f8fafc;--card:#0d1525;
  --max:1080px;--radius:14px;
}}
html{{scroll-behavior:smooth;}}
body{{
  font-family:'Inter',sans-serif;
  background:var(--bg);color:var(--text);
  line-height:1.6;-webkit-font-smoothing:antialiased;
}}

/* NAV */
nav{{
  position:sticky;top:0;z-index:100;
  background:rgba(8,13,24,0.9);backdrop-filter:blur(16px);
  border-bottom:1px solid var(--border);
  padding:0 3rem;height:64px;
  display:flex;align-items:center;justify-content:space-between;
}}
.nav-logo{{
  font-family:'Playfair Display',serif;
  font-size:1.25rem;color:var(--white);font-weight:700;
}}
.nav-links{{display:flex;gap:2.5rem;list-style:none;}}
.nav-links a{{
  color:var(--muted);text-decoration:none;
  font-size:0.8rem;font-weight:500;
  letter-spacing:0.1em;text-transform:uppercase;
  transition:color 0.2s;
}}
.nav-links a:hover{{color:var(--accent2);}}

/* WRAPPER */
.page{{width:100%;}}
section{{padding:5rem 3rem;max-width:var(--max);margin:0 auto;}}
.divider{{width:100%;height:1px;background:linear-gradient(90deg,transparent,var(--border2),transparent);}}

.section-label{{
  font-size:0.72rem;font-weight:600;
  letter-spacing:0.18em;text-transform:uppercase;
  color:var(--accent);margin-bottom:0.6rem;display:block;
}}
.section-title{{
  font-family:'Playfair Display',serif;
  font-size:2.4rem;font-weight:700;
  color:var(--white);line-height:1.15;margin-bottom:0.75rem;
}}
.section-sub{{
  color:var(--muted);font-size:0.975rem;
  max-width:520px;margin-bottom:3rem;line-height:1.75;
}}

/* HERO */
#hero{{
  padding-top:6rem;padding-bottom:6rem;
  max-width:var(--max);margin:0 auto;
  display:flex;align-items:center;gap:5rem;
  padding-left:3rem;padding-right:3rem;
}}
.profile-img{{
  width:240px;height:240px;border-radius:50%;
  object-fit:cover;flex-shrink:0;
  border:3px solid var(--border2);
  box-shadow:0 0 0 10px rgba(59,130,246,0.06),0 24px 60px rgba(0,0,0,0.55);
  display:block;
}}
.profile-placeholder{{
  width:240px;height:240px;border-radius:50%;flex-shrink:0;
  background:var(--bg3);border:3px solid var(--border2);
  display:flex;align-items:center;justify-content:center;
  font-size:4rem;
}}
.hero-eyebrow{{
  font-size:0.75rem;font-weight:600;
  letter-spacing:0.18em;text-transform:uppercase;
  color:var(--accent);margin-bottom:0.9rem;
}}
.hero-name{{
  font-family:'Playfair Display',serif;
  font-size:3.6rem;font-weight:700;color:var(--white);
  line-height:1.05;margin-bottom:0.5rem;letter-spacing:-1.5px;
}}
.hero-role{{
  font-size:1rem;color:var(--muted);
  font-weight:400;margin-bottom:1.75rem;
}}
.hero-bio{{
  font-size:0.975rem;color:#94a3b8;
  line-height:1.85;max-width:560px;margin-bottom:2rem;
}}
.hero-bio strong{{color:var(--text);font-weight:600;}}
.hero-btns{{display:flex;gap:0.75rem;flex-wrap:wrap;}}
.btn{{
  display:inline-flex;align-items:center;gap:0.4rem;
  padding:0.6rem 1.4rem;border-radius:8px;
  font-size:0.85rem;font-weight:600;
  text-decoration:none;transition:all 0.2s ease;
  letter-spacing:0.02em;cursor:pointer;border:none;
  font-family:'Inter',sans-serif;
}}
.btn-primary{{background:var(--accent);color:#fff;}}
.btn-primary:hover{{background:#2563eb;transform:translateY(-1px);box-shadow:0 8px 24px rgba(59,130,246,0.3);}}
.btn-ghost{{background:transparent;color:var(--text);border:1.5px solid var(--border2);}}
.btn-ghost:hover{{border-color:var(--accent);color:var(--accent2);transform:translateY(-1px);}}

/* SKILLS */
.icons-grid{{
  display:flex;flex-wrap:wrap;gap:1rem;
  margin-bottom:2.5rem;
}}
.icon-item{{
  display:flex;flex-direction:column;align-items:center;gap:0.5rem;
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:1rem;width:96px;
  transition:border-color 0.2s,transform 0.2s;
}}
.icon-item:hover{{border-color:var(--accent);transform:translateY(-3px);}}
.icon-item img{{width:42px;height:42px;object-fit:contain;}}
.icon-item span{{font-size:0.66rem;color:var(--muted);text-align:center;line-height:1.3;}}
.skills-pills{{display:flex;flex-wrap:wrap;gap:0.5rem;}}
.pill{{
  padding:0.38rem 0.9rem;border-radius:999px;
  font-size:0.775rem;font-weight:500;
  border:1px solid var(--border2);color:var(--muted);
  background:var(--card);letter-spacing:0.02em;
}}
.pill.core{{
  border-color:rgba(59,130,246,0.4);
  color:var(--accent2);background:rgba(59,130,246,0.07);
}}

/* PROJECTS */
.projects-list{{display:flex;flex-direction:column;gap:1.25rem;}}
.proj-card{{
  display:flex;align-items:center;gap:3rem;
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:2.25rem 2.5rem;
  transition:border-color 0.25s,transform 0.25s;
}}
.proj-card:hover{{border-color:var(--border2);transform:translateY(-2px);}}
.proj-content{{flex:1;min-width:0;}}
.proj-visual{{flex-shrink:0;width:240px;}}
.proj-img{{
  width:100%;height:165px;object-fit:cover;
  border-radius:10px;border:1px solid var(--border);display:block;
}}
.proj-img-placeholder{{
  width:100%;height:165px;background:var(--bg3);
  border:1px solid var(--border);border-radius:10px;
  display:flex;align-items:center;justify-content:center;
  font-size:2rem;color:var(--muted);
}}
.proj-tech{{
  font-size:0.7rem;font-weight:600;
  letter-spacing:0.12em;text-transform:uppercase;
  color:var(--accent);display:block;margin-bottom:0.5rem;
}}
.proj-title{{
  font-family:'Playfair Display',serif;
  font-size:1.25rem;font-weight:700;
  color:var(--white);margin-bottom:0.65rem;line-height:1.3;
}}
.proj-desc{{
  font-size:0.9rem;color:var(--muted);
  line-height:1.75;margin-bottom:1.1rem;
}}
.proj-link{{
  display:inline-flex;align-items:center;
  font-size:0.85rem;font-weight:600;color:var(--accent2);
  text-decoration:none;border-bottom:1px solid transparent;
  transition:border-color 0.2s;
}}
.proj-link:hover{{border-color:var(--accent2);}}

/* FACTS */
.facts-grid{{
  display:grid;grid-template-columns:repeat(3,1fr);gap:1.25rem;
}}
.fact-card{{
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:1.75rem;
  transition:border-color 0.2s;
}}
.fact-card:hover{{border-color:var(--border2);}}
.fact-emoji{{font-size:1.6rem;margin-bottom:0.875rem;display:block;}}
.fact-title{{font-size:0.975rem;font-weight:700;color:var(--white);margin-bottom:0.5rem;}}
.fact-text{{font-size:0.875rem;color:var(--muted);line-height:1.7;}}

/* CONTACT */
.contact-grid{{
  display:grid;grid-template-columns:1fr 1fr;
  gap:4rem;align-items:start;
}}
.contact-info h3{{
  font-family:'Playfair Display',serif;
  font-size:2rem;color:var(--white);margin-bottom:0.75rem;
}}
.contact-info p{{
  color:var(--muted);font-size:0.95rem;
  line-height:1.75;margin-bottom:2rem;
}}
.contact-links{{display:flex;flex-direction:column;gap:0.65rem;}}
.contact-link{{
  display:flex;align-items:center;gap:0.7rem;
  font-size:0.875rem;color:var(--muted);
  text-decoration:none;padding:0.7rem 1rem;
  border:1px solid var(--border);border-radius:10px;
  background:var(--card);transition:border-color 0.2s,color 0.2s;
}}
.contact-link:hover{{border-color:var(--accent);color:var(--accent2);}}
form{{display:flex;flex-direction:column;gap:0.8rem;}}
form input,form textarea{{
  width:100%;background:var(--card);
  border:1.5px solid var(--border);color:var(--text);
  border-radius:10px;padding:0.7rem 1rem;
  font-size:0.9rem;font-family:'Inter',sans-serif;
  outline:none;transition:border-color 0.2s;resize:none;
}}
form input::placeholder,form textarea::placeholder{{color:var(--muted);}}
form input:focus,form textarea:focus{{border-color:var(--accent);}}
form textarea{{height:130px;}}
form button{{
  align-self:flex-start;background:var(--accent);
  color:#fff;border:none;padding:0.7rem 1.75rem;
  border-radius:8px;font-size:0.875rem;font-weight:600;
  font-family:'Inter',sans-serif;cursor:pointer;
  transition:background 0.2s,transform 0.2s;letter-spacing:0.02em;
}}
form button:hover{{background:#2563eb;transform:translateY(-1px);}}

/* FOOTER */
.site-footer{{
  border-top:1px solid var(--border);padding:2rem 3rem;
  text-align:center;color:var(--muted);font-size:0.8rem;
  letter-spacing:0.05em;
}}
</style>
</head>
<body>

<nav>
  <span class="nav-logo">PG</span>
  <ul class="nav-links">
    <li><a href="#about">About</a></li>
    <li><a href="#skills">Skills</a></li>
    <li><a href="#projects">Projects</a></li>
    <li><a href="#contact">Contact</a></li>
  </ul>
</nav>

<div class="page">

<!-- HERO -->
<div id="about">
<div id="hero">
  <div>{profile_tag}</div>
  <div class="hero-text-col">
    <p class="hero-eyebrow">AI / ML Engineer &nbsp;·&nbsp; Munich, Germany</p>
    <h1 class="hero-name">Prateek<br>Ghorawat</h1>
    <p class="hero-role">Process Mining · Generative AI · Data & ML Systems</p>
    <p class="hero-bio">
      I build production-grade ML systems, LLM-driven workflows, and multi-agent AI solutions
      to tackle complex business and operational challenges.<br><br>
      At <strong>BMW Group</strong>, I developed predictive models and agentic AI for process
      and trade analytics — achieving <strong>95% accuracy</strong> in risk detection and automating
      insight generation via LLMs. At <strong>Celonis</strong>, I built large-scale data pipelines
      and ML models that reduced decision latency and drove measurable efficiency gains.<br><br>
      My focus: <strong>data → models → business outcomes</strong>.
    </p>
    <div class="hero-btns">
      <a href="https://drive.google.com/file/d/1S9gi4HjT3U3qB-kxKSPKuRCU6flZ4u7F/view?usp=sharing"
         class="btn btn-primary" target="_blank">📄 Resume</a>
      <a href="https://linkedin.com/in/prateek-ghorawat" class="btn btn-ghost" target="_blank">🔗 LinkedIn</a>
      <a href="https://github.com/prateekghorawat/" class="btn btn-ghost" target="_blank">🐙 GitHub</a>
    </div>
  </div>
</div>
</div>

<div class="divider"></div>

<!-- SKILLS -->
<div id="skills">
<section>
  <span class="section-label">Capabilities</span>
  <h2 class="section-title">Technical Skills</h2>
  <div class="icons-grid">{icons_markup}</div>
  <div class="skills-pills">
    <span class="pill core">⭐ AI Engineering & GenAI</span>
    <span class="pill core">⭐ Process Mining & Optimization</span>
    <span class="pill core">⭐ Data Engineering & ETL</span>
    <span class="pill core">⭐ Agentic AI & Frameworks</span>
    <span class="pill">CrewAI</span><span class="pill">AutoGen</span>
    <span class="pill">MCP</span><span class="pill">LangChain</span>
    <span class="pill">LangGraph</span><span class="pill">OpenAI GPT-4</span>
    <span class="pill">PyTorch</span><span class="pill">Scikit-learn</span>
    <span class="pill">SQL & ETL</span><span class="pill">MLflow & CI/CD</span>
    <span class="pill">BPMN & Simulation</span><span class="pill">Power BI</span>
    <span class="pill">NumPy · Pandas</span><span class="pill">Docker</span>
  </div>
</section>
</div>

<div class="divider"></div>

<!-- PROJECTS -->
<div id="projects">
<section>
  <span class="section-label">Work</span>
  <h2 class="section-title">Selected Projects</h2>
  <p class="section-sub">AI, ML, and data engineering — from industry to research.</p>
  <div class="projects-list">{projects_html}</div>
</section>
</div>

<div class="divider"></div>

<!-- FACTS -->
<section>
  <span class="section-label">About Me</span>
  <h2 class="section-title">A Bit More</h2>
  <div class="facts-grid">
    <div class="fact-card">
      <span class="fact-emoji">🤖</span>
      <div class="fact-title">Multi-Agent Whisperer</div>
      <p class="fact-text">I don't just build AI systems — I create AI teams. Using CrewAI and LangGraph, I orchestrate agents that collaborate, evaluate each other, and solve complex problems autonomously.</p>
    </div>
    <div class="fact-card">
      <span class="fact-emoji">🔍</span>
      <div class="fact-title">Process Mining Detective</div>
      <p class="fact-text">I teach machines to understand how automotive processes really work. Process mining and multi-agent simulations uncover inefficiencies that can save millions in operational costs.</p>
    </div>
    <div class="fact-card">
      <span class="fact-emoji">🌍</span>
      <div class="fact-title">Engineering Nomad</div>
      <p class="fact-text">From designing mechanical systems in India to building intelligent AI systems in Germany — from physical gears to neural networks, from Industry 3.0 to Industry 4.0.</p>
    </div>
  </div>
</section>

<div class="divider"></div>

<!-- CHATBOT -->
<section>
  <span class="section-label">AI Sidekick</span>
  <h2 class="section-title">Ask Me Anything</h2>
  <p class="section-sub">Want to know something specific? Ask my AI assistant.</p>
  <div style="border-radius:14px;overflow:hidden;border:1px solid var(--border);">
    <iframe
      src="https://pjpj4545-prateek-portfolio.hf.space"
      style="width:100%;height:660px;border:none;display:block;"
      allow="clipboard-write; encrypted-media; fullscreen"
      sandbox="allow-forms allow-scripts allow-same-origin">
    </iframe>
  </div>
</section>

<div class="divider"></div>

<!-- CONTACT -->
<div id="contact">
<section>
  <div class="contact-grid">
    <div class="contact-info">
      <span class="section-label">Get in Touch</span>
      <h3>Let's work<br>together.</h3>
      <p>Open to new opportunities, collaborations, or just a good conversation about AI and data.</p>
      <div class="contact-links">
        <a href="https://linkedin.com/in/prateek-ghorawat" class="contact-link" target="_blank">
          <span>🔗</span> linkedin.com/in/prateek-ghorawat
        </a>
        <a href="https://github.com/prateekghorawat" class="contact-link" target="_blank">
          <span>🐙</span> github.com/prateekghorawat
        </a>
        <a href="mailto:prateek.ghorawat1999@gmail.com" class="contact-link">
          <span>✉️</span> prateek.ghorawat1999@gmail.com
        </a>
      </div>
    </div>
    <div>
      <form action="https://formsubmit.co/prateek.ghorawat1999@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your Name" required>
        <input type="email" name="email" placeholder="Your Email" required>
        <textarea name="message" placeholder="Your Message" required></textarea>
        <button type="submit">Send Message</button>
      </form>
    </div>
  </div>
</section>
</div>

</div>

<div class="site-footer">© 2025 Prateek Ghorawat &nbsp;·&nbsp; Built with Streamlit</div>

</body>
</html>"""

components.html(html, height=99999, scrolling=False)
