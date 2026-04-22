import streamlit as st
from PIL import Image
import os
import streamlit.components.v1 as components

# ----- Page Configuration -----
st.set_page_config(
    page_title="Prateek Ghorawat – Portfolio",
    page_icon="📁",
    layout="wide"
)

# ----- Custom CSS -----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0a0f1e;
        color: #e2e8f0;
    }

    /* Remove default top padding */
    .reportview-container .main > div,
    .block-container {
        padding-top: 2rem !important;
        margin-top: 0rem;
    }

    /* ---- Hero ---- */
    .hero-name {
        font-family: 'DM Serif Display', serif;
        font-size: 4rem;
        font-weight: 400;
        color: #f8fafc;
        margin: 0 0 0.25rem 0;
        line-height: 1.1;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #38bdf8;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0 0 1.25rem 0;
    }
    .hero-text {
        font-size: 1.05rem;
        line-height: 1.8;
        color: #94a3b8;
        max-width: 680px;
    }
    .hero-text strong {
        color: #cbd5e1;
    }

    /* ---- Buttons ---- */
    .btn-row { margin-top: 1.5rem; }
    .btn {
        display: inline-block;
        margin: 0 0.75rem 0.5rem 0;
        padding: 0.55rem 1.4rem;
        font-size: 0.9rem;
        font-weight: 600;
        border-radius: 6px;
        text-decoration: none;
        color: #f1f5f9;
        background-color: #1e293b;
        border: 1.5px solid #334155;
        letter-spacing: 0.03em;
        transition: all 0.2s ease;
    }
    .btn:hover {
        background-color: #334155;
        border-color: #38bdf8;
        color: #38bdf8;
    }

    /* ---- Section Divider ---- */
    hr {
        border: none;
        border-top: 1px solid #1e293b;
        margin: 2.5rem 0;
    }

    /* ---- Section Headings ---- */
    h2, .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem !important;
        font-weight: 400 !important;
        color: #f8fafc !important;
        letter-spacing: -0.5px;
        margin-bottom: 1.5rem !important;
    }
    h3 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }

    /* ---- Skill Badges ---- */
    .skills-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin: 1.5rem 0;
    }
    .skill-badge {
        background: #1e293b;
        border: 1px solid #334155;
        color: #94a3b8;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .skill-badge.highlight {
        border-color: #0ea5e9;
        color: #38bdf8;
        background: #0c1a2e;
    }

    /* ---- Project Cards ---- */
    .project-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        transition: border-color 0.2s ease;
    }
    .project-card:hover {
        border-color: #334155;
    }
    .project-tech {
        font-size: 0.82rem;
        color: #38bdf8;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .project-desc {
        color: #94a3b8;
        line-height: 1.7;
        font-size: 1rem;
    }
    .project-link {
        display: inline-block;
        margin-top: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
        color: #38bdf8;
        text-decoration: none;
        border-bottom: 1px solid transparent;
        transition: border-color 0.2s;
    }
    .project-link:hover {
        border-color: #38bdf8;
    }

    /* ---- Facts Section ---- */
    .facts-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.25rem;
        margin-top: 1.5rem;
    }
    .fact-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
    }
    .fact-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.6rem;
    }
    .fact-card-text {
        font-size: 0.95rem;
        color: #64748b;
        line-height: 1.65;
    }

    /* ---- Contact Form ---- */
    .contact-form-wrapper input,
    .contact-form-wrapper textarea {
        width: 100%;
        background: #111827;
        border: 1.5px solid #1e293b;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 0.65rem 0.9rem;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
        font-family: 'DM Sans', sans-serif;
        outline: none;
        transition: border-color 0.2s;
        box-sizing: border-box;
    }
    .contact-form-wrapper input:focus,
    .contact-form-wrapper textarea:focus {
        border-color: #0ea5e9;
    }
    .contact-form-wrapper textarea {
        height: 130px;
        resize: vertical;
    }
    .contact-form-wrapper button {
        background: #0ea5e9;
        color: #fff;
        border: none;
        padding: 0.65rem 1.75rem;
        border-radius: 8px;
        font-size: 0.95rem;
        font-weight: 600;
        cursor: pointer;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 0.02em;
        transition: background 0.2s;
    }
    .contact-form-wrapper button:hover {
        background: #0284c7;
    }

    /* ---- Profile image wrapper ---- */
    .img-down-wrapper {
        margin-top: 1.5rem;
        display: flex;
        justify-content: center;
    }
    .img-down-wrapper img {
        border-radius: 50%;
        border: 3px solid #1e293b;
    }

    /* Streamlit image override */
    [data-testid="stImage"] > img {
        border-radius: 50% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════
col_image, col_text = st.columns([1, 2])

with col_image:
    try:
        img = Image.open("self_gen.png")
        st.markdown('<div class="img-down-wrapper">', unsafe_allow_html=True)
        st.image(img, use_container_width=False, width=380, output_format="PNG")
        st.markdown('</div>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Profile image not found. Place your image at `self_gen.png`.")

with col_text:
    st.markdown('<p class="hero-name">Prateek Ghorawat</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">AI / ML Engineer &nbsp;·&nbsp; Process Mining &nbsp;·&nbsp; Generative AI &nbsp;·&nbsp; Data Systems</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="hero-text">
        Hi, I'm Prateek — an <strong>AI / ML Engineer</strong> specializing in Process Mining, Machine Learning,
        Generative AI, and Data Engineering. I build production-grade ML systems, LLM-driven workflows,
        and multi-agent AI solutions to tackle complex business and operational challenges.<br><br>
        At <strong>BMW Group</strong>, I developed predictive models and agentic AI systems for process and
        trade analytics, achieving up to <strong>95% accuracy</strong> in risk detection and automating insight
        generation using LLMs. At <strong>Celonis</strong>, I worked at the intersection of process mining and AI —
        building large-scale data pipelines and ML models that reduced decision latency and drove measurable
        efficiency gains.<br><br>
        My focus: connecting <strong>data → models → business outcomes</strong> by combining ML, MLOps,
        and process intelligence into scalable AI systems that deliver real impact.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="btn-row">
            <a href="https://drive.google.com/file/d/1S9gi4HjT3U3qB-kxKSPKuRCU6flZ4u7F/view?usp=sharing"
               class="btn" target="_blank">📄 Resume</a>
            <a href="https://linkedin.com/in/prateek-ghorawat"
               class="btn" target="_blank">🔗 LinkedIn</a>
            <a href="https://github.com/prateekghorawat/"
               class="btn" target="_blank">🐙 GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ═══════════════════════════════════════════════════
# SKILLS SECTION
# ═══════════════════════════════════════════════════
st.markdown('<h2 class="section-title">Technical Skills</h2>', unsafe_allow_html=True)

icon_dir = "icons"
try:
    icon_files = [
        os.path.join(icon_dir, f)
        for f in sorted(os.listdir(icon_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
    ]
    cols_per_row = 5
    rows = [icon_files[i : i + cols_per_row] for i in range(0, len(icon_files), cols_per_row)]
    for row in rows:
        cols = st.columns(cols_per_row)
        for idx, icon_path in enumerate(row):
            with cols[idx]:
                try:
                    st.image(Image.open(icon_path), width=200)
                except Exception:
                    st.write("⚠️")
except FileNotFoundError:
    st.info("Add your icons to the `icons/` directory.")

# Skill category badges
st.markdown(
    """
    <div class="skills-badges">
        <span class="skill-badge highlight">⭐ AI Engineering & GenAI</span>
        <span class="skill-badge highlight">⭐ Process Mining & Optimization</span>
        <span class="skill-badge highlight">⭐ Data Engineering & ETL</span>
        <span class="skill-badge highlight">⭐ Agentic AI & Frameworks</span>
        <span class="skill-badge">CrewAI</span>
        <span class="skill-badge">AutoGen</span>
        <span class="skill-badge">MCP</span>
        <span class="skill-badge">LangChain</span>
        <span class="skill-badge">LangGraph</span>
        <span class="skill-badge">OpenAI / GPT-4</span>
        <span class="skill-badge">PyTorch</span>
        <span class="skill-badge">Scikit-learn</span>
        <span class="skill-badge">SQL & ETL</span>
        <span class="skill-badge">MLflow & CI/CD</span>
        <span class="skill-badge">BPMN & Simulation</span>
        <span class="skill-badge">Power BI</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ═══════════════════════════════════════════════════
# PROJECTS SECTION
# ═══════════════════════════════════════════════════
st.markdown('<h2 class="section-title">Selected Projects</h2>', unsafe_allow_html=True)

projects = [
    {
        "title": "Multi-Agent AI System for Process Intelligence",
        "technologies": "LangChain · LangGraph · MCP · Python · LLMs · Agentic AI",
        "description": (
            "Designed and deployed a multi-agent AI orchestration system at BMW Group that enables "
            "autonomous reasoning, coordination, and automation across complex business workflows. "
            "Integrated LangGraph for stateful agent control, with real-time monitoring interfaces "
            "and structured output validation."
        ),
        "link_url": "https://huggingface.co/spaces/pjpj4545/Prateek_portfolio",
        "link_text": "View Evaluator",
        "image_path": "projects_images/project_2.png"
    },
    {
        "title": "AI Prediction System for Trade & Risk Analytics",
        "technologies": "Scikit-learn · ML Pipelines · Feature Engineering · Python",
        "description": (
            "Built an end-to-end ML system achieving ~95% accuracy in anomaly and risk detection on "
            "large-scale operational datasets. Automated insight generation using LLM-based summarization, "
            "reducing analyst workload by 80%."
        ),
        "link_url": "https://github.com/prateekghorawat",
        "link_text": "View Project",
        "image_path": "projects_images/project_1.png"
    },
    {
        "title": "Advanced Image Generation with Wasserstein GANs",
        "technologies": "PyTorch · WGAN-GP · CelebA Dataset · NumPy · Matplotlib",
        "description": (
            "Built a sophisticated deep learning system generating realistic human portraits by training "
            "on 200,000+ celebrity face images (CelebA). Addressed training instability and mode collapse "
            "via Wasserstein loss with gradient penalty (WGAN-GP)."
        ),
        "link_url": "https://github.com/prateekghorawat/Generative-AI/tree/main/CelebA-WGAN-Exploring-Advanced-Image-Generation-with-WGAN-and-CelebA-Dataset-main",
        "link_text": "Project Summary",
        "image_path": "projects_images/project_1.png"
    },
    {
        "title": "Sales Analytics Data Warehouse Modernization",
        "technologies": "SQL Server · SSIS · T-SQL · Medallion Architecture · Star Schema · Power BI",
        "description": (
            "Architected a modern data warehouse using Medallion (Bronze → Silver → Gold) layers for "
            "systematic data quality improvement. Reduced query latency by 50% with optimized star "
            "schema design. Built end-to-end ETL pipelines from CSV ingestion to business-ready analytics."
        ),
        "link_url": "https://github.com/prateekghorawat/Data-Warehousing-/tree/main",
        "link_text": "View Project",
        "image_path": "projects_images/project_3.png"
    },
    {
        "title": "Advanced Fovea Detection & Eye Tracking System",
        "technologies": "PyTorch · OpenCV · YOLO · CNN · CUDA · Medical Imaging · Transfer Learning",
        "description": (
            "Developed a medical imaging solution combining CNN architectures with YOLO object detection "
            "to identify and track the fovea in retinal photographs — enabling automated detection of "
            "retinal changes indicative of vision deterioration or eye disease."
        ),
        "link_url": "https://github.com/prateekghorawat/Computer-Vision-Projects",
        "link_text": "View Project",
        "image_path": "projects_images/project_4.png"
    },
]

for proj in projects:
    col_text_col, col_img_col = st.columns([2, 1], gap="large")
    with col_text_col:
        st.markdown(
            f"""
            <div class="project-card">
                <div class="project-tech">{proj['technologies']}</div>
                <h3 style="margin:0 0 0.75rem 0;">{proj['title']}</h3>
                <p class="project-desc">{proj['description']}</p>
                <a href="{proj['link_url']}" class="project-link" target="_blank">
                    📄 {proj['link_text']} →
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_img_col:
        try:
            img = Image.open(proj["image_path"])
            st.image(img, use_container_width=True)
        except FileNotFoundError:
            st.markdown(
                '<div style="height:180px;background:#111827;border:1px solid #1e293b;'
                'border-radius:12px;display:flex;align-items:center;justify-content:center;'
                'color:#334155;font-size:2rem;">🔍</div>',
                unsafe_allow_html=True
            )

st.markdown("---")

# ═══════════════════════════════════════════════════
# FACTS / CHATBOT SECTION
# ═══════════════════════════════════════════════════
st.markdown('<h2 class="section-title">Facts About Me</h2>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="facts-grid">
        <div class="fact-card">
            <div class="fact-card-title">🤖 Multi-Agent Whisperer</div>
            <div class="fact-card-text">
                I don't just build AI systems — I create AI teams. While most engineers work with single
                models, I orchestrate entire crews of AI agents using CrewAI and LangGraph that collaborate,
                evaluate each other, and solve complex business problems autonomously.
            </div>
        </div>
        <div class="fact-card">
            <div class="fact-card-title">🔍 Automotive Process Mining Detective</div>
            <div class="fact-card-text">
                I teach machines to understand how automotive business processes actually work behind the
                scenes. Using process mining and multi-agent simulations, I uncover hidden inefficiencies
                and answer "what-if" scenarios that can save millions in operational costs.
            </div>
        </div>
        <div class="fact-card">
            <div class="fact-card-title">🌍 Engineering Nomad</div>
            <div class="fact-card-text">
                I went from designing mechanical systems in India to building intelligent AI systems in
                Germany — proving that curiosity and continuous learning can take you from physical gears
                to neural networks, literally from Industry 3.0 to Industry 4.0.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='margin-top:2rem;color:#64748b;font-size:1rem;'>"
    "💬 Want to know something specific about me? Ask my AI sidekick below:</p>",
    unsafe_allow_html=True
)

components.html(
    """
    <iframe
      src="https://pjpj4545-prateek-portfolio.hf.space"
      style="width: 100%; height: 700px; border: none; border-radius: 12px;"
      allow="clipboard-write; encrypted-media; fullscreen"
      sandbox="allow-forms allow-scripts allow-same-origin"
    ></iframe>
    """,
    height=720,
)

st.markdown("---")

# ═══════════════════════════════════════════════════
# CONTACT SECTION
# ═══════════════════════════════════════════════════
st.markdown('<h2 class="section-title">Get in Touch</h2>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#64748b;font-size:1rem;margin-bottom:1.5rem;'>"
    "Always open to new opportunities, collaborations, or just a good conversation about AI.</p>",
    unsafe_allow_html=True
)

contact_form = """
<div class="contact-form-wrapper">
<form action="https://formsubmit.co/prateek.ghorawat1999@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
    <input type="text" name="name" placeholder="Your Name" required>
    <input type="email" name="email" placeholder="Your Email" required>
    <textarea name="message" placeholder="Your Message" required></textarea>
    <button type="submit">Send Message</button>
</form>
</div>
"""
st.markdown(contact_form, unsafe_allow_html=True)
