import streamlit as st
from PIL import Image
import pandas as pd
import os
import streamlit.components.v1 as components
from streamlit_chat import message
import requests

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
    /* Remove default top padding so content starts at top */
    .reportview-container .main > div {
        padding-top: 0rem;
        margin-top: 0rem;
    }
    body {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .hero-name {
        font-size: 3.5rem;
        font-weight: 800;
        color: #fff;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.5rem;
        color: #94a3b8;
        margin: 0 0 1rem 0;
    }
    .hero-text {
        font-size: 1.125rem;
        line-height: 1.7;
        color: #cbd5e1;
    }
    .btn {
        display: inline-block;
        margin: 0.5rem 1rem 1rem 0;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 0.375rem;
        text-decoration: none;
        color: #fff;
    }
    .btn-resume { background-color: #1e293b; border: 2px solid #64748b;}
    .btn-linkedin { background-color: #1e293b; border: 2px solid #64748b; }
    .btn-github   { background-color: #1e293b; border: 2px solid #64748b; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----- Hero / About Section -----
col_image, col_text = st.columns([1, 2])

with col_image:
    st.markdown(
        """
        <style>
        .img-down-wrapper {
            margin-top: 2rem;
            display: flex;
            justify-content: center;
        }
        .img-down-wrapper img {
            border-radius: 50%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    try:
        img = Image.open("self_gen.png")
        st.markdown('<div class="img-down-wrapper">', unsafe_allow_html=True)
        st.image(
            img,
            use_container_width=False,
            width=400,
            output_format="PNG"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Profile image not found. Place your image at `self_gen.png`.")


with col_text:
    st.markdown(
        """
        <style>
        .hero-name {
            font-size: 4rem !important;   
            font-weight: 800;
            margin: -10;
        }
        .hero-subtitle {
            font-size: 1.75rem !important;  
            color: #94a3b8;
            margin: 0 0 1rem 0;
        }
        .hero-text {
            font-size: 1.125rem;
            line-height: 1.7;
            color: #cbd5e1;
        }
        .btn {
            display: inline-block;
            margin: 0.5rem 1rem 1rem 0;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 0.375rem;
            text-decoration: none;
            color: #fff;
        }
        .btn-resume { background-color: #1e293b; }
        .btn-linkedin { background-color: #1e293b; }
        .btn-github   { background-color: #1e293b;  }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<p class="hero-name">Prateek Ghorawat</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Process Mining · Generative AI · Data & ML Systems</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-text">'
        "I build production-grade ML systems, LLM-driven workflows, and multi-agent AI solutions "
        "to tackle complex business and operational challenges.<br><br>"
        "At <strong>BMW Group</strong>, I developed predictive models and agentic AI for process "
        "and trade analytics — achieving <strong>95% accuracy</strong> in risk detection and automating "
        "insight generation via LLMs. At <strong>Celonis</strong>, I built large-scale data pipelines "
        "and ML models that reduced decision latency and drove measurable efficiency gains.<br><br>"
        "My focus: <strong>data → models → business outcomes</strong>."
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<a href="https://drive.google.com/file/d/1ASoU2hOwNSdhIE5i0wobyO_P0XmIrelh/view?usp=sharing" class="btn btn-resume" target="_blank">View Resume</a>'
        '<a href="https://linkedin.com/in/prateek-ghorawat" class="btn btn-linkedin" target="_blank">LinkedIn</a>'
        '<a href="https://github.com/prateekghorawat/" class="btn btn-github" target="_blank">GitHub</a>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ----- Skills -----
st.markdown("## Technical & Professional Skills", unsafe_allow_html=True)

icon_dir = "icons"
if os.path.exists(icon_dir):
    icon_files = [
        os.path.join(icon_dir, f)
        for f in os.listdir(icon_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
    ]

    cols_per_row = 5
    rows = [
        icon_files[i : i + cols_per_row]
        for i in range(0, len(icon_files), cols_per_row)
    ]

    for row in rows:
        cols = st.columns(cols_per_row)
        for idx, icon_path in enumerate(row):
            with cols[idx]:
                try:
                    img = Image.open(icon_path)
                    st.image(img, width=200)
                except Exception:
                    st.write("⚠️ Unable to load")
else:
    st.info("Add your tech icons to the `icons/` folder to display them here.")

st.markdown(
    """
    <style>
    .skills-section {
        text-align: center;
        font-size: 1.2rem;
    }
    .skills-section h4 {
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .skills-categories {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin-bottom: 2rem;
    }
    .skills-categories div {
        margin: 0.5rem;
        font-size: 1.2rem;
    }
    .additional-skills {
        text-align: left;
        max-width: 600px;
        margin: 0 auto;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="skills-section">
        <div class="skills-categories">
            <div><strong>⭐ AI Engineering & GenAI</strong></div>
            <div><strong>⭐ Process Mining & Optimization</strong></div>
            <div><strong>⭐ Data Engineering & ETL</strong></div>
            <div><strong>⭐ Agentic AI & Frameworks</strong></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("**Core Technologies:**")
st.markdown(
    """
- **Agentic AI:** CrewAI, AutoGen, MCP, LangChain, LangGraph  
- **Data Stack:** SQL, Pandas, NumPy, ETL Pipelines, Data Warehousing, Power BI  
- **MLOps & AI:** PyTorch, Scikit-learn, OpenAI GPT-4, MLflow, CI/CD, Docker  
- **Process Intelligence:** BPMN, Event Simulation, Multi-Agent Systems  
    """,
    unsafe_allow_html=True
)

st.markdown("---")
# ----- Projects -----
st.subheader("Selected Projects")

projects = [
    {
        "title": "Multi-Agent AI System for Process Intelligence",
        "technologies": "LangChain · LangGraph · MCP · Python · LLMs",
        "description": "Designed and deployed a multi-agent AI orchestration system at BMW Group enabling autonomous reasoning, coordination, and automation across complex business workflows. Integrated LangGraph for stateful agent control with real-time monitoring and structured output validation.",
        "link_text": "View Demo",
        "link_url": "https://huggingface.co/spaces/pjpj4545/Prateek_portfolio",
        "image_path": "projects_images/project_2.png"
    },
    {
        "title": "AI Prediction System for Trade & Risk Analytics",
        "technologies": "Scikit-learn · ML Pipelines · Feature Engineering · Python",
        "description": "Built an end-to-end ML system achieving ~95% accuracy in anomaly and risk detection on large-scale operational datasets. Automated insight generation using LLM-based summarization, reducing analyst workload by 80%.",
        "link_text": "View Details",
        "link_url": "https://github.com/prateekghorawat/Prateek_Portfolio/blob/main/projects_images/i1.png",
        "image_path": "projects_images/i2.png"
    },
    {
        "title": "Sales Analytics Data Warehouse Modernization",
        "technologies": "UtilsForecast · NeuralForecast · TimeGPT · Chronos · Moirai · TimesFM · TimeLLM",
        "description": "Architected a modern data warehouse using Bronze → Silver → Gold layers for systematic data quality improvement. Reduced query latency by 50% with optimized star schema design and end-to-end ETL pipelines.",
        "link_text": "View Project",
        "link_url": "https://github.com/prateekghorawat/Data-Warehousing-/tree/main",
        "image_path": "projects_images/project_3.png"
    },
    {
        "title": "Time Series Forecasting: Blog Traffic Prediction with Multiple Models",
        "technologies": "SQL Server · SSIS · Medallion Architecture · Star Schema · Power BI",
        "description": "Built end-to-end time series forecasting pipeline comparing 8 models (SeasonalNaive, AutoARIMA, TimeGPT±exog, Chronos, Moirai, TimesFM, TimeLLM) on daily blog traffic data with weekly seasonality and exogenous features (holidays, new articles).",
        "link_text": "View Project",
        "link_url": "https://github.com/prateekghorawat/Data-Warehousing-/tree/main",
        "image_path": "projects_images/i3.png"
    },
    {
        "title": "Wasserstein GAN — Realistic Portrait Generation",
        "technologies": "PyTorch · WGAN-GP · CelebA Dataset · NumPy · Matplotlib",
        "description": "Built a deep learning system generating realistic human portraits by training on 200,000+ celebrity face images. Resolved training instability and mode collapse via Wasserstein loss with gradient penalty (WGAN-GP).",
        "link_text": "View Project",
        "link_url": "https://github.com/prateekghorawat/Generative-AI/tree/main/CelebA-WGAN-Exploring-Advanced-Image-Generation-with-WGAN-and-CelebA-Dataset-main",
        "image_path": "projects_images/project_1.png"
    },
    {
        "title": "Fovea Detection & Eye Tracking System",
        "technologies": "PyTorch · OpenCV · YOLO · CNN · CUDA · Medical Imaging",
        "description": "Medical imaging solution combining CNN architectures with YOLO object detection to identify and track the fovea in retinal photographs. Enables automated detection of retinal changes indicative of vision deterioration.",
        "link_text": "View Project",
        "link_url": "https://github.com/prateekghorawat/Computer-Vision-Projects",
        "image_path": "projects_images/project_4.png"
    },
]

for proj in projects:
    col_text, col_img = st.columns([2, 1], gap="large")
    with col_text:
        st.markdown(f"### {proj['title']}")
        st.markdown(f"*{proj['technologies']}*")
        st.write(proj["description"])
        st.markdown(f"[📄 {proj['link_text']}]({proj['link_url']})")
    with col_img:
        try:
            img = Image.open(proj["image_path"])
            st.image(img, use_container_width=True)
        except FileNotFoundError:
            st.write("🔍 Image not found")
    st.markdown("---")

st.markdown("<h2 style='text-align:center;'>Facts About Me</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .facts-container {
        text-align: center;
        margin: 2rem 0;
    }
    .fact-item {
        display: inline-block;
        text-align: left;
        max-width: 700px;
        margin-bottom: 2rem;
    }
    .fact-title {
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .fact-text {
        font-size: 1.125rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("#### Want to know something specific about me? Please ask my sidekick below!", unsafe_allow_html=True)

components.html(
    """
    <iframe
      src="https://pjpj4545-prateek-portfolio.hf.space"
      style="width: 100%; height: 700px; border: none; border-radius: 12px; margin-bottom: 2rem;"
      allow="clipboard-write; encrypted-media; fullscreen"
      sandbox="allow-forms allow-scripts allow-same-origin"
    ></iframe>
    """,
    height=740,
)

st.markdown(
    """
    <div class="facts-container">
        <div class="fact-item">
            <div class="fact-title">🤖 Multi-Agent Whisperer</div>
            <div class="fact-text">
                I don't just build AI systems — I create AI teams. Using CrewAI and LangGraph, I orchestrate agents that collaborate, evaluate each other, and solve complex problems autonomously.
            </div>
        </div>
        <div class="fact-item">
            <div class="fact-title">🔍 Process Mining Detective</div>
            <div class="fact-text">
                I teach machines to understand how automotive processes really work. Process mining and multi-agent simulations uncover inefficiencies that can save millions in operational costs.
            </div>
        </div>
        <div class="fact-item">
            <div class="fact-title">🌍 Engineering Nomad</div>
            <div class="fact-text">
                From designing mechanical systems in India to building intelligent AI systems in Germany — from physical gears to neural networks, from Industry 3.0 to Industry 4.0.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ----- Contact -----
st.subheader("Get in Touch")
st.write("Always open to new opportunities, collaborations, or just a good conversation about AI and data.")
contact_form = """
<form action="https://formsubmit.co/prateek.ghorawat1999@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
    <input type="text" name="name" placeholder="Your Name" required style="width:100%;padding:0.5rem;margin-bottom:0.5rem; border-radius: 5px; border: 1px solid #ccc;">
    <input type="email" name="email" placeholder="Your Email" required style="width:100%;padding:0.5rem;margin-bottom:0.5rem; border-radius: 5px; border: 1px solid #ccc;">
    <textarea name="message" placeholder="Your Message" required style="width:100%;padding:0.5rem;margin-bottom:0.5rem;height:100px; border-radius: 5px; border: 1px solid #ccc;"></textarea>
    <button type="submit" style="padding:0.6rem 1.2rem;background-color:#334155;color:#fff;border:none;border-radius:0.375rem;cursor:pointer;">Send Message</button>
</form>
"""
st.markdown(contact_form, unsafe_allow_html=True)
