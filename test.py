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
# Create two columns: image on left, text on right
col_image, col_text = st.columns([1, 2])

with col_image:
    # Inject CSS to add top margin to the image container
    st.markdown(
        """
        <style>
        .img-down-wrapper {
            margin-top: 2rem;  /* adjust this value as needed */
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
        # Wrap the image in the div
        st.markdown('<div class="img-down-wrapper">', unsafe_allow_html=True)
        st.image(
            img,
            use_container_width=False,
            width=400,  # increased size
            output_format="PNG"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Profile image not found. Place your image at `assets/profile.png`.")


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
    st.markdown('<p class="hero-subtitle">Engineering · AI · Data · Finance · Analytics</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-text">'
        "Hi, I'm Prateek!<br>"
        "I’m an AI Engineer specializing in Process Mining, Machine Learning, GenAI, Data Engineering, "
        "and Business Intelligence. At BMW AG, I build ML/GenAI workflows, forecasting pipelines, "
        "and multi-agent process simulations to drive measurable impact. Passionate about leveraging AI "
        "to solve real-world challenges in automotive, finance, and operations domains."
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<a href="https://drive.google.com/file/d/1S9gi4HjT3U3qB-kxKSPKuRCU6flZ4u7F/view?usp=sharing" class="btn btn-resume" target="_blank">View Resume</a>'
        '<a href="https://linkedin.com/in/prateek-ghorawat" class="btn btn-linkedin" target="_blank">LinkedIn</a>'
        '<a href="https://github.com/prateekghorawat/" class="btn btn-github" target="_blank">GitHub</a>',
        unsafe_allow_html=True
    )

st.markdown("---")



# ----- Skills -----
st.markdown("## Technical & Professional Skills", unsafe_allow_html=True)

# 1. Dynamically load all icon files from a directory
icon_dir = "icons"
icon_files = [
    os.path.join(icon_dir, f)
    for f in os.listdir(icon_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
]

# 2. Determine grid size (e.g., 5 columns)
cols_per_row = 5
rows = [
    icon_files[i : i + cols_per_row]
    for i in range(0, len(icon_files), cols_per_row)
]

# 3. Render each row
for row in rows:
    cols = st.columns(cols_per_row)
    for idx, icon_path in enumerate(row):
        with cols[idx]:
            try:
                img = Image.open(icon_path)
                st.image(img, width=200)
            except Exception:
                st.write("⚠️ Unable to load")

# 4. Skill categories beneath


# Custom CSS for centered and larger font
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

# Centered skill categories
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

# Additional Skills (left-aligned)
st.markdown("**Additional Skills:**")
st.markdown(
    """
- **Agentic AI:** CrewAI, AutoGen, MCP (Model Context Protocol)  
- **Data Stack:** SQL, Pandas, NumPy, ETL Pipelines, Data Warehousing  
- **MLOps:** MLflow, CI/CD, Model Deployment, Experiment Tracking  
- **Process Intelligence:** BPMN, Event Simulation, Multi-Agent Systems  
    """,
    unsafe_allow_html=True
)

st.markdown("---")
# ----- Projects -----
st.subheader("Projects")

projects = [
    {
        "title": "Advanced Image Generation with Wasserstein GANs",
        "technologies": "PyTorch/TensorFlow | WGAN-GP | CelebA dataset | NumPy | Pandas | Matplotlib | Gradient penalty optimization | Advanced loss functions | Image preprocessing",
        "description": (
            "Built a sophisticated deep learning system that generates realistic human"
            "portraits by training on over 200,000 celebrity face images from the CelebA dataset. "
            "Addressed critical limitations of traditional GANs including training instability "
            "and mode collapse via advanced Wasserstein loss implementation."
        ),
        "link_text": "Project Summary",
        "link_url": "https://github.com/prateekghorawat/Generative-AI/tree/main/CelebA-WGAN-Exploring-Advanced-Image-Generation-with-WGAN-and-CelebA-Dataset-main",
        "image_path": "projects_images/project_1.png"
    },
    {
        "title": "AI Agent Evaluation System",
        "technologies": "LangChain | LangGraph | OpenAI GPT-4 | Playwright | Gradio | Pydantic | Python Async | State Management | Browser Automation | MCP",
        "description": (
            "Developed an intelligent evaluation system using LangGraph state management to orchestrate worker and evaluator agents. The system features automated browser interactions through Playwright, structured output validation with Pydantic, and a live web interface for real-time monitoring and interaction."
        ),
        "link_text": "View Evaluator",
        "link_url": "https://huggingface.co/spaces/pjpj4545/Prateek_portfolio",
        "image_path": "projects_images/project_2.png"
    },
    {
        "title": "Sales Analytics Data Warehouse Modernization ",
        "technologies": "SQL Server | SSIS | T-SQL | ETL Pipelines | Star Schema | Medallion Architecture | Data Modeling | Business Intelligence | Performance Optimization",
        "description": (
            "Developed a modern data warehouse using SQL Server implementing industry-standard Medallion Architecture for systematic data quality improvement. The system processes raw data from CSV files through three distinct layers: Bronze (raw ingestion), Silver (cleansed/validated), and Gold (business-ready analytics) with optimized star schema design for reporting."
        ),
        "link_text": "View Project",
        "link_url": "https://github.com/prateekghorawat/Data-Warehousing-/tree/main",
        "image_path": "projects_images/project_3.png"
    },
    {
        "title": "Advanced Fovea Detection & Eye Tracking System",
        "technologies": "PyTorch | OpenCV | YOLO | CNN | CUDA | Image Segmentation | Medical Imaging | Transfer Learning | Computer Vision | Deep Learning",
        "description": (
            "Developed an advanced medical imaging solution combining CNN architectures with YOLO object detection to identify and track the fovea (central vision region) in retinal photographs. The system addresses critical challenges in ophthalmology by enabling automated detection of retinal changes that could indicate vision deterioration or eye diseases."
        ),
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

# Inject CSS for centered fact items
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
    .fact-emoji {
        font-size: 2rem;
        vertical-align: middle;
        margin-right: 0.5rem;
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
st.markdown("#### Want to know Something specific about me, please ask my sidekick", unsafe_allow_html=True)

components.html(
    """
    <iframe
      src="https://pjpj4545-prateek-portfolio.hf.space"
      style="width: 100%; height: 700px; border: none;"
      allow="clipboard-write; encrypted-media; fullscreen"
      sandbox="allow-forms allow-scripts allow-same-origin"
    ></iframe>
    """,
    height=700,
)
# ----- Facts About Me -----

# Centered facts
st.markdown(
    """
    <div class="facts-container">
        <div class="fact-item">
            <div class="fact-title">🤖 Multi-Agent Whisperer</div>
            <div class="fact-text">
                I don't just build AI systems I create AI teams! While most people work with single AI models,
                I orchestrate entire crews of AI agents using CrewAI and LangGraph that collaborate, evaluate each other,
                and solve complex business problems autonomously.
            </div>
        </div>
        <div class="fact-item">
            <div class="fact-title">🔍 Automotive Process Mining Detective</div>
            <div class="fact-text">
                I'm the person who teaches machines to understand how automotive business processes actually work behind the scenes.
                Using process mining and multi-agent simulations, I help discover hidden inefficiencies and answer “what-if” scenarios
                that can save millions in operational costs.
            </div>
        </div>
        <div class="fact-item">
            <div class="fact-title">🌍 Engineering Nomad</div>
            <div class="fact-text">
                I went from designing mechanical systems in India to building intelligent AI systems in Germany proving that curiosity
                and continuous learning can take you from physical gears to neural networks, literally from Industry 3.0 to Industry 4.0!
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ----- Contact -----
st.subheader("Get in Touch")
st.write("Always open to new opportunities or collaborations.")
contact_form = """
<form action="https://formsubmit.co/prateek.ghorawat1999@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
    <input type="text" name="name" placeholder="Your Name" required style="width:100%;padding:0.5rem;margin-bottom:0.5rem;">
    <input type="email" name="email" placeholder="Your Email" required style="width:100%;padding:0.5rem;margin-bottom:0.5rem;">
    <textarea name="message" placeholder="Your Message" required style="width:100%;padding:0.5rem;margin-bottom:0.5rem;height:100px;"></textarea>
    <button type="submit" style="padding:0.6rem 1.2rem;background-color:#334155;color:#fff;border:none;border-radius:0.375rem;">Send</button>
</form>
"""
st.markdown(contact_form, unsafe_allow_html=True)




