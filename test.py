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

# ----- Hero Section -----
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
        st.image(img, width=400)
        st.markdown('</div>', unsafe_allow_html=True)
    except:
        st.warning("Profile image not found.")

with col_text:
    st.markdown('<p class="hero-name">Prateek Ghorawat</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI / ML Engineer · Process Mining · Generative AI · Data & AI Systems</p>', unsafe_allow_html=True)

    st.markdown(
        '<div class="hero-text">'
        "Hi, I'm Prateek!<br>"
        "I’m an AI / ML Engineer specializing in Process Mining, Machine Learning, Generative AI, "
        "and Data Engineering. I build production-grade ML systems, LLM-driven workflows, and "
        "multi-agent AI solutions to solve complex business and operational problems.<br><br>"
        "At BMW Group, I developed predictive models and agentic AI systems for process and trade analytics, "
        "achieving up to 95% accuracy in risk detection and automating insight generation using LLMs. "
        "At Celonis, I worked at the intersection of process mining and AI, building large-scale data pipelines "
        "and ML models on complex operational datasets to drive measurable business impact—reducing decision latency, "
        "improving efficiency, and enabling faster, data-driven decisions.<br><br>"
        "I focus on connecting data → models → business outcomes, combining ML, MLOps, and process intelligence "
        "to build scalable AI systems that deliver real impact."
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
st.markdown("## Technical & Professional Skills")

icon_dir = "icons"
icon_files = [os.path.join(icon_dir, f) for f in os.listdir(icon_dir) if f.endswith((".png",".jpg",".jpeg",".svg"))]

cols_per_row = 5
rows = [icon_files[i:i+cols_per_row] for i in range(0,len(icon_files),cols_per_row)]

for row in rows:
    cols = st.columns(cols_per_row)
    for idx, icon in enumerate(row):
        with cols[idx]:
            try:
                st.image(Image.open(icon), width=200)
            except:
                st.write("⚠️")

st.markdown(
"""
<div style="text-align:center;">
<strong>⭐ AI Engineering & GenAI</strong> &nbsp;&nbsp;
<strong>⭐ Process Mining & Optimization</strong> &nbsp;&nbsp;
<strong>⭐ Data Engineering & ETL</strong> &nbsp;&nbsp;
<strong>⭐ Agentic AI & Frameworks</strong>
</div>
""", unsafe_allow_html=True)

st.markdown("**Additional Skills:**")
st.markdown("""
- Agentic AI: CrewAI, AutoGen, MCP  
- Data Stack: SQL, Pandas, NumPy, ETL  
- MLOps: MLflow, CI/CD, Deployment  
- Process Intelligence: BPMN, Simulation  
""")

st.markdown("---")

# ----- Projects -----
st.subheader("Projects")

projects = [
    {
        "title": "Multi-Agent AI System for Process Intelligence (BMW)",
        "technologies": "LangChain | LangGraph | MCP | Python | LLMs",
        "description": "Built a multi-agent AI system enabling reasoning, coordination, and automation across complex business workflows.",
        "link_url": "https://github.com/prateekghorawat",
        "image_path": "projects_images/project_2.png"
    },
    {
        "title": "AI Prediction System for Trade & Risk",
        "technologies": "Scikit-learn | ML Pipelines | Feature Engineering",
        "description": "Developed ML system achieving ~95% accuracy for anomaly and risk detection in operational datasets.",
        "link_url": "https://github.com/prateekghorawat",
        "image_path": "projects_images/project_1.png"
    },
    {
        "title": "Sales Data Warehouse (Medallion Architecture)",
        "technologies": "SQL | ETL | Power BI",
        "description": "Built scalable data warehouse reducing query latency by 50% with optimized pipelines.",
        "link_url": "https://github.com/prateekghorawat/Data-Warehousing-/tree/main",
        "image_path": "projects_images/project_3.png"
    },
    {
        "title": "LLM Automation for KPI Analysis",
        "technologies": "LangChain | OpenAI | Python",
        "description": "Reduced manual KPI analysis effort by 80% using LLM-based automation.",
        "link_url": "https://huggingface.co/spaces/pjpj4545/Prateek_portfolio",
        "image_path": "projects_images/project_2.png"
    }
]

for proj in projects:
    col_text, col_img = st.columns([2,1])
    with col_text:
        st.markdown(f"### {proj['title']}")
        st.markdown(f"*{proj['technologies']}*")
        st.write(proj["description"])
        st.markdown(f"[View Project]({proj['link_url']})")
    with col_img:
        try:
            st.image(Image.open(proj["image_path"]))
        except:
            st.write("Image not found")
    st.markdown("---")

# ----- Contact -----
st.subheader("Get in Touch")
st.write("Always open to opportunities.")

st.markdown("""
<form action="https://formsubmit.co/prateek.ghorawat1999@gmail.com" method="POST">
<input type="text" name="name" placeholder="Your Name" required>
<input type="email" name="email" placeholder="Your Email" required>
<textarea name="message" placeholder="Message" required></textarea>
<button type="submit">Send</button>
</form>
""", unsafe_allow_html=True)
