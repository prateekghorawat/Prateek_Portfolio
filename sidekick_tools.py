from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from dotenv import load_dotenv
import os
import requests
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

load_dotenv(override=True)
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"
serper = GoogleSerperAPIWrapper()

# Your existing Playwright and other functions
async def playwright_tools():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright

def push(text: str):
    """Send a push notification to the user"""
    requests.post(pushover_url, data={"token": pushover_token, "user": pushover_user, "message": text})
    return "success"

def get_file_tools():
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()

# PORTFOLIO SPECIFIC TOOLS

def get_prateek_cv_info(query_type: str = "full"):
    """Get Prateek's CV information directly embedded - no parsing needed"""
    
    cv_data = {
        "basic_info": {
            "name": "Prateek Ghorawat",
            "email": "prateek.ghorawat1999@gmail.com", 
            "linkedin": "https://linkedin.com/in/prateek-ghorawat",
            "github": "https://github.com/prateekghorawat",
            "phone": "+49 17677872804",
            "portfolio": "https://prateek-ghorawat.super.site/",
            "location": "Munich, Germany",
            "cv_download": "Direct download available upon request"
        },
        
        "summary": "Skilled in AI, transforming complex data into actionable insights with precision and efficiency passionate about building intelligent systems that solve real-world challenges.",
        
        "current_status": {
            "role": "AI Engineer (Working Student) – Process Mining",
            "company": "BMW AG, Munich",
            "duration": "May 2025 – Present",
            "availability": "Open to new opportunities and collaborations"
        },
        
        "skills": {
            "process_intelligence": ["Process Mining", "Event Simulation", "Multi-Agent Systems", "Celonis", "BPMN", "Business Process"],
            "agentic_ai": ["CrewAI", "LangChain", "LangGraph", "MCP", "Agent Orchestration", "AutoGen", "Prompt Engineering"],
            "programming": ["Python", "C++", "SQL", "Pandas", "NumPy", "Flask", "GitHub", "CI/CD (GitHub Actions)"],
            "ml_analytics": ["TensorFlow", "PyTorch", "Scikit-Learn", "Series Forecasting", "Statistical Analysis", "Feature Engineering"],
            "data_engineering": ["ETL Pipelines", "AWS Glue", "Snowflake", "Databricks", "PostgreSQL", "Data Warehouse / Lakehouse"],
            "business_analytics": ["Business Intelligence", "KPI Development", "Process Optimization", "Performance Metrics", "ROI Analysis"],
            "cloud": ["AWS (SageMaker, EC2, Lambda, Redshift)", "Docker", "MLflow"],
            "visualization": ["Streamlit", "Plotly", "Matplotlib", "Seaborn", "Power BI", "Real-time Monitoring", "Signavio"]
        },
        
        "experience": [
            {
                "company": "BMW AG, Munich",
                "role": "AI Engineer (Working Student) – Process Mining",
                "duration": "May 2025 – Present",
                "type": "Current Position",
                "achievements": [
                    "Built and monitored ML/GenAI workflows with MLflow for experiment tracking and model versioning, deploying models on AWS EC2 to support business scenario testing",
                    "Working on investigation of Orchestration engines using Celonis OE, UiPath and N8N",
                    "Developed low-latency ETL and forecasting pipelines (SQL, Python, Snowflake, BigQuery), improving forecast accuracy from 78% to 92%",
                    "Working on process simulation multi-agent workflows with CrewAI, and LangChain, and dashboards for live process visibility and KPI monitoring in order to answer 'WHAT if scenarios'"
                ]
            },
            {
                "company": "BMW AG, Munich", 
                "role": "Master Thesis (Process AI) - Engineering ML & GenAI for Business Optimization",
                "duration": "October 2024 – March 2025",
                "type": "Thesis Project",
                "achievements": [
                    "Built Random Forest and CatBoost models to estimate customs duties (95% accuracy) and deployed via Flask API",
                    "Leveraged GPT, LLaMA, LangChain, and LangGraph to analyse cost-saving failures and implement an LLM-based judgment module for BI insights",
                    "Designed and A/B tested RAG pipelines, QLoRA-fine-tuned models, and agent-based systems to optimize performance and relevance"
                ]
            },
            {
                "company": "BMW AG, Munich",
                "role": "Machine Learning Automation Intern – Foreign Trade", 
                "duration": "April 2024 – October 2024",
                "type": "Internship",
                "achievements": [
                    "Automated ML/DL solutions for the customs department, incorporating FTA and preferential-rate regulations to streamline tariff calculations",
                    "Engineered and fine-tuned CO₂ emissions and price-forecast models, managing ETL, APIs, and data modeling to cut analyst workload by 40%",
                    "Integrated LLMs (LangChain) for business-workflow automation and supported POC-to-tender project management across cross-functional teams"
                ]
            },
            {
                "company": "Lacritz Ai",
                "role": "Machine Learning Intern",
                "duration": "May 2023 – July 2023", 
                "type": "Internship",
                "achievements": [
                    "Implemented NER and intent recognition for the company chatbot using fine-tuned BERT and NLP techniques",
                    "Applied transfer learning, GANs, and computer vision for document filtering and classification",
                    "Integrated solutions via APIs and AWS (Lambda, SageMaker, Bedrock) to connect customer queries to automated responses"
                ]
            },
            {
                "company": "Johnson Controls",
                "role": "Graduate Engineer Trainee",
                "duration": "October 2021 – August 2022",
                "type": "Full-time",
                "achievements": [
                    "Project Management and development of Automated tool for Quotation Creation using industrial engineering",
                    "Design of Control Systems and MEP services as well as CES development(logistics) and POC's with Simulation"
                ]
            }
        ],
        
        "education": [
            {
                "institution": "Hochschule Schmalkalden",
                "degree": "M.Eng in Mechatronics and Robotics",
                "duration": "October 2022 - Present",
                "status": "Current",
                "coursework": ["Artificial Intelligence", "Robotic Vision", "Computational Science", "Statistics", "Image Processing", "Project Management"]
            },
            {
                "institution": "Jain University - Bangalore, India",
                "degree": "B-Tech in Mechanical Engineering", 
                "duration": "August 2017 - June 2021",
                "status": "Completed",
                "coursework": ["Machine Design", "Programming languages", "Applied Mathematics", "Economics", "Inventory design"]
            }
        ],
        
        "projects": [
            {
                "name": "Advanced Object Detection and Tracking",
                "description": "Key points on the Face are detected and after successful implementation, Location of Fovea in the Eye has been Detected and analysed, which can be used to Identify changes in vision of retina. (IOU metrics achieved: 67%). Utilised the models generated in this process and with tweaking implemented them for Visual Search of required class. (Accuracy achieved: 87%)",
                "skills": ["CNN", "YOLO", "Image pre-processing", "Optimization", "Image Segmentation", "CUDA"],
                "code_link": "Available upon request",
                "status": "Completed"
            },
            {
                "name": "Multilingual Brochure Generation System", 
                "description": "Developed a dynamic system that scrapes company websites, extracts key information, and generates engaging, humorous Brochures. Integrated web scraping (BeautifulSoup) to analyze landing pages to transform raw data into structured, marketing-friendly content with retained context and humor. Implemented real-time streaming responses for interactive brochure generation and multilingual translation.",
                "skills": ["LLMs", "API Integration", "MLFlow", "Jenkins", "Prompt Engineering", "LLMOps", "MCP"],
                "code_link": "Available upon request",
                "status": "Completed"
            },
            {
                "name": "Sales Analytics Data Warehouse Modernization",
                "description": "Designed a Bronze–Silver–Gold data warehouse with star schema for consolidated sales insights. Automated ETL via T-SQL and SSIS, handling incremental loads, deduplication, and SCDs. Improved performance with partitioning and indexing, cutting query times by 50%. Enforced data quality and enabled self-service Power BI reporting with documented data models.",
                "skills": ["SQL Server", "SSIS", "Azure Data Factory", "T-SQL", "Power BI", "Azure SQL Database"],
                "code_link": "Available upon request",
                "status": "Completed"
            }
        ],
        
        "languages": {
            "English": "C1 - Professional working proficiency",
            "German": "A2 - Elementary proficiency", 
            "Hindi": "C2 - Native proficiency"
        },
        
        "certifications": [
            "Machine Learning Specialization",
            "Data Science BootCamp",
            "TensorFlow Developer",
            "Pytorch Developer", 
            "Agentic AI Engineering Course",
            "SQL and Databases Bootcamp"
        ],
        
        "key_achievements": [
            "Improved forecast accuracy from 78% to 92% using ML pipelines",
            "Built Random Forest and CatBoost models with 95% accuracy",
            "Cut analyst workload by 40% through ML automation",
            "Reduced query times by 50% through data warehouse optimization"
        ]
    }
    
    if query_type == "full":
        return cv_data
    elif query_type in cv_data:
        return cv_data[query_type]
    else:
        return f"Available sections: {', '.join(cv_data.keys())}"

def get_github_info(username: str = "prateekghorawat"):
    """Fetch GitHub profile and repository information"""
    try:
        # Get profile info
        profile_url = f"https://api.github.com/users/{username}"
        profile_response = requests.get(profile_url)
        
        if profile_response.status_code != 200:
            return "GitHub profile not found or API error. Please check the username."
        
        profile_data = profile_response.json()
        
        # Get repositories
        repos_url = f"https://api.github.com/users/{username}/repos"
        repos_response = requests.get(repos_url)
        repos_data = repos_response.json()
        
        return {
            "profile": {
                "name": profile_data.get("name"),
                "bio": profile_data.get("bio"),
                "public_repos": profile_data.get("public_repos"),
                "followers": profile_data.get("followers"),
                "following": profile_data.get("following"),
                "company": profile_data.get("company"),
                "location": profile_data.get("location"),
                "blog": profile_data.get("blog"),
                "created_at": profile_data.get("created_at")
            },
            "top_repos": [
                {
                    "name": repo["name"],
                    "description": repo["description"],
                    "language": repo["language"],
                    "stars": repo["stargazers_count"],
                    "forks": repo["forks_count"],
                    "url": repo["html_url"],
                    "updated_at": repo["updated_at"]
                }
                for repo in sorted(repos_data, key=lambda x: x["stargazers_count"], reverse=True)[:5]
                if isinstance(repos_data, list)
            ]
        }
    except Exception as e:
        return f"Error fetching GitHub data: {str(e)}"

def get_contact_info():
    """Get contact and availability information"""
    return {
        "contact": {
            "email": "prateek.ghorawat1999@gmail.com",
            "linkedin": "https://linkedin.com/in/prateek-ghorawat",
            "github": "https://github.com/prateekghorawat",
            "portfolio": "https://prateek-ghorawat.super.site/",
            "phone": "+49 17677872804",
            "location": "Munich, Germany"
        },
        "availability": "Open to new opportunities and collaborations",
        "preferred_contact": "Email or LinkedIn for professional inquiries",
        "timezone": "Central European Time (CET)",
        "response_time": "Usually responds within 24 hours"
    }

def get_project_details(project_name: str = "all"):
    """Get detailed information about specific projects"""
    cv_data = get_prateek_cv_info("projects")
    
    if project_name == "all":
        return cv_data
    
    for project in cv_data:
        if project_name.lower() in project["name"].lower():
            return project
    
    return f"Project not found. Available projects: {', '.join([p['name'] for p in cv_data])}"

def get_work_experience_details(company: str = "all"):
    """Get detailed work experience information"""
    cv_data = get_prateek_cv_info("experience")
    
    if company == "all":
        return cv_data
    
    for exp in cv_data:
        if company.lower() in exp["company"].lower():
            return exp
    
    return f"Company not found. Available companies: {', '.join([e['company'] for e in cv_data])}"

def answer_as_prateek(query: str):
    """Answer questions as if Prateek is personally responding"""
    cv_info = get_prateek_cv_info("full")
    
    # Create context for personal responses
    personal_context = f"""
    I am Prateek Ghorawat, currently working as an AI Engineer at BMW AG in Munich, specializing in Process Mining and Agentic AI.
    
    Key Facts About Me:
    - Current Role: {cv_info['current_status']['role']} at {cv_info['current_status']['company']}
    - Location: {cv_info['basic_info']['location']}
    - Expertise: AI, Machine Learning, Process Intelligence, Agentic AI Frameworks
    - Recent Achievement: Improved forecast accuracy from 78% to 92% using ML pipelines
    - Education: Currently pursuing M.Eng in Mechatronics and Robotics
    - Contact: {cv_info['basic_info']['email']}
    
    I'm passionate about building intelligent systems that solve real-world challenges, particularly in the automotive industry.
    """
    
    return {
        "context": personal_context,
        "query": query,
        "instruction": "Respond as Prateek would personally answer this question, using first person perspective and conversational tone."
    }

async def portfolio_tools():
    """Get all portfolio-specific tools"""
    
    # CV Information Tool (comprehensive)
    cv_tool = Tool(
        name="get_cv_info",
        func=get_prateek_cv_info,
        description="Get Prateek's comprehensive CV information including experience, skills, education, projects, certifications, and contact details. Use 'full' for complete info or specify section like 'skills', 'experience', 'projects', etc."
    )
    
    # GitHub Tool  
    github_tool = Tool(
        name="get_github_info",
        func=get_github_info,
        description="Get Prateek's GitHub profile information, repositories, and project details. Requires GitHub username."
    )
    
    # Contact Information Tool
    contact_tool = Tool(
        name="get_contact_info",
        func=get_contact_info,
        description="Get Prateek's contact information, availability, and preferred communication methods"
    )
    
    # Project Details Tool
    project_tool = Tool(
        name="get_project_details",
        func=get_project_details,
        description="Get detailed information about Prateek's specific projects. Use 'all' for all projects or specify project name"
    )
    
    # Work Experience Tool
    experience_tool = Tool(
        name="get_work_experience",
        func=get_work_experience_details,
        description="Get detailed work experience information. Use 'all' for all experience or specify company name like 'BMW' or 'Lacritz'"
    )
    
    # Personal Response Tool
    personal_tool = Tool(
        name="answer_as_prateek",
        func=answer_as_prateek,
        description="Get context and instructions for answering questions as if Prateek is personally responding"
    )
    
    return [cv_tool, github_tool, contact_tool, project_tool, experience_tool, personal_tool]

async def other_tools():
    """Get all tools including existing ones and portfolio tools"""
    
    # Existing tools
    push_tool = Tool(
        name="send_push_notification", 
        func=push, 
        description="Use this tool when you want to send a push notification"
    )
    
    file_tools = get_file_tools()

    tool_search = Tool(
        name="search",
        func=serper.run,
        description="Use this tool when you want to get the results of an online web search"
    )

    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    python_repl = PythonREPLTool()
    
    # Get portfolio tools
    portfolio_tools_list = await portfolio_tools()
    
    # Combine all tools
    all_tools = file_tools + [push_tool, tool_search, python_repl, wiki_tool] + portfolio_tools_list
    
    return all_tools

# Additional utility function for evaluator integration
def get_prateek_context_for_evaluator():
    """Get formatted context specifically for the evaluator function"""
    cv_info = get_prateek_cv_info("full")
    
    context = f"""
    PORTFOLIO OWNER INFORMATION:
    
    Name: {cv_info['basic_info']['name']}
    Current Role: {cv_info['current_status']['role']} at {cv_info['current_status']['company']}
    Location: {cv_info['basic_info']['location']}
    
    KEY SKILLS:
    - Agentic AI: {', '.join(cv_info['skills']['agentic_ai'])}
    - ML/Analytics: {', '.join(cv_info['skills']['ml_analytics'])}
    - Programming: {', '.join(cv_info['skills']['programming'])}
    
    RECENT ACHIEVEMENTS:
    {' | '.join(cv_info['key_achievements'])}
    
    CONTACT:
    Email: {cv_info['basic_info']['email']}
    LinkedIn: {cv_info['basic_info']['linkedin']}
    GitHub: {cv_info['basic_info']['github']}
    Portfolio: {cv_info['basic_info']['portfolio']}
    
    INSTRUCTION: Answer all questions as if you are Prateek personally responding. Use first person perspective, be conversational but professional, and provide specific details from the experience and projects when relevant.
    """
    
    return context

