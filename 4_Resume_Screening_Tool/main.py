from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_resume_match_score(resume_text: str, job_description_text: str) -> float:
    embeddings = model.encode([resume_text, job_description_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(similarity * 100, 2)

resumes = [
    """
    Experienced Software Developer with 5+ years in full stack development using Python, Django, ReactJS, and PostgreSQL.
    Led multiple teams delivering SaaS products in Agile environments. Proficient in REST APIs, CI/CD, and cloud platforms like AWS and GCP.
    Strong problem-solving skills and a passion for clean, scalable code.
    """,
    """
    Data Scientist with strong background in Statistics, Machine Learning, and Deep Learning. 
    Skilled in Python, TensorFlow, Scikit-Learn, and NLP. Worked on predictive modeling, sentiment analysis, and recommendation systems. 
    Published research papers and participated in Kaggle competitions with top rankings.
    """,
    """
    Front-End Developer with expertise in HTML, CSS, JavaScript, and React. 
    Built interactive web applications with focus on UI/UX design and performance optimization. 
    Familiar with version control (Git) and basic backend (Node.js, Express).
    """,
]

job_descriptions = [
    """
    We are hiring a Full Stack Developer with strong proficiency in Python, Django, and React. 
    Experience in building scalable web applications, RESTful APIs, and working with PostgreSQL databases is required. 
    Familiarity with cloud deployment (AWS/GCP) and Agile methodologies is a plus.
    """,
    """
    Looking for a Machine Learning Engineer with solid understanding of statistical models, deep learning frameworks (TensorFlow, PyTorch), and natural language processing. 
    Should have experience working with large datasets and developing end-to-end ML pipelines.
    """,
    """
    Seeking a talented Front-End Engineer to develop modern and responsive web applications. 
    Must be skilled in React, CSS, JavaScript, and have good knowledge of UI/UX principles.
    Experience with design systems and performance tuning is an advantage.
    """
]

for i, resume in enumerate(resumes):
    print(f"\n📄 Resume {i+1} Match Scores:")
    for j, jd in enumerate(job_descriptions):
        score = get_resume_match_score(resume, jd)
        print(f"  ↪️  JD {j+1}: {score}% match")
