import io
import json
import re
import time
from typing import Any, Dict, List

import streamlit as st
from PIL import Image, ImageOps
from huggingface_hub import InferenceClient
from google import genai
from google.genai import types

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


# ============================================================
# DEVELOPER SETTINGS — ONLY CHANGE THESE 4 VALUES
# ============================================================

HF_TOKEN = st.secrets["HF_TOKEN"]

HF_MODEL = "Qwen/Qwen3-32B"

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_MODEL = "gemini-3.5-flash"


# ============================================================
# APP SETTINGS
# ============================================================

MAX_EXPERIENCE = 5
MAX_EDUCATION = 5
MAX_PROJECTS = 5
MAX_CERTIFICATIONS = 5

st.set_page_config(
    page_title="AI Resume Builder",
    page_icon="📄",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def clean_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [clean_text(x) for x in value if clean_text(x)]
    return [clean_text(value)]


def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
    return name.strip("_") or "resume"


def extract_json(text: str) -> Dict[str, Any]:
    text = clean_text(text)
    if not text:
        raise ValueError("AI ne empty response diya.")

    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("AI response valid JSON mein nahi tha.")


def validate_settings():
    if not HF_TOKEN or "PASTE_YOUR" in HF_TOKEN:
        return False, "Hugging Face token add karo."
    if not GEMINI_API_KEY or "PASTE_YOUR" in GEMINI_API_KEY:
        return False, "Gemini API key add karo."
    if not HF_MODEL:
        return False, "HF model missing hai."
    if not GEMINI_MODEL:
        return False, "Gemini model missing hai."
    return True, ""


def is_temporary_high_demand_error(error: Exception) -> bool:
    """Retry only temporary overload/rate-limit style API errors."""
    text = str(error).lower()
    temporary_terms = [
        "high demand",
        "overloaded",
        "temporarily unavailable",
        "service unavailable",
        "unavailable",
        "resource_exhausted",
        "resource exhausted",
        "rate limit",
        "ratelimit",
        "too many requests",
        "429",
        "503",
    ]
    return any(term in text for term in temporary_terms)


def retry_high_demand_request(request_function, service_name: str):
    """
    Send the request again every 3 minutes when the provider reports
    a temporary high-demand/overload/rate-limit condition.
    Other errors are raised immediately.
    """
    attempt = 1

    while True:
        try:
            return request_function()
        except Exception as error:
            if not is_temporary_high_demand_error(error):
                raise

            wait_seconds = 180
            minutes = wait_seconds // 60

            st.warning(
                f"{service_name} is currently busy/high demand. "
                f"Request محفوظ ہے. Attempt {attempt} failed. "
                f"Dobara request {minutes} minutes baad automatically send hogi."
            )

            progress = st.progress(0)
            status_box = st.empty()

            for remaining in range(wait_seconds, 0, -1):
                progress.progress((wait_seconds - remaining) / wait_seconds)
                status_box.info(
                    f"Automatic retry in {remaining // 60:02d}:{remaining % 60:02d}"
                )
                time.sleep(1)

            progress.progress(1.0)
            status_box.info("Retrying request now...")
            attempt += 1



# ============================================================
# USER FORM
# ============================================================

def collect_user_data():
    data = {}

    st.markdown(
        '<div class="section-title">1. Profile Picture</div>',
        unsafe_allow_html=True,
    )

    photo = st.file_uploader(
        "Upload your professional picture",
        type=["jpg", "jpeg", "png", "webp"],
        help="Clear face photo upload karna recommended hai.",
    )

    st.markdown(
        '<div class="section-title">2. Personal Information</div>',
        unsafe_allow_html=True,
    )

    name = st.text_input("Full Name *", placeholder="Example: Ubaid Ullah")
    email = st.text_input("Email Address *", placeholder="Example: yourname@gmail.com")
    phone = st.text_input("Phone Number (Optional)", placeholder="+92 300 1234567")
    location = st.text_input("City / Country (Optional)", placeholder="Karachi, Pakistan")
    linkedin = st.text_input("LinkedIn Profile (Optional)")
    github = st.text_input("GitHub / Portfolio (Optional)")

    st.markdown(
        '<div class="section-title">3. Describe Yourself</div>',
        unsafe_allow_html=True,
    )

    summary = st.text_area(
        "Describe yourself briefly",
        placeholder="Tell us about yourself, your profession, strengths, interests and career goals.",
        height=130,
    )

    st.markdown(
        '<div class="section-title">4. Skills</div>',
        unsafe_allow_html=True,
    )

    skills_text = st.text_area(
        "Your Skills *",
        placeholder="Python\nHTML\nCSS\nJavaScript\nGraphic Design",
        height=130,
    )

    skills = [x.strip() for x in skills_text.splitlines() if x.strip()]

    st.markdown(
        '<div class="section-title">5. Work Experience</div>',
        unsafe_allow_html=True,
    )

    experience_count = st.number_input(
        "Number of experiences",
        min_value=0,
        max_value=MAX_EXPERIENCE,
        value=1,
        step=1,
    )

    experiences = []

    for i in range(int(experience_count)):
        st.markdown(f"### Experience {i + 1}")

        job_title = st.text_input(
            "Job Title",
            key=f"job_title_{i}",
            placeholder="Frontend Developer",
        )
        company = st.text_input(
            "Company / Organization",
            key=f"company_{i}",
            placeholder="ABC Company",
        )
        start_date = st.text_input(
            "Start Date",
            key=f"start_date_{i}",
            placeholder="Jan 2024",
        )
        end_date = st.text_input(
            "End Date",
            key=f"end_date_{i}",
            placeholder="Present",
        )
        responsibilities = st.text_area(
            "What did you do?",
            key=f"responsibilities_{i}",
            placeholder="Describe your responsibilities, achievements and work.",
            height=110,
        )

        experiences.append({
            "job_title": job_title,
            "company": company,
            "start_date": start_date,
            "end_date": end_date,
            "responsibilities": responsibilities,
        })

    st.markdown(
        '<div class="section-title">6. Education</div>',
        unsafe_allow_html=True,
    )

    education_count = st.number_input(
        "Number of education entries",
        min_value=0,
        max_value=MAX_EDUCATION,
        value=1,
        step=1,
    )

    education = []

    for i in range(int(education_count)):
        st.markdown(f"### Education {i + 1}")

        qualification = st.text_input(
            "Degree / Qualification",
            key=f"qualification_{i}",
            placeholder="BS Computer Science",
        )
        institution = st.text_input(
            "School / College / University",
            key=f"institution_{i}",
            placeholder="University Name",
        )
        education_start = st.text_input(
            "Start Year",
            key=f"education_start_{i}",
            placeholder="2020",
        )
        education_end = st.text_input(
            "End Year",
            key=f"education_end_{i}",
            placeholder="2024",
        )

        education.append({
            "qualification": qualification,
            "institution": institution,
            "start_year": education_start,
            "end_year": education_end,
        })

    st.markdown(
        '<div class="section-title">7. Languages</div>',
        unsafe_allow_html=True,
    )

    languages_text = st.text_area(
        "Which languages do you speak?",
        placeholder="English - Fluent\nUrdu - Native\nHindi - Conversational",
        height=100,
    )

    languages = [x.strip() for x in languages_text.splitlines() if x.strip()]

    st.markdown(
        '<div class="section-title">8. Projects (Optional)</div>',
        unsafe_allow_html=True,
    )

    project_count = st.number_input(
        "Number of projects",
        min_value=0,
        max_value=MAX_PROJECTS,
        value=0,
        step=1,
    )

    projects = []

    for i in range(int(project_count)):
        st.markdown(f"### Project {i + 1}")

        project_name = st.text_input("Project Name", key=f"project_name_{i}")
        project_description = st.text_area(
            "Project Description",
            key=f"project_description_{i}",
            height=90,
        )
        project_link = st.text_input(
            "Project Link (Optional)",
            key=f"project_link_{i}",
        )

        projects.append({
            "name": project_name,
            "description": project_description,
            "link": project_link,
        })

    st.markdown(
        '<div class="section-title">9. Certifications (Optional)</div>',
        unsafe_allow_html=True,
    )

    certification_count = st.number_input(
        "Number of certifications",
        min_value=0,
        max_value=MAX_CERTIFICATIONS,
        value=0,
        step=1,
    )

    certifications = []

    for i in range(int(certification_count)):
        cert_name = st.text_input("Certificate Name", key=f"cert_name_{i}")
        cert_org = st.text_input("Organization", key=f"cert_org_{i}")
        cert_year = st.text_input("Year", key=f"cert_year_{i}")

        certifications.append({
            "name": cert_name,
            "organization": cert_org,
            "year": cert_year,
        })

    st.markdown(
        '<div class="section-title">10. Additional Information (Optional)</div>',
        unsafe_allow_html=True,
    )

    achievements = st.text_area("Achievements / Awards", height=90)
    interests = st.text_area("Interests / Hobbies", height=90)

    data["personal"] = {
        "name": name,
        "email": email,
        "phone": phone,
        "location": location,
        "linkedin": linkedin,
        "github": github,
    }
    data["summary"] = summary
    data["skills"] = skills
    data["experience"] = experiences
    data["education"] = education
    data["languages"] = languages
    data["projects"] = projects
    data["certifications"] = certifications
    data["achievements"] = achievements
    data["interests"] = interests

    return data, photo


# ============================================================
# RETRY SETTINGS / HELPERS
# ============================================================

RETRY_DELAY_SECONDS = 30
MAX_RETRIES = 10


def is_retryable_error(error: Exception) -> bool:
    """
    Retry only temporary API/service errors.
    Permanent configuration/programming errors are not retried.
    """

    message = str(error).lower()

    retry_terms = [
        "503",
        "service unavailable",
        "high demand",
        "currently experiencing high demand",
        "temporarily unavailable",
        "overloaded",
        "429",
        "rate limit",
        "rate_limit",
        "resource exhausted",
        "too many requests",
        "internal server error",
        "502",
        "504",
        "gateway timeout",
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
    ]

    return any(term in message for term in retry_terms)


def run_with_retry(
    function,
    label: str,
    max_retries: int = MAX_RETRIES,
):
    """
    Runs an API request and automatically retries temporary
    service/model errors every 30 seconds.

    User-entered form data remains in Streamlit session state,
    so the user does not have to fill the form again.
    """

    attempt = 1

    while True:
        try:
            return function()

        except Exception as error:

            if not is_retryable_error(error):
                raise

            if attempt >= max_retries:
                raise RuntimeError(
                    f"{label} failed after {max_retries} attempts. "
                    f"Last error: {error}"
                ) from error

            next_attempt = attempt + 1

            st.warning(
                f"{label}: temporary service/model error detected. "
                f"Retry {next_attempt} will be sent in "
                f"{RETRY_DELAY_SECONDS} seconds."
            )

            progress = st.progress(0)
            countdown = st.empty()

            for remaining in range(
                RETRY_DELAY_SECONDS,
                0,
                -1
            ):
                elapsed = RETRY_DELAY_SECONDS - remaining

                progress.progress(
                    int(
                        (elapsed / RETRY_DELAY_SECONDS) * 100
                    )
                )

                countdown.info(
                    f"Retry {next_attempt} in {remaining} seconds..."
                )

                time.sleep(1)

            countdown.empty()
            progress.empty()

            attempt = next_attempt


# ============================================================
# HUGGING FACE
# ============================================================

def generate_hf_draft(user_data: Dict[str, Any]) -> Dict[str, Any]:
    client = InferenceClient(
        api_key=HF_TOKEN,
        provider="auto",
    )

    system_prompt = """
You are an expert professional resume writer.

Transform raw user-provided information into professional resume content.

RULES:
1. Never invent jobs, companies, degrees, dates, certificates, achievements or skills.
2. Improve grammar and wording professionally.
3. Keep the meaning of the user's information.
4. Missing sections must remain empty.
5. Do not create fake information just to fill space.
6. Write concise professional resume content.
7. Create strong bullet points from actual responsibilities.
8. If experience is missing, make the resume suitable for a student/fresher using available education, skills and projects.
9. Do not include empty sections.
10. Return ONLY valid JSON.

JSON:
{
  "headline": "",
  "professional_summary": "",
  "skills": [],
  "languages": [],
  "experience": [
    {
      "job_title": "",
      "company": "",
      "dates": "",
      "bullets": []
    }
  ],
  "education": [
    {
      "qualification": "",
      "institution": "",
      "dates": ""
    }
  ],
  "projects": [
    {
      "name": "",
      "description": "",
      "link": ""
    }
  ],
  "certifications": [
    {
      "name": "",
      "organization": "",
      "year": ""
    }
  ],
  "achievements": [],
  "interests": []
}
"""

    user_prompt = f"""
Here is the user's raw resume information:

{json.dumps(user_data, ensure_ascii=False, indent=2)}

Create professional resume content according to the rules.
"""

    def send_request():
        return client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=5000,
        )

    response = retry_high_demand_request(
        send_request,
        "Hugging Face / Qwen"
    )

    return extract_json(response.choices[0].message.content)


# ============================================================
# GEMINI FINAL POLISH
# ============================================================

def polish_with_gemini(
    user_data: Dict[str, Any],
    hf_draft: Dict[str, Any],
) -> Dict[str, Any]:

    client = genai.Client(api_key=GEMINI_API_KEY)

    schema = {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "professional_summary": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}},
            "languages": {"type": "array", "items": {"type": "string"}},
            "experience": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "job_title": {"type": "string"},
                        "company": {"type": "string"},
                        "dates": {"type": "string"},
                        "bullets": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["job_title", "company", "dates", "bullets"],
                },
            },
            "education": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "qualification": {"type": "string"},
                        "institution": {"type": "string"},
                        "dates": {"type": "string"},
                    },
                    "required": ["qualification", "institution", "dates"],
                },
            },
            "projects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "link": {"type": "string"},
                    },
                    "required": ["name", "description", "link"],
                },
            },
            "certifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "organization": {"type": "string"},
                        "year": {"type": "string"},
                    },
                    "required": ["name", "organization", "year"],
                },
            },
            "achievements": {"type": "array", "items": {"type": "string"}},
            "interests": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "headline",
            "professional_summary",
            "skills",
            "languages",
            "experience",
            "education",
            "projects",
            "certifications",
            "achievements",
            "interests",
        ],
    }

    prompt = f"""
You are the final professional resume editor.

Original user information:
{json.dumps(user_data, ensure_ascii=False, indent=2)}

First AI draft:
{json.dumps(hf_draft, ensure_ascii=False, indent=2)}

Improve the draft into polished professional resume content.

STRICT RULES:
- Never invent information.
- Never create fake employment, education, certificates, skills or achievements.
- Improve grammar, clarity and professional wording.
- Keep bullets concise and impactful.
- If optional information is missing, REMOVE that section.
- Never return an empty section just for design purposes.
- If experience is missing, emphasize education, skills and projects when available.
- If projects are missing, do not create projects.
- If certifications are missing, do not create certifications.
- If achievements are missing, return an empty array.
- Make the final resume look natural even with limited data.
- Output content only. Python handles the visual design and PDF.
- Return only JSON matching the schema.
"""

    def send_request():
        return client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )

    response = retry_high_demand_request(
        send_request,
        "Gemini"
    )

    if getattr(response, "parsed", None):
        if isinstance(response.parsed, dict):
            return response.parsed

    return extract_json(response.text)


# ============================================================
# PHOTO
# ============================================================

def prepare_profile_image(uploaded_file):
    if uploaded_file is None:
        return None

    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.fit(
        image,
        (700, 700),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92)
    output.seek(0)
    return output


# ============================================================
# PDF
# ============================================================

NAVY = colors.HexColor("#123B67")
BLUE = colors.HexColor("#1769AA")
DARK = colors.HexColor("#172033")
GREY = colors.HexColor("#687385")
LIGHT_GREY = colors.HexColor("#D9E0E8")


def register_fonts():
    # Use ReportLab built-in Helvetica fonts.
    # This avoids font-family errors on Windows/Linux deployments.
    return None


def build_resume_pdf(user_data, resume, photo_file) -> bytes:
    register_fonts()

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=13 * mm,
        leftMargin=13 * mm,
        topMargin=13 * mm,
        bottomMargin=13 * mm,
        title="Professional Resume",
        author="AI Resume Builder",
    )

    styles = getSampleStyleSheet()

    name_style = ParagraphStyle(
        "Name",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=24,
        leading=28,
        textColor=NAVY,
        spaceAfter=3,
    )

    headline_style = ParagraphStyle(
        "Headline",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=BLUE,
        spaceAfter=6,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=5,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.7,
        leading=12,
        textColor=DARK,
        spaceAfter=3,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.7,
        leading=10,
        textColor=GREY,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
        spaceAfter=2,
    )

    story = []

    personal = user_data.get("personal", {})
    name = clean_text(personal.get("name"))
    headline = clean_text(resume.get("headline"))
    summary = clean_text(resume.get("professional_summary"))

    header_left = []

    if name:
        header_left.append(Paragraph(name, name_style))

    if headline:
        header_left.append(Paragraph(headline, headline_style))

    contact_parts = []

    for key in ["email", "phone", "location", "linkedin", "github"]:
        value = clean_text(personal.get(key))
        if value:
            contact_parts.append(value)

    if contact_parts:
        header_left.append(
            Paragraph("  •  ".join(contact_parts), small_style)
        )

    header_right = []

    if photo_file:
        header_right.append(
            RLImage(photo_file, width=30 * mm, height=30 * mm)
        )

    header_table = Table(
        [[header_left, header_right]],
        colWidths=[155 * mm, 30 * mm],
    )

    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    story.append(header_table)
    story.append(Spacer(1, 5 * mm))

    if summary:
        story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
        story.append(Paragraph(summary, body_style))

    left_column = []
    right_column = []

    skills = clean_list(resume.get("skills"))
    if skills:
        left_column.append(Paragraph("SKILLS", section_style))
        for skill in skills:
            left_column.append(Paragraph("• " + skill, body_style))

    languages = clean_list(resume.get("languages"))
    if languages:
        left_column.append(Paragraph("LANGUAGES", section_style))
        for language in languages:
            left_column.append(Paragraph("• " + language, body_style))

    experience = resume.get("experience", [])
    if experience:
        right_column.append(Paragraph("WORK EXPERIENCE", section_style))

        for item in experience:
            title = clean_text(item.get("job_title"))
            company = clean_text(item.get("company"))
            dates = clean_text(item.get("dates"))

            heading = title
            if company:
                heading += f" — {company}"

            if heading:
                right_column.append(Paragraph(f"<b>{heading}</b>", body_style))

            if dates:
                right_column.append(Paragraph(dates, small_style))

            for bullet in clean_list(item.get("bullets")):
                right_column.append(Paragraph("• " + bullet, bullet_style))

            right_column.append(Spacer(1, 3))

    education = resume.get("education", [])
    if education:
        right_column.append(Paragraph("EDUCATION", section_style))

        for item in education:
            qualification = clean_text(item.get("qualification"))
            institution = clean_text(item.get("institution"))
            dates = clean_text(item.get("dates"))

            if qualification:
                right_column.append(
                    Paragraph(f"<b>{qualification}</b>", body_style)
                )
            if institution:
                right_column.append(Paragraph(institution, body_style))
            if dates:
                right_column.append(Paragraph(dates, small_style))

            right_column.append(Spacer(1, 3))

    projects = resume.get("projects", [])
    if projects:
        right_column.append(Paragraph("PROJECTS", section_style))

        for project in projects:
            project_name = clean_text(project.get("name"))
            description = clean_text(project.get("description"))
            link = clean_text(project.get("link"))

            if project_name:
                right_column.append(
                    Paragraph(f"<b>{project_name}</b>", body_style)
                )
            if description:
                right_column.append(Paragraph(description, body_style))
            if link:
                right_column.append(Paragraph(link, small_style))

            right_column.append(Spacer(1, 3))

    certifications = resume.get("certifications", [])
    if certifications:
        right_column.append(Paragraph("CERTIFICATIONS", section_style))

        for cert in certifications:
            cert_name = clean_text(cert.get("name"))
            organization = clean_text(cert.get("organization"))
            year = clean_text(cert.get("year"))

            text = cert_name
            if organization:
                text += f" — {organization}"
            if year:
                text += f" ({year})"

            if text:
                right_column.append(Paragraph("• " + text, body_style))

    achievements = clean_list(resume.get("achievements"))
    if achievements:
        right_column.append(Paragraph("ACHIEVEMENTS", section_style))
        for item in achievements:
            right_column.append(Paragraph("• " + item, body_style))

    interests = clean_list(resume.get("interests"))
    if interests:
        right_column.append(Paragraph("INTERESTS", section_style))
        for item in interests:
            right_column.append(Paragraph("• " + item, body_style))

    if left_column and right_column:
        main_table = Table(
            [[left_column, right_column]],
            colWidths=[55 * mm, 125 * mm],
        )

        main_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEAFTER", (0, 0), (0, 0), 0.5, LIGHT_GREY),
                    ("RIGHTPADDING", (0, 0), (0, 0), 8 * mm),
                    ("LEFTPADDING", (1, 0), (1, 0), 8 * mm),
                    ("LEFTPADDING", (0, 0), (0, 0), 0),
                    ("RIGHTPADDING", (1, 0), (1, 0), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )

        story.append(main_table)
    elif right_column:
        story.extend(right_column)
    elif left_column:
        story.extend(left_column)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# UI
# ============================================================

st.markdown(
    '<div class="main-title">AI Resume Builder</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'Apni information fill karein aur AI professional resume banayega.'
    '</div>',
    unsafe_allow_html=True,
)

user_data, uploaded_photo = collect_user_data()

st.markdown("---")

generate = st.button(
    "✨ Generate Professional Resume",
    type="primary",
    use_container_width=True,
)

if generate:
    if not clean_text(user_data["personal"]["name"]):
        st.error("Full Name required hai.")
        st.stop()

    if not clean_text(user_data["personal"]["email"]):
        st.error("Email Address required hai.")
        st.stop()

    if not user_data["skills"]:
        st.error("Kam az kam ek skill add karo.")
        st.stop()

    if not uploaded_photo:
        st.error("Profile picture upload karo.")
        st.stop()

    settings_ok, settings_error = validate_settings()

    if not settings_ok:
        st.error(settings_error)
        st.stop()

    with st.status(
        "Resume generate ho raha hai...",
        expanded=True,
    ) as status:
        try:
            st.write("Step 1/2 — Hugging Face se professional draft...")
            hf_draft = run_with_retry(
                lambda: generate_hf_draft(user_data),
                "Hugging Face request",
            )

            st.write("Step 2/2 — Gemini final professional editing...")
            final_resume = run_with_retry(
                lambda: polish_with_gemini(user_data, hf_draft),
                "Gemini request",
            )

            status.update(
                label="Resume ready!",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            status.update(
                label="Resume generation failed",
                state="error",
                expanded=True,
            )
            st.error("AI request complete nahi ho saki.")
            st.code(str(e))
            st.stop()

    try:
        profile_image = prepare_profile_image(uploaded_photo)

        pdf_bytes = build_resume_pdf(
            user_data,
            final_resume,
            profile_image,
        )

        filename = (
            safe_filename(user_data["personal"]["name"])
            + "_Resume.pdf"
        )

        st.success("Professional resume successfully create ho gaya.")

        st.download_button(
            label="📥 Download Resume PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )

        with st.expander("Resume information preview"):
            st.json(final_resume)

    except Exception as e:
        st.error("PDF banate waqt problem aayi.")
        st.code(str(e))
