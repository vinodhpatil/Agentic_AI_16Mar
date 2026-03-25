import streamlit as st
from resume_processor import load_resume, analyze_resume, store_to_vectorstore, run_self_query
import os

st.set_page_config(page_title="AI Resume Analyzer", page_icon=":briefcase:")
st.title("AI Resume Analyzer")
st.markdown("Upload your resume and job description to get a detailed analysis of how well your resume matches the job requirements.")

job_desc=st.text_area("Paste Job Description", height=200)
uploaded_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])

if st.button("Analyze & store") and uploaded_file and job_desc:
   with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

   with st.spinner("Analyzing resume..."):
        docs = load_resume(uploaded_file.name)
        report = analyze_resume(docs, job_desc)
        store_to_vectorstore(docs)
        st.success("Resume analyzed and stored successfully!")
        st.success("Analysis Report and stored")
        st.subheader("AI Resume summary")
        st.write(report)
        st.download_button("Download Report", report, file_name="resume_analysis_report.txt")

st.divider()

st.subheader("Query your resume data")
query = st.text_input("Enter your query about the resume data")

if st.button("Search Resume") and query:
    with st.spinner("Searching..."):
        results = run_self_query(query)
        if results:
            for i, res in enumerate(results, 1):
                st.markdown(f"**Result {i}:**")
                st.write(res.page_content.strip())
        else:
            st.warning("No relevant information found.")
