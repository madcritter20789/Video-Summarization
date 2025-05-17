import streamlit as st

import csv

st.title("My Small Business")

st.subheader("Description of my small business", divider="violet")

with st.form("Ask me a question:", clear_on_submit=True):
    st.subheader("Ask me a question:")

    description = st.text_area(label="A description of why you're reaching out, including any questions you have.")

    email = st.text_input(label="What email can I reach you at?")

    submitted = st.form_submit_button("Submit")

if submitted:
    with open('user_requests.csv', 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        writer.writerow([description, email])

    st.rerun()