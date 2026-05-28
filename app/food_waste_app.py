import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Anti Food Waste — Aalborg",
    page_icon="🛒",
)

pg = st.navigation([
    st.Page("pages/Clearance_Offers.py", title="Clearance Offers", icon="🛒"),
    st.Page("pages/Recipe_Finder.py",    title="Recipe Finder",    icon="🍽️"),
])
pg.run()
