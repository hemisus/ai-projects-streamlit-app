import streamlit as st

st.set_page_config(
    page_title="AI Projects",
    page_icon="🐸",
)

st.write("# Welcome to my AI Web Application!👋")

st.sidebar.success("Select a project above.")

st.markdown(
    """
    이 애플리케이션은 지금까지 인공지능을 공부하면서 직접 구현한 프로그램들을 정리하고, 빠르게 실행해보기 위해 만들어졌습니다.

    👈 사이드바에서 지금까지 구현한 프로젝트들을 확인할 수 있습니다.
    ### Hemisus
    - [Github](https://github.com/hemisus)
    - [Instagram](https://www.instagram.com/hemisus_/)
    - [SoundCloud](https://soundcloud.com/hemisus)
"""
)