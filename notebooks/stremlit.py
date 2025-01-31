import streamlit as st
st.header("this is me")
st.write("hi ganesh")
st.subheader("this is sub header")
st.selectbox("select language",['python','java','c'])

st.checkbox('c++')

st.slider('this is slider',0,100)
st.select_slider("select entry",["best","average","worst"])