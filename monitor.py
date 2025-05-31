import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt

st.header('YOLO training monitor')


csv_file = 'results.csv'

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)

    df.columns = df.columns.str.strip()

    show_box_train = st.checkbox("Box Loss (Train)", value=True)
    show_cls_train = st.checkbox("Class Loss (Train)")
    show_box_val = st.checkbox("Box Loss (Val)")
    show_cls_val = st.checkbox("Class Loss (Val)")


    fig, ax = plt.subplots(figsize=(10, 6))

    if show_box_train:
        ax.plot(df["epoch"], df["train/box_loss"], label="Box Loss (Train)", color='blue')
    if show_cls_train:
        ax.plot(df["epoch"], df["train/cls_loss"], label="Class Loss (Train)", color='orange')
    if show_box_val:
        ax.plot(df["epoch"], df["val/box_loss"], label="Box Loss (Val)", color='green')
    if show_cls_val:
        ax.plot(df["epoch"], df["val/cls_loss"], label="Class Loss (Val)", color='red')


    if show_box_train or show_cls_train or show_box_val or show_cls_val:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    else:
        st.warning("لطفاً حداقل یک نمودار را انتخاب کنید.")
else:
    st.error("no results.csv found")