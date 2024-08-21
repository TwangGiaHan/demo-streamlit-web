import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import base64

st.write("## Remove background from your image 🌄")
st.write(":dog: Try uploading an image to watch the background magically removed. Full quality images can be downloaded from the sidebar. :grin:")

st.sidebar.header("Upload and Dowload :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Dowload the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def fix_image(upload):
    image = Image.open(upload)
    col1, col2 = st.columns(2)
    col1.write("Original Image :camera:")
    col1.image(image)

    fixed = remove(image)
    col2.write("Fixed Image 🔧")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", mime="image/png")

my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(my_upload)
else:
    fix_image("./assets/panda.jpg")
