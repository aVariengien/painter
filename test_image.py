import streamlit as st


img_b64 = open("image.txt", "r").read()

# Add sliders to control the position
right_position = st.slider("Image Position (right)", min_value=-1000, max_value=500, value=20, step=10)
top_position = st.slider("Image Position (top)", min_value=-1000, max_value=500, value=20, step=10)

# Create a container for the markdown and image
image_container = f"""
<div style="float: right;"">
    <img src="data:image/png;base64,{img_b64}" style="max-height: 20vw; width: auto; max-width: 100%;" />
</div>

""".format(img_b64=img_b64, right_position=right_position, top_position=top_position)

st.markdown(f"""
# Sample Markdown Content
Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
IMG1
{image_container}
This is some sample text that will flow around the image.
The image is positioned absolutely and won't interfere with the text layout.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
IMG2
{image_container}
Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
""".format(image_container=image_container), unsafe_allow_html=True)