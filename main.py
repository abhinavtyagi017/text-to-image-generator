import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

# Load pipeline with CPU-safe settings
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None   # Disable safety checker (avoids black images)
    )
    return pipe.to("cpu")

pipe = load_model()

# Streamlit UI
st.set_page_config(page_title="Text-to-Image Generator", page_icon="🎨")
st.title("🎨 Text-to-Image Generator")

prompt = st.text_input("Enter a prompt:", "A cat riding a cycle")

if st.button("Generate Image"):
    with st.spinner("Generating... please wait (CPU may take ~1 min)"):
        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        image.save("generated_image.png")
        st.image(image, caption="Generated Image", use_column_width=True)
        st.success("✅ Image generated successfully!")
