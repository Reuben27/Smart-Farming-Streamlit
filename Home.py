import streamlit as st

st.set_page_config(page_title = "CS 499 Project Course")

st.title("CS 499: Project Course")
st.text("Advisor: Prof. Shanmugananthan Raman")
st.write("Our project focused on enhancing agricultural insights using drone imagery of cotton crops. We've developed a multi-faceted solution aimed at aiding in crop analysis and dataset generations. Our project includes polygon annotation to precisely outline crop features, object detection for identifying specific elements within the crops, and image segmentation for detailed analysis. Additionally, we've explored GANs for generating the cotton crop dataset. To make our findings accessible, we've deployed a user-friendly Streamlit interface showcasing object detection, image segmentation, and a streamlined 3D model pipeline. This project reflects our commitment to leveraging technology for practical agricultural applications.")
st.write("This project delves into the transformative potential of smart farming in organic agriculture, utilizing drone-based aerial imagery. The study aims to construct a detailed representation of the farm terrain and its vegetation cover. By generating a 3D model of the farm terrain and calculating key vegetation indices such as NDVI, NDWI, LAI, Vegetation Cover Fraction, and Soil Moisture Index (SMI), our study aims to provide detailed insights into crop health and vitality. Emphasizing techniques such as index calculation, channel integration, and image super-resolution, we aim to offer a comprehensive toolset for optimizing agricultural management practices. Additionally, the exploration of SAHI for small object detection aids in accurately identifying and monitoring minute features within the agricultural landscape.")

st.sidebar.success("Choose your Demo") 