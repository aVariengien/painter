import streamlit as st
from processors import chunk_markdown
from concurrent.futures import ThreadPoolExecutor
import json
from processors import Annotation, AnnotatedChunk
import anthropic
from typing import List, Callable
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import random
from difflib import SequenceMatcher
from runware import Runware, IImageInference
import asyncio
import os
import base64
import uuid
import aiohttp
from PIL import Image, ImageDraw, ImageFont
import io
import time
from file_import import is_valid_path_or_url, import_file, safe_filename
from prompts import QUOTE_EXTRACT_PROMPT
import re

def force_text_quote(text: str, chunk_text: str, threshold: float = 0.8) -> str:
    """
    Find the best matching substring in chunk_text that is similar to text.
    Uses SequenceMatcher to find the closest match above the threshold.
    
    Args:
        text: The text to find
        chunk_text: The full text to search in
        threshold: Minimum similarity ratio (0-1) to consider a match
        
    Returns:
        The best matching substring from chunk_text, or the original text if no good match is found
    """
    # If exact match exists, return it
    if text in chunk_text:
        return text
    
    # Split into words for more efficient matching
    text_words = text.split()
    chunk_words = chunk_text.split()
    
    # If text is too short or too long compared to chunk, return original
    if len(text_words) < 2 or len(text_words) > len(chunk_words):
        return text
    
    # Use a sliding window of similar length
    window_size = len(text_words)
    best_match = None
    best_ratio = 0.0
    
    # Pre-calculate text length for ratio comparison
    text_len = len(text)
    
    # Use a sliding window with step size of 1 word
    for i in range(len(chunk_words) - window_size + 1):
        window = ' '.join(chunk_words[i:i + window_size])
        
        # Quick length check to avoid unnecessary ratio calculation
        if abs(len(window) - text_len) > 20:  # Allow some flexibility in length
            continue
            
        ratio = SequenceMatcher(None, text, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = window
            # Early exit if we find a perfect match
            if best_ratio >= 1.0:
                break
    
    # Return the best match if it's above threshold
    if best_ratio >= threshold and best_match:
        return best_match
    return text

def force_text_quote_tag(text: str, tagged_chunk_text: str, start_tag: int, end_tag: int) -> str:
    """Find the exact quote in the tagged text by looking at the tags.
    
    Args:
        text: The text to find, containing tags like <tag id="12">text<tag id="14">
        tagged_chunk_text: The full text with tags to search in
        
    Returns:
        The exact quote from tagged_chunk_text between the matching tags
    """
    # Find all tag numbers in the input text
    # tag_numbers = re.findall(r'<tag id="(\d+)"/>', text)
    # if not tag_numbers:
    #     print("No tag!!!")
    #     return text


        
    # # Get start and end tag numbers
    # tag_start = max(int(tag_numbers[0])-1, 0)
    # tag_end = int(tag_numbers[-1])
    
    # Find positions of these tags in tagged_chunk_text
    if start_tag > 0:
        start_tag -=1

    tag_start = f'<tag id="{start_tag}"/>'
    tag_end = f'<tag id="{end_tag}"/>'
    
    start_pos = tagged_chunk_text.find(tag_start)
    end_pos = tagged_chunk_text.find(tag_end)
    
    if start_pos == -1 or end_pos == -1:
        print("Tag not found!!!")
        print(f"{start_pos} {end_pos}")
        return text
        
    # Extract text between tags
    quote = tagged_chunk_text[start_pos + len(tag_start):end_pos]
    print(f"{tag_start} - {tag_end}")
    print("====")
    print("TEXT")
    print(text)


    print("QUOTE")
    print(quote)
    
    # Remove any remaining tags from the quote
    quote = re.sub(r'<tag id="\d+"/>', '', quote)
    
    quote.strip()
    print("CLEAN QUOTE")
    print(quote)
    print("======")

    if quote and quote[0] == '\n':
        quote = quote[1:]
    return quote


client = AsyncOpenAI()

async def annotate_chunk_with_analogies(chunk: AnnotatedChunk, index: int = -1) -> AnnotatedChunk:
    """Process a single chunk with Claude to extract annotations."""
    start_time = time.time()
    class AnalogyList(BaseModel):
        class Analogy(BaseModel):
            excerpt: str = Field(..., description="A short key quote (1-2 sentences) copied VERBATIM from the text, keep the <tag id='XX'/> from the text.")
            excerpt_tag_start: int = Field(..., description="""The id of the tag before the start of the excerpt. The excerpt should start _after_ this tag. For instance, if the tag is <tag id="42"/> this field is 42.""")
            excerpt_tag_end: int = Field(..., description="""The id of the tag at the end of the excerpt. For instance, if the tag is <tag id="42"/> this field is 42.""")
            visual_description: str = Field(..., description="A concise description of a photography that illustrates the excerpt, optionally including an art style")
        analogies: list[Analogy] = Field(..., description="List of 1-3 key passages with their corresponding visual descriptions")

    # Create tagged version of chunk text with incrementing tags every 5 spaces
    tagged_chunk_text = ""
    tag_counter = 0
    space_counter = 0
    
    for char in chunk.chunk_text:
        tagged_chunk_text += char
        if char == " ":
            space_counter += 1
            if space_counter % 5 == 0:
                tagged_chunk_text += f'<tag id="{tag_counter}"/>'
                tag_counter += 1
        if char in ".,:;!?":
            tagged_chunk_text += f'<tag id="{tag_counter}"/>'
            tag_counter += 1

    response = await client.responses.parse(
        model="gpt-4.1-mini-2025-04-14",
        input=[
            {"role": "system", "content": QUOTE_EXTRACT_PROMPT},
            {
                "role": "user",
                "content": tagged_chunk_text,
            },
        ],
        text_format=AnalogyList
    )
    annotations = []
    if response.output_parsed and response.output_parsed.analogies:
        for analogy in response.output_parsed.analogies:
            analogy.excerpt = force_text_quote_tag(analogy.excerpt, tagged_chunk_text, analogy.excerpt_tag_start, analogy.excerpt_tag_end)
            if analogy.excerpt in chunk.chunk_text:
                annotation = Annotation(
                    text=analogy.excerpt,
                    type="analogy",
                    associated_text=analogy.visual_description
                )
                annotations.append(annotation)
                st.session_state["quote_found"] +=1
            else:
                print(f"Excerpt not found in chunk: {analogy.excerpt}")
                st.session_state["quote_not_found"] += 1

    chunk.annotations = annotations
    print(f"Annotated chunk {index} in {time.time() - start_time:.2f} seconds. Output tokens: {response.usage.input_tokens}, Input tokens: {response.usage.output_tokens}")
    return chunk

async def process_chunks_in_parallel(chunks: List[AnnotatedChunk], f: Callable[[AnnotatedChunk], AnnotatedChunk], max_chunk_idx: int = None) -> List[AnnotatedChunk]:
    """Process chunks in parallel using async OpenAI calls.
    
    Args:
        chunks: List of chunks to process
        f: Function to apply to each chunk
        max_chunk_idx: Maximum chunk index to process (exclusive). If None, process all chunks.
    """
    if max_chunk_idx is None:
        max_chunk_idx = len(chunks)
    
    # Only process chunks that haven't been annotated yet
    tasks = [f(chunk, index) for index, chunk in enumerate(chunks[:max_chunk_idx]) if not chunk.annotations]
    if tasks:
        processed_chunks = await asyncio.gather(*tasks)
        # Update the original chunks with the processed results
        for i, chunk in enumerate(chunks[:max_chunk_idx]):
            if not chunk.annotations and i < len(processed_chunks):
                chunk.annotations = processed_chunks[i].annotations
    return chunks

def highlight_text(text: str, annotations: List[Annotation]) -> str:
    """Highlight annotated text in the chunk."""
    result = text
    for annotation in annotations:
        # random color
        color = '#{:06x}'.format(random.randint(0, 0xFFFFFF))
        highlighted = f"""<span style="background-color: {color}">{annotation.text}</span>"""
        result = result.replace(annotation.text, highlighted)
    return result

async def generate_analogy_image(visual_description: str, index: int = -1) -> str:
    """
    Generate an image based on the visual description of an analogy.
    
    Args:
        visual_description: The visual description to use as a prompt
        
    Returns:
        Base64 encoded image string
    """
    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))
    await runware.connect()
    
    request_image = IImageInference(
        positivePrompt=visual_description,
        model="runware:101@1",
        numberResults=1,
        negativePrompt="blurry, low quality, distorted",
        height=512,
        width=768,
    )
    
    images = await runware.imageInference(requestImage=request_image)
    print(f"Generated image {index}")
    if images:
        # Download the image asynchronously using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(images[0].imageURL) as response:
                if response.status == 200:
                    content = await response.read()
                    image_base64 = base64.b64encode(content).decode('utf-8')
                    return image_base64
    return None

def compose_image(generated_image_base64: str, text_before: str, text: str, text_after: str, associated_text: str) -> str:
    """
    Create a composed image with generated image and description.
    
    Args:
        generated_image_base64: Base64 encoded generated image
        text_before: The sentence before the quote (unused)
        text: The quote from the text (unused)
        text_after: The sentence after the quote (unused)
        associated_text: The description of the analogy
        
    Returns:
        Base64 encoded composed image
    """
    # Convert base64 to PIL Image
    image_data = base64.b64decode(generated_image_base64)
    gen_image = Image.open(io.BytesIO(image_data))
    
    # Create a new image with white background
    width = gen_image.width
    height = gen_image.height + 220  # Extra space for description
    composed_image = Image.new('RGB', (width, height), 'white')
    
    # Paste the generated image at the top
    composed_image.paste(gen_image, (0, 0))
    
    # Create a drawing context
    draw = ImageDraw.Draw(composed_image)
    
    # Load fonts
    #try:
    #    font = ImageFont.truetype("./fonts/Montserrat-MediumItalic.ttf", 20)
    #except IOError:
    font = ImageFont.load_default(size=20)
    
    def wrap_text(text: str, max_width: int, font: ImageFont.FreeTypeFont) -> list[str]:
        """Helper function to wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            line_width = draw.textlength(' '.join(current_line), font=font)
            if line_width > max_width:
                if len(current_line) > 1:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
                else:
                    lines.append(' '.join(current_line))
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        return lines
    
    # Add the description below the image
    max_width = width - 40  # Leave some margin
    y = gen_image.height + 20
    
    # Draw description in black
    desc_lines = wrap_text(associated_text, max_width, font)
    for line in desc_lines:
        draw.text((20, y), line, font=font, fill='black')
        y += 20
    
    # Convert back to base64
    buffered = io.BytesIO()
    composed_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def get_sentence_before_and_after(text: str, chunk_text: str) -> tuple[str, str]:
    """
    Get the context around the text, up to 30 words before and after.
    Tries to start and stop at punctuation when possible, but ensures at least 10 words.
    
    Args:
        text: The target text to find context for
        chunk_text: The full text to search in
        
    Returns:
        Tuple of (text_before, text_after)
    """
    # Find the position of the text in the chunk
    pos = chunk_text.find(text)
    if pos == -1:
        return "", ""
    
    # Split into words
    words = chunk_text.split()
    target_words = text.split()
    
    # Find the position of the first word of our target text
    target_start = -1
    for i in range(len(words) - len(target_words) + 1):
        if words[i:i+len(target_words)] == target_words:
            target_start = i
            break
    
    if target_start == -1:
        return "", ""
    
    # Get words before and after
    min_words = 10
    max_words = 30
    
    # Get words before
    before_words = words[max(0, target_start - max_words):target_start]
    before_text = " ".join(before_words)
    
    # Try to find a good starting point for before_text
    if len(before_words) >= min_words:
        # Look for punctuation marks
        punctuation_marks = ['.', ';', ':', '!', '?', ',']
        for mark in punctuation_marks:
            last_punct = before_text.rfind(mark)
            if last_punct != -1 and len(before_text[last_punct+1:].split()) >= min_words:
                before_text = before_text[last_punct+1:].strip()
                break
    
    # Get words after
    after_words = words[target_start + len(target_words):target_start + len(target_words) + max_words]
    after_text = " ".join(after_words)
    
    # Try to find a good ending point for after_text
    if len(after_words) >= min_words:
        # Look for punctuation marks
        punctuation_marks = ['.', ';', ':', '!', '?', ',']
        for mark in punctuation_marks:
            first_punct = after_text.find(mark)
            if first_punct != -1 and len(after_text[:first_punct+1].split()) >= min_words:
                after_text = after_text[:first_punct+1].strip()
                break
    
    return before_text, after_text


#html_img_container = """
#<div style="position: absolute; z-index: 1;">
#    <div style="position: relative; right: -33vw; top: -21vw; z-index: 1;">
#        <img src="data:image/png;base64,{img_b64}" style="max-height: 32vw; width: auto; max-width: 100%";/>
#    </div>
#</div>
#"""

html_img_container = """
<div style="float: right;">
    <img src="data:image/png;base64,{img_b64}" style="max-height: 420px; width: auto; max-width: 100%";/>
</div>
"""


async def generate_all_images(doc, max_chunk_idx: int = None):
    """Generate images for all annotations in parallel.
    
    Args:
        doc: The document containing chunks to process
        max_chunk_idx: Maximum chunk index to process (exclusive). If None, process all chunks.
    """
    if max_chunk_idx is None:
        max_chunk_idx = len(doc.annotated_chunks)
    
    tasks = []
    idx = 0
    for chunk in doc.annotated_chunks[:max_chunk_idx]:
        for annotation in chunk.annotations:
            if annotation.type == "analogy" and not annotation.image:  # Only process annotations without images
                tasks.append(generate_analogy_image(annotation.associated_text, idx))
                idx += 1
    
    # Run all image generation tasks in parallel
    results = await asyncio.gather(*tasks)
    
    # Update the document with generated images
    image_map = {}
    result_idx = 0
    for chunk in doc.annotated_chunks[:max_chunk_idx]:
        for annotation in chunk.annotations:
            if annotation.type == "analogy" and not annotation.image:
                image_base64 = results[result_idx]
                if image_base64:
                    placeholder = f"[GEN_IMG_{str(uuid.uuid4())[:8]}]"

                    # add a border to the image
                    sentence_before, sentence_after = get_sentence_before_and_after(annotation.text, chunk.chunk_text)

                    composed_image_b64 = compose_image(image_base64, sentence_before, annotation.text, sentence_after, annotation.associated_text)
                    image_map[placeholder] = html_img_container.format(img_b64=composed_image_b64)
                    annotation.image = placeholder
                result_idx += 1
    
    # Update the document's display markdown
    display_markdown = doc.text_only
    for chunk in doc.annotated_chunks[:max_chunk_idx]:
        for annotation in chunk.annotations:
            if annotation.type == "analogy" and annotation.image:
                # Add the image and description below the annotation text
                image_placeholder = f"{annotation.image}"
                description = f"*{annotation.associated_text}*"

                # replace only the first occurrence of the annotation text, only if it's not already in the display markdown
                if f"""<span style="background-color: #FFFF00">{annotation.text}</span>""" not in display_markdown:
                    chunk.chunk_text = chunk.chunk_text.replace(
                        annotation.text,
                        f"{image_placeholder}{f"""<span style="background-color: #FFFF00">{annotation.text}</span>"""}",
                        1
                    )
                    display_markdown = display_markdown.replace(
                        annotation.text,
                        f"\n\n{image_placeholder}\n\n{f"""<span style="background-color: #FFFF00">{annotation.text}</span>"""}",
                        1
                    )

                    print("ANOTATION TEXT", annotation.text)
    
    doc.display_markdown = display_markdown
    doc.image_map.update(image_map)
    return doc


# check if the document is already loaded by checking the session state
if "doc" not in st.session_state:
    st.session_state["doc"] = None
    st.session_state["quote_not_found"] = 0
    st.session_state["quote_found"] = 0


def jina_import(url: str):
    """Import content from a Jina URL.
    
    Args:
        url (str): The URL to import from
        
    Returns:
        tuple: (content, title)
    """
    import requests
    
    # Append Jina prefix if not already present
    if not url.startswith("https://r.jina.ai/"):
        url = f"https://r.jina.ai/{url}"
    
    # Get the content
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch content from {url}")
    
    content = response.text
    
    # Extract title from first line
    first_line = content.split('\n')[0]
    if first_line.startswith("Title: "):
        title = first_line[7:]  # Remove "Title: " prefix
    else:
        title = url.split('/')[-1]  # Use last part of URL as fallback
    
    return content, title


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Painter - Visual Analogy Generator",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
    )


    # Set theme to light
    st.markdown("""
        <style>
            .stApp {
                background-color: #ffffff;
            }
            .stSidebar {
                background-color: #f0f2f6;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Website Painter 🎨")


    # Add controls to sidebar
    with st.sidebar:

        st.markdown("""Add images to your reading!
1. Enter the url in the field
2. Click 'import file'
3. Choose the number of chunks to illustrate.""")
        file_path = st.sidebar.text_area("Enter an URL (online only, PDFs are supported).")
        import_files = st.sidebar.button("⏩️ Import file")
        chunk_size = st.sidebar.slider("Chunk size", help="Smaller chunks size means higher image density.", min_value=2000, max_value=10000, value=6000)
        if import_files:
            if is_valid_path_or_url(file_path):
                try:
                    # add a spinner
                    with st.spinner("Importing file..."):
                        file, title = jina_import(file_path) # old version file_import
                        title = safe_filename(title)
                        if not file_path.endswith(".md"):
                            with open(f"./files/{title}.md", "w") as f:
                                f.write(file)
                            st.session_state["doc"] = chunk_markdown(f"./files/{title}.md", chunk_size=chunk_size)
                        else:
                            st.session_state["doc"] = chunk_markdown(file_path, chunk_size=chunk_size)
                            
                except Exception as e:
                    st.info(f"Error while importing {file_path}: {e}")
            else:
                st.info(f"Invalid filename {file_path}")

        if st.session_state["doc"] is not None:
            max_chunks = st.slider("Number of chunks to process", min_value=1, max_value=len(st.session_state["doc"].annotated_chunks), value=1)
            
            if st.button("Generate AI Images"):
                with st.spinner("Generating images..."):
                    st.session_state["quote_not_found"] = 0
                    st.session_state["quote_found"]= 0
                    # Process chunks and generate images up to max_chunks
                    st.session_state["doc"].annotated_chunks = asyncio.run(process_chunks_in_parallel(
                        st.session_state["doc"].annotated_chunks, 
                        annotate_chunk_with_analogies,
                        max_chunk_idx=max_chunks
                    ))
                    st.info("Text annotation finished!")
                    st.info(f"{st.session_state["quote_not_found"]} quotes not found, {st.session_state["quote_found"]} found.")
                    print("Annotation done")
                    st.session_state["doc"] = asyncio.run(generate_all_images(
                        st.session_state["doc"],
                        max_chunk_idx=max_chunks
                    ))
                    st.success("Images generated successfully!")

    # Display the document with or without images
    if st.session_state["doc"] is not None:
        md_str = st.session_state["doc"].get_display_ready(st.session_state["doc"].display_markdown)
        st.markdown(md_str, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

