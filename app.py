import streamlit as st
from processors import chunk_markdown
from concurrent.futures import ThreadPoolExecutor
import json
from processors import Annotation, AnnotatedChunk
import anthropic
from typing import List, Callable
from pydantic import BaseModel
from openai import AsyncOpenAI
import random
from difflib import SequenceMatcher
from runware import Runware, IImageInference
import asyncio
import os
import base64
import uuid
import requests
from PIL import Image, ImageDraw, ImageFont
import io


def force_text_quote(text: str, chunk_text: str, threshold: float = 0.8) -> str:
    """
    Find the best matching substring in chunk_text that is similar to text.
    Uses SequenceMatcher to find the closest match above the threshold.
    
    Args:
        text: The text to find
        chunk_text: The text to search in
        threshold: Minimum similarity ratio (0-1) to consider a match
        
    Returns:
        The best matching substring from chunk_text, or the original text if no good match is found
    """
    # If exact match exists, return it
    if text in chunk_text:
        return text
        
    # Find the best matching substring
    best_match = None
    best_ratio = 0.0
    
    # Try different window sizes around the original text length
    window_size = len(text)
    for i in range(max(1, window_size - 5), window_size + 6):
        for j in range(len(chunk_text) - i + 1):
            substring = chunk_text[j:j+i]
            ratio = SequenceMatcher(None, text, substring).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = substring
    
    # Return the best match if it's above threshold, otherwise return original text
    if best_ratio >= threshold and type(best_match) == str:
        return best_match
    return text

async def annotate_chunk_with_analogies(chunk: AnnotatedChunk) -> AnnotatedChunk:
    """Process a single chunk with Claude to extract annotations."""
    client = AsyncOpenAI()

    class AnalogyList(BaseModel):
        class Analogy(BaseModel):
            excerpt: str
            visual_description: str
        analogies: list[Analogy]

    response = await client.responses.parse(
        model="gpt-4.1-mini-2025-04-14",
        input=[
            {"role": "system", "content": "Extract passages (0-3) from the text that contain an analogy or a metaphor that can be visually represented. Repeat exactly the text from the input. Associate a precise visual representation to the passage."},
            {
                "role": "user",
                "content": chunk.chunk_text,
            },
        ],
        text_format=AnalogyList
    )
    annotations = []
    if response.output_parsed and response.output_parsed.analogies:
        for analogy in response.output_parsed.analogies:
            analogy.excerpt = force_text_quote(analogy.excerpt, chunk.chunk_text)
            if analogy.excerpt in chunk.chunk_text:
                annotation = Annotation(
                    text=analogy.excerpt,
                    type="analogy",
                    associated_text=analogy.visual_description
                )
                annotations.append(annotation)
            else:
                print(f"Excerpt not found in chunk: {analogy.excerpt}")

    chunk.annotations = annotations
    return chunk

async def process_chunks_in_parallel(chunks: List[AnnotatedChunk], f: Callable[[AnnotatedChunk], AnnotatedChunk]) -> List[AnnotatedChunk]:
    """Process chunks in parallel using async OpenAI calls."""
    tasks = [f(chunk) for chunk in chunks]
    processed_chunks = await asyncio.gather(*tasks)
    return processed_chunks

def highlight_text(text: str, annotations: List[Annotation]) -> str:
    """Highlight annotated text in the chunk."""
    result = text
    for annotation in annotations:
        # random color
        color = '#{:06x}'.format(random.randint(0, 0xFFFFFF))
        highlighted = f"""<span style="background-color: {color}">{annotation.text}</span>"""
        result = result.replace(annotation.text, highlighted)
    return result

async def generate_analogy_image(visual_description: str) -> str:
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
        model="runware:100@1",
        numberResults=1,
        negativePrompt="blurry, low quality, distorted",
        height=512,
        width=512,
    )
    
    images = await runware.imageInference(requestImage=request_image)
    if images:
        # Download the image and convert to base64
        response = requests.get(images[0].imageURL)
        if response.status_code == 200:
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            return image_base64 #f"![](data:image/png;base64,{image_base64})"
    return None

def compose_image(generated_image_base64: str, text_before: str, text: str, text_after: str, associated_text: str) -> str:
    """
    Create a composed image with text on the left and generated image on the right.
    
    Args:
        generated_image_base64: Base64 encoded generated image
        text_before: The sentence before the quote
        text: The quote from the text
        text_after: The sentence after the quote
        associated_text: The description of the analogy
        
    Returns:
        Base64 encoded composed image
    """
    # Convert base64 to PIL Image
    image_data = base64.b64decode(generated_image_base64)
    gen_image = Image.open(io.BytesIO(image_data))
    
    # Create a new image with white background
    width = gen_image.width * 2  # Double width for text
    height = gen_image.height + 240  # Extra space for description
    composed_image = Image.new('RGB', (width, height), 'white')
    
    # Paste the generated image on the right
    composed_image.paste(gen_image, (gen_image.width, 0))
    
    # Create a drawing context
    draw = ImageDraw.Draw(composed_image)
    
    # Load fonts
    try:
        font1 = ImageFont.truetype("./fonts/Montserrat-Medium.ttf", 30)
        font2 = ImageFont.truetype("./fonts/Montserrat-MediumItalic.ttf", 17)
    except IOError:
        font1 = ImageFont.load_default()
        font2 = ImageFont.load_default()
    
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
    
    # Add the text on the left
    max_width = gen_image.width - 40  # Leave some margin
    y = 20
    
    # Draw text_before in black
    if text_before:
        before_lines = wrap_text(text_before, max_width, font1)
        for line in before_lines:
            draw.text((20, y), line, font=font1, fill='black')
            y += 30
    
    # Draw text in orange
    text_lines = wrap_text(text, max_width, font1)
    for line in text_lines:
        draw.text((20, y), line, font=font1, fill='#FFA500')  # Orange color
        y += 30
    
    # Draw text_after in black
    if text_after:
        after_lines = wrap_text(text_after, max_width, font1)
        for line in after_lines:
            draw.text((20, y), line, font=font1, fill='black')
            y += 30
    
    # Add the description below the generated image
    desc_lines = wrap_text(associated_text, max_width, font2)
    y = gen_image.height + 20
    for line in desc_lines:
        draw.text((gen_image.width + 20, y), line, font=font2, fill='black')
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

async def generate_all_images(doc):
    """Generate images for all annotations in parallel."""
    tasks = []
    for chunk in doc.annotated_chunks:
        for annotation in chunk.annotations:
            if annotation.type == "analogy":
                tasks.append(generate_analogy_image(annotation.associated_text))
    
    # Run all image generation tasks in parallel
    results = await asyncio.gather(*tasks)
    
    # Update the document with generated images
    image_map = {}
    result_idx = 0
    for chunk in doc.annotated_chunks:
        for annotation in chunk.annotations:
            if annotation.type == "analogy":
                image_base64 = results[result_idx]
                if image_base64:
                    placeholder = f"[GEN_IMG_{str(uuid.uuid4())[:8]}]"

                    # add a border to the image
                    sentence_before, sentence_after = get_sentence_before_and_after(annotation.text, chunk.chunk_text)

                    
                    composed_image_b64 = compose_image(image_base64, sentence_before, annotation.text, sentence_after, annotation.associated_text)
                    image_map[placeholder] = f"![](data:image/png;base64,{composed_image_b64})"
                    annotation.image = placeholder
                result_idx += 1
    
    # Update the document's display markdown
    display_markdown = doc.text_only
    for chunk in doc.annotated_chunks:
        for annotation in chunk.annotations:
            if annotation.type == "analogy" and annotation.image:
                # Add the image and description below the annotation text
                image_placeholder = f"{annotation.image}"
                description = f"*{annotation.associated_text}*"

                display_markdown = display_markdown.replace(
                    annotation.text,
                    f"{f"""<span style="background-color: #FFFF00">{annotation.text}</span>"""}\n\n{image_placeholder}\n\n"
                )

                chunk.chunk_text = chunk.chunk_text.replace(
                    annotation.text,
                    f"{f"""<span style="background-color: #FFFF00">{annotation.text}</span>"""}\n\n{image_placeholder}\n\n"
                )
    
    doc.display_markdown = display_markdown
    doc.image_map.update(image_map)
    return doc

def main():
    st.title("Document Annotation Viewer")
    
    # Load and process the document
    doc = chunk_markdown("source/A_Vision_of_Metascience.md", chunk_size=2000)
    print("Chunking done")
    
    # Process chunks in parallel with async OpenAI calls
    doc.chunk_annotations = asyncio.run(process_chunks_in_parallel(doc.annotated_chunks, annotate_chunk_with_analogies))
    print("Annotation done")
    
    # Generate all images in parallel
    doc = asyncio.run(generate_all_images(doc))
    print("Image generation done")
    # Display the document with images
    st.markdown(doc.get_display_ready(doc.display_markdown), unsafe_allow_html=True)

if __name__ == "__main__":
    main()

