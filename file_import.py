
# %%
from docling.document_converter import DocumentConverter
import logging
import time
from pathlib import Path

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import FigureElement, InputFormat, Table
from docling.datamodel.pipeline_options import PdfPipelineOptions, PipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, HTMLFormatOption

from readability import Document  # type: ignore
from urllib.parse import urlparse
from markdownify import markdownify as md  # type: ignore
from typing import List
import re
from bs4 import BeautifulSoup  # type: ignore
import html2text
import markdown  # type: ignore

import requests  # type: ignore
import base64
import mimetypes
import io
from urllib.parse import urljoin, urlparse


import time
import io
import fitz
from PIL import Image
import requests
from tqdm import tqdm
import concurrent.futures
import functools
import logging
import subprocess
import os
from urllib.parse import urlparse, urljoin


from PyPDF2 import PdfReader, PdfWriter
import tempfile
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class IllustratedDocument:
    raw_content: str
    text_only: str
    image_map: Dict[str, str]
    
    def get_display_ready(self, text: str) -> str:
        """Replace image placeholders with actual image content."""
        result = text
        for placeholder, image_content in self.image_map.items():
            result = result.replace(placeholder, image_content)
        return result

def create_document(markdown_content: str) -> IllustratedDocument:
    """Create a Document object from markdown content."""
    # Regular expression to find image tags
    image_pattern = r'!\[.*?\]\(data:image/[^;]+;base64,[^)]+\)'
    
    # Create a map of placeholders to image content
    image_map = {}
    text_only = markdown_content
    
    # Replace each image with a placeholder
    for match in re.finditer(image_pattern, markdown_content):
        image_content = match.group(0)
        placeholder = f"[IMG_{str(uuid.uuid4())[:8]}]"
        image_map[placeholder] = image_content
        text_only = text_only.replace(image_content, placeholder)
    
    return IllustratedDocument(
        raw_content=markdown_content,
        text_only=text_only,
        image_map=image_map
    )


def timeout_decorator(timeout_duration):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_duration)
                except concurrent.futures.TimeoutError:
                    logging.warning(f"Function {func.__name__} timed out after {timeout_duration} seconds")
                    return None
        return wrapper
    return decorator

def import_pdf_with_simpletex(pdf_path: str, uat_token: str):
    """
    Import PDF using SimpleTex OCR API.
    
    Args:
        pdf_path (str): Path to the PDF file
        uat_token (str): SimpleTex API authentication token
    
    Returns:
        tuple: (markdown_content, document_name)
    """
    def pillow_image_to_file_binary(image):
        bytes_io = io.BytesIO()
        image.save(bytes_io, format='PNG')
        return bytes_io.getvalue()

    def convert_pdf_to_images(pdf_binary, dpi=100):
        doc = fitz.open("pdf", pdf_binary)
        images = []
        for i in range(doc.page_count):
            page = doc[i]
            image = page.get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", [image.width, image.height], image.samples)
            images.append(image)
        return images

    def pdf_ocr(image):
        api_url = "https://server.simpletex.cn/api/doc_ocr/"
        header = {"token": uat_token}
        img_file = {"file": pillow_image_to_file_binary(image)}
        res = requests.post(api_url, files=img_file, data={}, headers=header).json()
        return res["res"]["content"]

    try:
        # Read PDF file
        file_binary = open(pdf_path, 'rb').read()
        images = convert_pdf_to_images(file_binary)
        
        # Process each page
        final_markdown_content = ""
        for image in tqdm(images, desc="Processing pages with SimpleTex OCR"):
            markdown_page = pdf_ocr(image)
            final_markdown_content += markdown_page + "\n\n---\n\n"
        
        # Get document name from path
        doc_name = Path(pdf_path).stem
        
        return final_markdown_content, doc_name
    
    except Exception as e:
        logging.error(f"Error in SimpleTex OCR processing: {str(e)}")
        raise


def try_local_pdf_import(path: str):
    return import_pdf(path)

def download_pdf(url: str, cache_folder: str = "downloads/pdf") -> str:
    """
    Download a PDF from a given URL and save it to the cache folder.

    Args:
    url (str): URL of the PDF to download.
    cache_folder (str): Folder to save the downloaded PDF.

    Returns:
    str: Path to the downloaded PDF file.
    """
    # Create cache folder if it doesn't exist
    os.makedirs(cache_folder, exist_ok=True)

    # Extract filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    # Full path for the downloaded file
    file_path = os.path.join(cache_folder, filename)

    # Check if file already exists in cache
    if os.path.exists(file_path):
        print(f"PDF already in cache: {file_path}")
        return file_path

    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Save the file
    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"PDF downloaded and saved to: {file_path}")
    return file_path

def get_image_base64(image_url: str, base_url: str = None) -> str:
    """Download image and convert it to base64."""
    try:
        # Handle protocol-relative URLs (starting with //)
        if image_url.startswith('//'):
            image_url = 'https:' + image_url

        # If the image URL is relative, make it absolute
        elif base_url and not bool(urlparse(image_url).netloc):
            image_url = urljoin(base_url, image_url)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        }
        response = requests.get(image_url, headers=headers)
        if response.status_code == 200:
            # Determine the image type
            content_type = response.headers.get('content-type')
            if not content_type:
                # Try to guess the content type from the URL
                content_type, _ = mimetypes.guess_type(image_url)
            if not content_type:
                content_type = 'image/png'  # default to PNG if we can't determine type
            
            # Convert to base64
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            return f"data:{content_type};base64,{image_base64}"
    except Exception as e:
        print(f"Error processing image {image_url}: {str(e)}")
        return image_url
    return image_url

def download_html(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    return response.content

def process_images_in_html(html_content: str, base_url: str = None) -> str:
    """Convert all images in HTML to base64 embedded images."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Process all img tags
    for img in soup.find_all('img'):
        src = img.get('src')
        if src:
            base64_src = get_image_base64(src, base_url)
            img['src'] = base64_src
    
    return str(soup)



def get_heading_list(content: str, idx: int, markdown: bool = False) -> List[str]:
    """Given a document content (markdown or html), return the list of heading that covers the text at the string position idx"""
    if markdown:
        return get_markdown_headings(content, idx)
    else:
        return get_html_headings(content, idx)


def get_markdown_headings(content: str, idx: int) -> List[str]:
    headings = []  # type: List[str]
    current_level = 0
    lines = content.split("\n")
    current_position = 0

    for line in lines:
        line_length = len(line) + 1  # +1 for the newline character
        if current_position + line_length > idx:
            break

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

            while current_level >= level:
                if headings:
                    headings.pop()
                current_level -= 1

            headings.append(heading_text)
            current_level = level

        current_position += line_length

    return headings

def get_html_headings(content: str, idx: int) -> List[str]:
    headings = []  # type: List[str]
    current_position = 0
    heading_pattern = re.compile(r"<h([1-6]).*?>(.*?)</h\1>", re.DOTALL)

    for match in heading_pattern.finditer(content):
        start, end = match.span()
        if start > idx:
            break

        level = int(match.group(1))
        heading_text = re.sub(r"<.*?>", "", match.group(2)).strip()

        while len(headings) >= level:
            headings.pop()

        headings.append(heading_text)
        current_position = end
    return headings


def remove_script_tags(html_content):
    """Return the document without script tag."""
    soup = BeautifulSoup(html_content, "html.parser")
    script_tags = soup.find_all("script")
    for tag in script_tags:
        tag.decompose()
    return str(soup)


def import_pdf(path: str):
    pdf_path = Path(path)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    # Get total number of pages
    pdf = PdfReader(pdf_path)
    total_pages = len(pdf.pages)

    # Create temporary directory for single pages
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Initialize empty markdown string
        full_markdown = ""
        
        # Process each page with progress bar
        for page_num in tqdm(range(total_pages), desc="Converting PDF pages"):
            # Create single-page PDF
            writer = PdfWriter()
            writer.add_page(pdf.pages[page_num])
            temp_pdf_path = temp_dir_path / f"page_{page_num + 1}.pdf"
            
            with open(temp_pdf_path, "wb") as temp_pdf:
                writer.write(temp_pdf)
            
            # Convert single page to markdown

            conv_res = doc_converter.convert(temp_pdf_path)
            page_markdown = conv_res.document.export_to_markdown(
                delim="\n", 
                image_mode=ImageRefMode.EMBEDDED
            )
            
            # Append to full markdown with page separator
            if page_markdown.strip():  # Only add non-empty pages
                if full_markdown:
                    full_markdown += "\n\n---\n\n"  # Page separator
                full_markdown += page_markdown

    return full_markdown, pdf_path.stem


def fix_markdown_links(markdown_content: str) -> str:
    """
    Fix markdown links by removing unwanted spaces.
    """
    # Pattern to match markdown links: [text](url)
    pattern = r'\[(.*?)\]\((.*?)\)'
    
    def clean_link(match):
        text, url = match.groups()
        # Clean up the URL by removing unwanted spaces
        clean_url = url.strip().replace(' ', '')
        return f'[{text}]({clean_url})'
    
    # Replace all markdown links with cleaned versions
    fixed_content = re.sub(pattern, clean_link, markdown_content)
    return fixed_content

def import_html(html_content: str, url: str = None):
    # First process all images to base64
    html_content = process_images_in_html(html_content, url)
    
    doc = Document(html_content)
    title = doc.title()
    
    # Convert to markdown with embedded images
    content = markdown.markdown(html2text.html2text(remove_script_tags(html_content)))
    content = content.replace("\n", " ") #remove some unnecessary line breaks
    content = md(
        content,
        convert=[
            "h1", "h2", "h3", "h4", "h5", "h6",
            "p", "a", "strong", "b", "em", "i",
            "ul", "ol", "li", "blockquote", "code",
            "pre", "img", "hr", "table", "tr",
            "th", "td", "br",
        ],
    )
    content = fix_markdown_links(content)
    content.replace("$", "\$")
    return content, title

def convert_epub_to_html(epub_file: str) -> str:
    """
    Convert EPUB file to HTML using pandoc.

    Args:
        epub_file (str): Path to the EPUB file

    Returns:
        str: Path to the generated HTML file
    """
    base_name = os.path.splitext(epub_file)[0]
    html_file = f"{base_name}.html"

    try:
        subprocess.run(['pandoc', '-f', 'epub', '-t', 'html', '-o', html_file, epub_file], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting EPUB to HTML: {e}")
        raise

    return html_file

def import_file(file_name: str, simpletex_uat: str = "mC1CUanhaUOBBlVROBgOtVzUEjyxppa22XP8HbXYybKCn9RaHpSp4sV0QyRvx2r2"):
    """
    Import file with fallback to SimpleTex OCR for PDFs.
    
    Args:
        file_name (str): Path or URL to the file
        simpletex_uat (str): SimpleTex API authentication token
    """
    if file_name.startswith("http://") or file_name.startswith("https://"):
        parsed_url = urlparse(file_name)
        if parsed_url.path.endswith(".pdf") or "arxiv.org/pdf/" in file_name:
            pdf_path = download_pdf(file_name)
            file_name = "./" + pdf_path
    
    if file_name.endswith(".pdf"):
        # Try local PDF import first
        result = try_local_pdf_import(file_name)
        
        # If local import times out or fails and SimpleTex token is provided, use SimpleTex
        if result is None and simpletex_uat:
            logging.info("Local PDF import timed out, falling back to SimpleTex OCR")
            return import_pdf_with_simpletex(file_name, simpletex_uat)
        elif result is None:
            raise TimeoutError("PDF import timed out and no SimpleTex token provided")

    
    # Handle other file types as before
    if file_name.startswith("http://") or file_name.startswith("https://"):
        html_content = download_html(file_name)
        result = import_html(html_content, file_name)
    elif file_name.endswith(".html"):
        with open(file_name, "r") as f:
            html_content = f.read()
        result = import_html(html_content, file_name)
    elif file_name.endswith(".epub"):
        # Convert EPUB to HTML using pandoc
        html_file = convert_epub_to_html(file_name)
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        os.remove(html_file)  # Clean up temporary HTML file
        result = import_html(html_content, file_name)
    content, title = result
    if "[no-title]" in title:
        return content, os.path.basename(file_name)
    else:
        return result

# %%
#import_file("./local/algebra_of_wealth.html")
# %%

# files = ["https://estinst.ee/wp-content/uploads/2017/03/589_Estonian_Language_2015_WEB.pdf",
# "https://www.radicalphilosophy.com/article/the-philosophical-disability-of-reason",
# "https://www.lesswrong.com/posts/5yFj7C6NNc8GPdfNo/subskills-of-listening-to-wisdom",
# "https://www.lesswrong.com/posts/nAsMfmxDv6Qp7cfHh/fabien-s-shortform?commentId=gGDAXomb2ihucF4Ls",
# "https://www.planned-obsolescence.org/scale-schlep-and-systems/",
# "https://en.wikipedia.org/wiki/As_We_May_Think",
# "https://shs.cairn.info/revue-reseaux-2013-1-page-163?lang=fr"]


# simpletex_token = "mC1CUanhaUOBBlVROBgOtVzUEjyxppa22XP8HbXYybKCn9RaHpSp4sV0QyRvx2r2"

# for i, file_name in enumerate(files):
#     print(file_name)
#     file, title = import_file(file_name, simpletex_token)
#     with open(f"./files/{title}.md", "w") as f:
#         f.write(file)

# # %%

# file, title = import_file("https://estinst.ee/wp-content/uploads/2017/03/589_Estonian_Language_2015_WEB.pdf")


# # %%
# source = "https://arxiv.org/pdf/2312.10091"  # document per local path or URL
# converter = DocumentConverter()
# result = converter.convert(source)

# with open("text1.md", "w") as f:
#     f.write(result.document.export_to_html())

# # %%
# source = "https://alexandrevariengien.com/an-alive-blackboard"  # document per local path or URL
# converter = DocumentConverter()
# result = converter.convert(source)

# with open("text2.html", "w") as f:
#     f.write(result.document.export_to_html())

# # %%


# pipeline_options = PdfPipelineOptions()
# pipeline_options.images_scale = 2.0
# pipeline_options.generate_page_images = True
# pipeline_options.generate_picture_images = True

# doc_converter = DocumentConverter(
#     format_options={
#         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
#     }
# )

# look_before_you_leap = "https://arxiv.org/pdf/2312.10091"  # document per local path or URL
# result = doc_converter.convert(look_before_you_leap)

# with open("look_b_leap.html", "w") as f:
#     f.write(result.document.export_to_html())


# # %%




# # # Save markdown with externally referenced pictures
# # md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
# # conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

# # # Save HTML with externally referenced pictures
# # html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
# # conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

# end_time = time.time() - start_time

# print(f"Document converted and figures exported in {end_time:.2f} seconds.")
# # %%
