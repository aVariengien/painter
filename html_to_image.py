# %%
from PIL import Image
import io
import base64
from html2image import Html2Image
import markdown
from bs4 import BeautifulSoup
import re
import os
import tempfile

def create_gradient_html(markdown_text: str, target_sentence: str, context_words: int = 30) -> str:
    """
    Create an HTML string with gradient text size around a target sentence.
    
    Args:
        markdown_text: The full markdown text
        target_sentence: The sentence to highlight
        context_words: Number of words to show before and after
        
    Returns:
        HTML string with gradient text
    """
    # Convert markdown to HTML
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    
    # Find the position of the target sentence
    pos = text.find(target_sentence)
    if pos == -1:
        return ""
    
    # Split text into words
    words = text.split()
    target_words = target_sentence.split()
    
    # Find the word position
    target_start = -1
    for i in range(len(words) - len(target_words) + 1):
        if words[i:i+len(target_words)] == target_words:
            target_start = i
            break
    
    if target_start == -1:
        return ""
    
    # Get context words
    before_words = words[max(0, target_start - context_words):target_start]
    after_words = words[target_start + len(target_words):target_start + len(target_words) + context_words]
    
    # Create HTML with gradient font sizes
    html_parts = []
    
    # Before text with decreasing font size (smaller as it gets further)
    for i, word in enumerate(before_words):  # Reverse to make closest text largest
        reverse_index = len(before_words) - i - 1
        size = max(12, 24 - (reverse_index * 0.4))  # Decrease from 24px to 12px
        print(size)
        html_parts.append(f'<span style="font-size: {size}px;">{word}</span> ')
    
    # Target sentence in orange
    html_parts.append(f'<span style="font-size: 32px; color: #FFA500;">{target_sentence}</span> ')
    
    # After text with decreasing font size (smaller as it gets further)
    for i, word in enumerate(after_words):
        size = max(12, 24 - (i * 0.4))  # Decrease from 24px to 12px
        html_parts.append(f'<span style="font-size: {size}px;">{word}</span> ')
    
    # Create final HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: 'Montserrat', sans-serif;
                line-height: 1.6;
                padding: 20px;
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
            }}
        </style>
    </head>
    <body>
        <div style="text-align: justify;">
            {''.join(html_parts)}
        </div>
    </body>
    </html>
    """
    return html

def html_to_image(html: str) -> Image.Image:
    """
    Convert HTML to PIL Image.
    
    Args:
        html: HTML string to convert
        
    Returns:
        PIL Image object
    """
    hti = Html2Image()
    # Use new headless mode flags
    hti.browser.flags = [
        '--headless=new',
        '--disable-gpu',
        '--hide-scrollbars',
        '--no-sandbox',
        '--disable-dev-shm-usage'
    ]
    
    # Set the output path to current directory
    hti.output_path = os.getcwd()
    
    # Create a unique temporary file name
    temp_file = 'temp_screenshot.png'
    
    # Save HTML to temporary file and convert to image
    hti.screenshot(html_str=html, save_as=temp_file)
    
    # Open the image with PIL
    image = Image.open(os.path.join(hti.output_path, temp_file))
    
    # Clean up the temporary file
    try:
        os.remove(os.path.join(hti.output_path, temp_file))
    except:
        pass
    
    return image

def test_gradient_text():
    """Test function for gradient text visualization."""
    # Example markdown text
    markdown_text = """
    # The Future of Science

    **Science has always been a collaborative endeavor, but the ways in which scientists collaborate have evolved dramatically over time.**
    From the solitary work of early natural philosophers to the massive international collaborations of today, the social structure of science has undergone profound changes.

    One key development has been the rise of scientific institutions. These organizations provide the infrastructure and resources needed for modern research. 
    They also create environments where scientists can interact, share ideas, and build upon each other's work.

    Another important factor is the increasing specialization of scientific knowledge. As our understanding of the natural world has grown more complex, 
    scientists have had to focus on increasingly narrow areas of study. This specialization has made collaboration essential, as no single researcher 
    can master all the knowledge needed to solve complex problems.
    """
    
    # Target sentence to highlight
    target_sentence = "One key development has been the rise of scientific institutions."
    
    # Create HTML with gradient text
    html = create_gradient_html(markdown_text, target_sentence)
    
    # Convert to image
    image = html_to_image(html)
    
    # Save the result
    image.save('gradient_text_test.png')
    print("Test image saved as 'gradient_text_test.png'")

if __name__ == "__main__":
    test_gradient_text()

# %%
