from pydantic import BaseModel, Field


QUOTE_EXTRACT_PROMPT = """Extract a few (1-3) short key passages that are 1 to 2 sentence-long from the text that captures the main story of the text. 

If a reader only reads these passages, then they got the gist of the text content. Repeat exactly the text from the input. The text is in html form, you should keep the elements of the form <tag id='XX'/> that appear, they are crucial for aligning the text.

From the excerpt, write a concise description of a photography that illustrates the excerpt."""


MAXI_PROMPT = """You are a literary analyst with expertise in organizing complex texts into meaningful 2 to 4 logical sections based on thematic elements, narrative arcs, or major plot developments. I need you to analyze a text I provide and break it down into coherent thematic sections.

Important guidelines:
* Extract VERBATIM quotes for all passages, excerpts, and key quotes
* Use mardown formatting to add emphasis like bold and italic to the section descriptions
* Choose section colors to add a visual identity to the section. Choose contrasting colors for the sections so they are easily recognizable.
* Include 1-3 key passages that are around half a page long that best represent the section's themes
* Your landscape descriptions should be detailed and evocative, capturing the section's emotional essence
* Identify 1-3 key quotes per chapter that highlight important moments, revelations, or character development
* Make sure section introductions clearly articulate the thematic questions being explored and how they connect to previous sections
* Please maintain the exact JSON structure provided. This analysis will be used for creating a visual and thematic guide to the book.

# Text

{TEXT}"""


class SectionColor(BaseModel):
    name: str = Field(..., description="The name of the color assigned to this section. Choose contrasting colors for the sections so they are easily recognizable.")
    html_color: str = Field(..., description="The html hex code of the color like #1A2B3C. Choose contrasting colors for the sections so they are easily recognizable.")

class KeyPassage(BaseModel):
    passage_start_tag: str = Field(..., description="The tag id value of the start of the passage. Example value: 'tag-342'.")
    passage_end_tag: str = Field(..., description="The tag id value of the end of the passage. Example value: 'tag-342'.")
    passage_post_process: str = Field(..., description="Keep this field empty, for further processing.")
    chapter: str = Field(..., description="The chapter where this passage appears")

class Chapter(BaseModel):
    chapter_name: str = Field(..., description="The name or title of the chapter, add the name as it is in the table of content")
    chapter_comment: str = Field(..., description="A brief comment or summary about the chapter")
    chapter_start_tag: str = Field(..., description="The tag id value of the start of the chapter. Example value: 'tag-342'.")
    chapter_end_tag: str = Field(..., description="The tag id value of the end of the chapter. Example value: 'tag-342'.")
    key_quotes: list[str]

class Section(BaseModel):
    section_name: str = Field(..., description="The name of the section from the book")
    section_introduction: str = Field(
        ..., 
        description="Introduction to the key questions of this section and how it connects to the previous section by answering the key questions from that section"
    )
    section_color: SectionColor
    key_passages: list[KeyPassage]
    visual_landscape_description: str = Field(
        ..., 
        description="A detailed description of a landscape that illustrates the section's themes or mood"
    )
    chapters: list[Chapter]
    image_b64: str = Field(..., description="Keep this field empty, for further processing.")

class TextStructure(BaseModel):
    sections: list[Section]