# Website Painter 🎨

A Streamlit web application that enhances your reading experience by automatically generating visual analogies for key passages in your text. The app analyzes your content, identifies important quotes, and creates AI-generated images to illustrate them.

## 🌟 Features

- Import content from URLs (including PDFs) 
- Automatic text analysis and quote extraction
- AI-generated images for key passages
- Document structure analysis and visualization
- Save and load your annotated documents
- Beautiful markdown rendering with embedded images

## 🚀 Live Demo

Try the application online: [Website Painter](https://website-painter.streamlit.app/)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/website-painter.git
cd website-painter
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
export RUNWARE_API_KEY="your_runware_api_key"
```

## 🏃‍♂️ Running the Application

```bash
streamlit run app.py
```

## 📝 Usage

1. Enter a URL in the sidebar
2. Click "Import file" to load the content
3. Use the slider to select how many chunks to process
4. Click "Generate AI Images" to create visualizations
5. Optionally, click "Generate Text Structure" to show a the document's structure, with sumary of the sections

## 💾 Saving and Loading

- Your documents are automatically saved with their annotations and images
- Access your saved documents from the sidebar
- Load previous documents or delete them as needed


