# Language Translation Tool

![Cover](Multilingual.png)

## Overview

This project offers a structured approach to automating the translation of text across multiple document formats, including **DOCX**, **PPTX**, and **PDF**. It leverages modern language models to integrate rule-based and context-aware mechanisms, allowing for high-precision, consistent translations that adapt to the unique needs of the user.

By automating repetitive tasks, this tool shifts the role of a manual translator to that of a **quality assurance specialist**, enabling users to focus on refining outputs rather than starting from scratch. This approach significantly enhances both productivity and accuracy, making it possible to handle larger volumes of work with greater confidence.

At its core, the design is centered on two fundamental pillars:
1. **Customizable Translation Logic**: A system prompt defines rules such as “Do not translate personal names, internationally recognized technical terms, or trademarked terms,” ensuring that domain-specific or technical requirements are respected without manual intervention.
2. **Context-Aware Adaptation**: By considering the nuances of language, tone, and even dialects, the system creates translations that are more than direct mappings, reflecting the intent and meaning behind the original text.

### Key Components

1. **TranslationBackend**
   - **Role**: Processes and translates text from the input documents.
   - **Features**:
     - Structured to handle multi-format inputs while preserving document integrity (e.g., layouts, fonts, tables).
     - Implements advanced parsing to isolate translatable elements while leaving non-essential elements untouched (e.g., URLs, metadata).
     - Context adaptation through fine-tuned language models ensures dialect-appropriate outputs.

2. **TranslationUI**
   - **Role**: Provides a clean, interactive interface for managing uploads, tracking progress, and reviewing translated outputs.
   - **Features**:
     - Tracks translation stages to provide transparency into the process.
     - Supports real-time feedback and correction workflows, allowing users to refine results with minimal friction.
     - Designed for accessibility and scalability across use cases.

Here’s the full Usage section in markdown for you to copy directly into your README.md:

## Usage

Follow these steps to set up and run the application:

1. **Clone the Repository**  
   Clone the project to your local machine:
   ```bash
   git clone https://github.com/dneish2/LanguageTranslation.git
   cd LanguageTranslation

	2.	Install Dependencies
Use the requirements.txt file to install the necessary Python modules:

pip install -r requirements.txt

	3.	Set Up Environment Variables
Ensure you have an .env file in the project root with your OpenAI API key:

OPENAI_API_KEY=your-api-key-here

	4.	Run the Application
Start the application by running the main script:

python TranslationUI.py

	5.	Access the Interface
Once the server starts, open your browser and navigate to:

http://127.0.0.1:3030

This will load the NiceGUI-based user interface, where you can upload documents, select target languages, and manage translations.

### Prerequisites
- **Python Version**: Ensure Python 3.8 or higher is installed.
- **Dependencies**: Install all required packages listed in `requirements.txt`.
- **Environment Variables**: Add your OpenAI API key in a `.env` file located in the project root:
