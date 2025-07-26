# Language Translation Tool

![Cover](Multilingual.png)

## Overview

This tool makes it easy to translate full documents like DOCX, PDF, and PPTX while keeping the original formatting and structure intact. It uses a flexible logic layer to understand what should be translated and how, working alongside language models to produce clean, editable outputs ready for use or refinement.

By automating repetitive tasks, this tool shifts the role of a manual translator to that of a **quality assurance specialist**, enabling **power users** to focus on refining outputs rather than starting from scratch. This approach significantly enhances both productivity and accuracy, making it possible to handle larger volumes of work with greater confidence.

At its core, the design is centered on two principles:
1. **Customizable Translation Logic**: A customizable system prompt governs how the tool handles names, technical terms, and domain-specific phrases. This flexibility ensures accurate and context-aware output to match nuanced need. 
2. **Human-Centered Document Interaction**: Instead of managing files through code or digging through folders, documents are surfaced directly in the interface. You can preview, translate, and export with just a few clicks.

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
   ```

2. **Install Dependencies**
   Use the requirements.txt file to install the necessary Python modules:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**
   Ensure you have an .env file in the project root with your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Run the Application**
   Start the application by running the main script:
   ```bash
   python TranslationUI.py
   ```

5. **Access the Interface**
   Once the server starts, open your browser and navigate to:
   http://127.0.0.1:8080
This will load the NiceGUI-based user interface, where you can upload documents, select target languages, and manage translations.

### Prerequisites
- **Python Version**: Ensure Python 3.8 or higher is installed.
- **Dependencies**: Install all required packages listed in `requirements.txt`.
- **Environment Variables**: Add your OpenAI API key in a `.env` file located in the project root:
