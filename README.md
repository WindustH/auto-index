# PDF Auto Index

This project automatically adds a table of contents (bookmarks) to a PDF ebook using a Vision Language Model (VLM) to analyze the index pages.

## How it works

1.  **Finds Index Pages**: It iterates through the PDF to identify the pages that contain the table of contents.
2.  **Calculates Page Offset**: It determines the difference between the page numbers listed in the index and the actual page numbers in the PDF document. This is common in ebooks where front matter (like title pages and prefaces) isn't numbered.
3.  **Generates Bookmarks**: It uses a VLM to parse the index pages and convert them into a format compatible with the `pdftk` tool.
4.  **Applies Offset**: It adjusts the generated bookmark page numbers with the calculated offset.
5.  **Adds Bookmarks to PDF**: It uses `pdftk` to create a new PDF file with the bookmarks included.

## Requirements

-   Python 3.7+
-   `uv` (or `pip`) for package management
-   `pdftk`: This tool must be installed on your system. You can find installation instructions [here](https://www.pdflabs.com/tools/pdftk-the-pdf-toolkit/).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd pdf-auto-index
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

3.  **Configure your environment variables:**
    Create a `.env` file in the project root and add your API credentials:
    ```
    API_KEY="your_api_key"
    BASE_URL="your_base_url"
    VISION_MODEL="your_vision_model"
    ```

## Usage

Run the script from the command line, providing the input and output PDF paths:

```bash
python main.py /path/to/your/ebook.pdf /path/to/your/ebook_with_index.pdf
```

The script will print its progress to the console.
