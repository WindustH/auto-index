import argparse
import base64
import io
import os
import re
import subprocess
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass

import fitz  # PyMuPDF
from openai import OpenAI
from PIL import Image
from pypdf import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configuration
@dataclass
class Config:
    api_key: str = os.getenv("API_KEY", "")
    base_url: str = os.getenv("BASE_URL", "")
    vision_model: str = os.getenv("VISION_MODEL", "")
    temp_bookmark_file: str = "bookmarks.txt"


class PDFImageProcessor:
    """Handles PDF page to image conversion and image processing."""

    @staticmethod
    def extract_page_as_image(pdf_path: str, page_index: int) -> Image.Image:
        """Extract a specific page from PDF as an image."""
        doc = fitz.open(pdf_path)
        try:
            page = doc.load_page(page_index)
            # Use getattr to avoid Pylance error
            get_pixmap = getattr(page, "get_pixmap")
            pix = get_pixmap()
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            return image
        finally:
            doc.close()

    @staticmethod
    def convert_to_base64_webp(image: Image.Image) -> str:
        """Convert PIL image to base64-encoded WebP format."""
        buffer = io.BytesIO()
        image.save(buffer, format="webp")
        encoded_bytes = base64.b64encode(buffer.getvalue())
        return encoded_bytes.decode("utf-8")


class VisionLLMClient:
    """Handles all interactions with the vision language model."""

    def __init__(self, config: Config):
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model = config.vision_model

    def _send_vision_request(self, images: List[str], prompt: str) -> str:
        """Send a request to the vision LLM with images and prompt."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for base64_image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/webp;base64,{base64_image}",
                        "detail": "high",
                    },
                }
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],  # type: ignore
            stream=False,
        )
        return response.choices[0].message.content

    def is_index_page(self, page_image: Image.Image) -> bool:
        """Determine if a page is an index/table of contents page."""
        base64_image = PDFImageProcessor.convert_to_base64_webp(page_image)
        prompt = (
            "Is this an index page or table of contents? Answer with 'yes' or 'no'."
        )
        response = self._send_vision_request([base64_image], prompt)
        return "yes" in response.lower()

    def extract_first_index_entry(self, page_image: Image.Image) -> Tuple[int, str]:
        """Extract the first item from an index page and return its page number and text."""
        base64_image = PDFImageProcessor.convert_to_base64_webp(page_image)
        prompt = (
            "Find the first item in this index page and return its page number and "
            "the text of the item. Format: 'page_number, item_text'. "
            "Only return the page number and text, nothing else."
        )
        response = self._send_vision_request([base64_image], prompt)

        # Clean response of any formatting artifacts
        cleaned_response = self._clean_llm_response(response)

        try:
            page_number_str, item_text = cleaned_response.split(",", 1)
            return int(page_number_str.strip()), item_text.strip()
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"Failed to parse index entry from response: {cleaned_response}"
            ) from e

    def page_contains_content(self, page_image: Image.Image, target_text: str) -> bool:
        """Check if a page contains content matching the target text."""
        base64_image = PDFImageProcessor.convert_to_base64_webp(page_image)
        prompt = (
            f"Does this page contain content related to: '{target_text}'? "
            "Look for the actual chapter/section content, not just a reference. "
            "Answer with 'yes' or 'no'."
        )
        response = self._send_vision_request([base64_image], prompt)
        return "yes" in response.lower()

    def generate_pdftk_bookmarks(self, index_pages: List[Image.Image]) -> str:
        """Convert index pages to pdftk bookmark format."""
        base64_images = [
            PDFImageProcessor.convert_to_base64_webp(page) for page in index_pages
        ]

        prompt = """
Convert the following index pages into pdftk bookmark format.

Required format for each bookmark:
BookmarkBegin
BookmarkTitle: [Title of the bookmark]
BookmarkLevel: [1 for main chapters, 2 for sub-chapters, etc.]
BookmarkPageNumber: [Page number from the index]

Example:
BookmarkBegin
BookmarkTitle: Chapter 1: Introduction
BookmarkLevel: 1
BookmarkPageNumber: 5
BookmarkBegin
BookmarkTitle: 1.1 Overview
BookmarkLevel: 2
BookmarkPageNumber: 7

Convert ALL entries from the provided index pages into this exact format.
        """.strip()

        return self._send_vision_request(base64_images, prompt)

    @staticmethod
    def _clean_llm_response(response: str) -> str:
        """Remove common LLM response artifacts."""
        cleaned = response.strip()

        # Remove common formatting artifacts
        artifacts = ["<|begin_of_box|>", "<|end_of_box|>"]
        for artifact in artifacts:
            if cleaned.startswith(artifact):
                cleaned = cleaned[len(artifact) :]
            if cleaned.endswith(artifact):
                cleaned = cleaned[: -len(artifact)]

        return cleaned.strip()


class IndexPageDetector:
    """Handles detection and processing of index pages in a PDF."""

    def __init__(self, vision_client: VisionLLMClient):
        self.vision_client = vision_client

    def find_index_pages(self, pdf_path: str) -> Tuple[List[Image.Image], int]:
        """
        Find all consecutive index pages in a PDF.

        Returns:
            Tuple of (list of index page images, index of first non-index page)
        """
        print("Scanning for index pages...")

        pdf_reader = PdfReader(pdf_path)
        index_pages = []
        first_index_found = False
        content_start_index = 0

        for page_index in range(len(pdf_reader.pages)):
            page_image = PDFImageProcessor.extract_page_as_image(pdf_path, page_index)

            if self.vision_client.is_index_page(page_image):
                first_index_found = True
                index_pages.append(page_image)
                print(f"Found index page at position {page_index + 1}")
            elif first_index_found:
                # Found first non-index page after index pages
                content_start_index = page_index
                break

        return index_pages, content_start_index


class BookmarkGenerator:
    """Handles bookmark generation and PDF modification."""

    def __init__(self, vision_client: VisionLLMClient, config: Config):
        self.vision_client = vision_client
        self.config = config

    def calculate_page_offset(
        self, pdf_path: str, index_pages: List[Image.Image], content_start_index: int
    ) -> int:
        """Calculate the offset between index page numbers and actual page numbers."""
        if not index_pages:
            raise ValueError("No index pages provided")

        print("Calculating page number offset...")

        # Get first entry from index
        first_page_num, first_item_text = self.vision_client.extract_first_index_entry(
            index_pages[0]
        )
        print(f"First index entry: '{first_item_text}' (page {first_page_num})")

        # Find actual page containing this content
        pdf_reader = PdfReader(pdf_path)
        actual_page_num = self._find_content_page(
            pdf_path, first_item_text, content_start_index, len(pdf_reader.pages)
        )

        if actual_page_num == 0:
            raise RuntimeError(
                f"Could not locate content for '{first_item_text}' in the PDF"
            )

        offset = actual_page_num - first_page_num
        print(f"Page offset calculated: {offset}")
        return offset

    def _find_content_page(
        self, pdf_path: str, target_text: str, start_index: int, total_pages: int
    ) -> int:
        """Find the page containing specific content."""
        print(f"Searching for content: '{target_text}'...")

        for page_index in range(start_index, total_pages):
            page_image = PDFImageProcessor.extract_page_as_image(pdf_path, page_index)

            if self.vision_client.page_contains_content(page_image, target_text):
                return page_index + 1  # Convert to 1-based page numbering

        return 0  # Not found

    @staticmethod
    def apply_page_offset(bookmark_text: str, offset: int) -> str:
        """Apply page number offset to bookmark text."""

        def replace_page_number(match):
            original_page = int(match.group(1))
            new_page = original_page + offset
            return f"BookmarkPageNumber: {new_page}"

        return re.sub(r"BookmarkPageNumber: (\d+)", replace_page_number, bookmark_text)

    def add_bookmarks_to_pdf(
        self, bookmark_text: str, input_path: str, output_path: str
    ):
        """Add bookmarks to PDF using pdftk."""
        self._verify_pdftk_installation()

        print("Adding bookmarks to PDF...")

        # Write bookmarks to temporary file
        with open(self.config.temp_bookmark_file, "w", encoding="utf-8") as f:
            f.write(bookmark_text)

        try:
            # Use pdftk to add bookmarks
            subprocess.run(
                [
                    "pdftk",
                    input_path,
                    "update_info",
                    self.config.temp_bookmark_file,
                    "output",
                    output_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            print(f"Successfully created PDF with bookmarks: {output_path}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"pdftk failed: {e.stderr}") from e

        finally:
            # Clean up temporary file
            if os.path.exists(self.config.temp_bookmark_file):
                os.remove(self.config.temp_bookmark_file)

    @staticmethod
    def _verify_pdftk_installation():
        """Verify that pdftk is installed and accessible."""
        try:
            subprocess.run(
                ["pdftk", "--version"], check=True, capture_output=True, text=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "pdftk is not installed or not accessible. "
                "Please install pdftk to use this tool."
            ) from e


class PDFBookmarkProcessor:
    """Main class orchestrating the PDF bookmark addition process."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.vision_client = VisionLLMClient(self.config)
        self.index_detector = IndexPageDetector(self.vision_client)
        self.bookmark_generator = BookmarkGenerator(self.vision_client, self.config)

    def process_pdf(self, input_path: str, output_path: str):
        """Process a PDF to add bookmarks based on its index pages."""
        try:
            # Step 1: Find index pages
            index_pages, content_start_index = self.index_detector.find_index_pages(
                input_path
            )

            if not index_pages:
                print("No index pages found in the PDF.")
                return False

            print(f"Found {len(index_pages)} index page(s)")

            # Step 2: Calculate page offset
            offset = self.bookmark_generator.calculate_page_offset(
                input_path, index_pages, content_start_index
            )

            # Step 3: Generate bookmarks
            print("Generating bookmark structure...")
            bookmark_text = self.vision_client.generate_pdftk_bookmarks(index_pages)

            # Step 4: Apply offset
            bookmark_text = self.bookmark_generator.apply_page_offset(
                bookmark_text, offset
            )

            # Step 5: Add bookmarks to PDF
            self.bookmark_generator.add_bookmarks_to_pdf(
                bookmark_text, input_path, output_path
            )

            print("PDF bookmark processing completed successfully!")
            return True

        except Exception as e:
            print(f"Error processing PDF: {e}")
            return False


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Add bookmarks to a PDF ebook based on its index pages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.pdf output.pdf
  python main.py /path/to/ebook.pdf /path/to/ebook_with_bookmarks.pdf

Requirements:
  - pdftk must be installed and accessible in PATH
  - Environment variables must be set: SILICON_API_KEY, SILICON_BASE_URL, SILICON_VISION_MODEL
        """,
    )

    parser.add_argument("input_path", type=str, help="Path to the input PDF file")

    parser.add_argument(
        "output_path",
        type=str,
        help="Path where the output PDF with bookmarks will be saved",
    )

    return parser


def main():
    """Main entry point of the application."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' does not exist.")
        return 1

    # Validate output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return 1

    # Process the PDF
    processor = PDFBookmarkProcessor()
    success = processor.process_pdf(args.input_path, args.output_path)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
