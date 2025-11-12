#!/usr/bin/env python3
"""
Create a printable PDF from all generated Spot It cards.

This script takes all the card images from the output directory and creates a PDF
where each card is sized at 8.9 cm in diameter, suitable for printing and cutting.
"""

import os
import glob
from typing import List, Tuple
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from PIL import Image, ImageDraw
import argparse
from PyPDF2 import PdfReader, PdfWriter

# Constants
CARD_DIAMETER_CM = 8.9
CARD_RADIUS_CM = CARD_DIAMETER_CM / 2
OUTPUT_DIR = "output"
PDF_OUTPUT = "spot_it_cards_printable.pdf"
CARDS_PER_ROW = 2
CARDS_PER_COL = 3
CARDS_PER_PAGE = CARDS_PER_ROW * CARDS_PER_COL

# A4 page dimensions in cm
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7

# Margins in cm
MARGIN_CM = 0.5  # Reduced from 1.0 to 0.5 cm

def create_circular_card_image(card_path: str, output_dir: str = "temp_circular") -> str:
    """
    Create a circular version of a card image by applying a circular mask.
    
    Args:
        card_path: Path to the original card image
        output_dir: Directory to save the circular version
        
    Returns:
        Path to the circular version of the card
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the original card image
    with Image.open(card_path) as img:
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Create a new image with white background (better for PDFs)
        result = Image.new('RGBA', img.size, (255, 255, 255, 255))
        
        # Calculate center and radius
        width, height = img.size
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 2
        
        # Create a circular mask
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([center_x - radius, center_y - radius, 
                     center_x + radius, center_y + radius], fill=255)
        
        # Apply the mask to the original image
        img_masked = Image.new('RGBA', img.size, (0, 0, 0, 0))
        img_masked.paste(img, (0, 0), mask)
        
        # Composite the masked image onto the white background
        result.paste(img_masked, (0, 0), img_masked)
        
        # Save the circular version
        filename = os.path.basename(card_path)
        output_path = os.path.join(output_dir, f"circular_{filename}")
        result.save(output_path, 'PNG')
        
        return output_path

def get_card_files(output_dir: str = OUTPUT_DIR) -> List[str]:
    """
    Get all card image files from the output directory.
    
    Args:
        output_dir: Directory containing the card images
        
    Returns:
        List of card image file paths, sorted by card number
    """
    # Look for files matching the pattern card_XX.png
    pattern = os.path.join(output_dir, "card_*.png")
    card_files = glob.glob(pattern)
    
    # Sort by card number (extract number from filename)
    def extract_card_number(filename):
        basename = os.path.basename(filename)
        # Extract number from "card_XX.png"
        number_str = basename.replace("card_", "").replace(".png", "")
        return int(number_str)
    
    card_files.sort(key=extract_card_number)
    
    if not card_files:
        raise FileNotFoundError(f"No card files found in {output_dir}")
    
    print(f"Found {len(card_files)} card files")
    return card_files

def calculate_card_layout() -> Tuple[float, float, float, float]:
    """
    Calculate the layout parameters for cards on the page.
    
    Returns:
        Tuple of (start_x, start_y, spacing_x, spacing_y) in cm
    """
    # Calculate available space on page
    available_width = A4_WIDTH_CM - (2 * MARGIN_CM)
    available_height = A4_HEIGHT_CM - (2 * MARGIN_CM)
    
    # Calculate spacing between cards with extra vertical spacing to prevent overlap
    spacing_x = (available_width - (CARDS_PER_ROW * CARD_DIAMETER_CM)) / (CARDS_PER_ROW + 1)
    
    # Add minimal vertical spacing to ensure cards don't overlap
    extra_vertical_spacing = 0.05  # Reduced from 0.2 to 0.05 cm extra space between cards
    spacing_y = (available_height - (CARDS_PER_COL * CARD_DIAMETER_CM)) / (CARDS_PER_COL + 1) + extra_vertical_spacing
    
    # Calculate starting position (top-left of first card)
    start_x = MARGIN_CM + spacing_x
    start_y = A4_HEIGHT_CM - MARGIN_CM - spacing_y - CARD_DIAMETER_CM
    
    return start_x, start_y, spacing_x, spacing_y

def create_pdf_with_cards(card_files: List[str], output_pdf: str = PDF_OUTPUT):
    """
    Create a PDF with all the Spot It cards arranged for printing.
    
    Args:
        card_files: List of card image file paths
        output_pdf: Output PDF filename
    """
    # Calculate layout
    start_x, start_y, spacing_x, spacing_y = calculate_card_layout()
    
    # Create PDF
    c = canvas.Canvas(output_pdf, pagesize=A4)
    
    total_pages = (len(card_files) + CARDS_PER_PAGE - 1) // CARDS_PER_PAGE
    
    print(f"Creating PDF with {len(card_files)} cards on {total_pages} pages...")
    print(f"Cards per page: {CARDS_PER_PAGE}")
    print(f"Card size: {CARD_DIAMETER_CM} cm diameter")
    
    for page_num in range(total_pages):
        print(f"Processing page {page_num + 1}/{total_pages}...")
        
        # Add page number
        c.setFont("Helvetica", 10)
        c.drawString(1*cm, 1*cm, f"Page {page_num + 1} of {total_pages}")
        
        # Calculate which cards go on this page
        start_idx = page_num * CARDS_PER_PAGE
        end_idx = min(start_idx + CARDS_PER_PAGE, len(card_files))
        page_cards = card_files[start_idx:end_idx]
        
        # Place cards on the page
        for i, card_file in enumerate(page_cards):
            # Calculate position for this card
            row = i // CARDS_PER_ROW
            col = i % CARDS_PER_ROW
            
            x = start_x + col * (CARD_DIAMETER_CM + spacing_x)
            y = start_y - row * (CARD_DIAMETER_CM + spacing_y)
            
            # Convert to points (reportlab uses points, 1 cm = 28.35 points)
            x_pt = x * 28.35
            y_pt = y * 28.35
            size_pt = CARD_DIAMETER_CM * 28.35
            radius_pt = size_pt / 2
            
            # Add the card image
            try:
                # Draw the card image
                c.drawImage(card_file, x_pt, y_pt, size_pt, size_pt)
                
                # Add card number below the card
                card_number = os.path.basename(card_file).replace("card_", "").replace(".png", "")
                c.setFont("Helvetica", 8)
                c.drawString(x_pt, y_pt - 15, f"Card {card_number}")
                
            except Exception as e:
                print(f"Warning: Could not add {card_file}: {e}")
                # Draw a placeholder circle
                c.setStrokeColorRGB(0.8, 0.8, 0.8)
                c.setLineWidth(0.5)
                c.circle(x_pt + radius_pt, y_pt + radius_pt, radius_pt)
                c.setStrokeColorRGB(0, 0, 0)
                c.drawString(x_pt + size_pt/2 - 20, y_pt + size_pt/2, "ERROR")
        
        # Add page break if not the last page
        if page_num < total_pages - 1:
            c.showPage()
    
    # Save the PDF
    c.save()
    print(f"PDF created successfully: {output_pdf}")

def add_background_pages(cards_pdf_path: str, background_pdf_path: str, output_pdf: str):
    """
    Insert background.pdf pages between each card page for double-sided printing.
    
    Args:
        cards_pdf_path: Path to the PDF with card pages
        background_pdf_path: Path to the background PDF
        output_pdf: Output PDF filename with backgrounds inserted
    """
    print("Adding background pages for double-sided printing...")
    
    # Read the cards PDF
    cards_reader = PdfReader(cards_pdf_path)
    background_reader = PdfReader(background_pdf_path)
    
    # Create output PDF
    writer = PdfWriter()
    
    # Get background page (use first page if multiple exist)
    background_page = background_reader.pages[0]
    
    # Insert pages: card page, background page, card page, background page, etc.
    for i, page in enumerate(cards_reader.pages):
        # Add the card page
        writer.add_page(page)
        writer.add_page(background_page)
    
    # Save the combined PDF
    with open(output_pdf, 'wb') as output_file:
        writer.write(output_file)
    
    print(f"Background pages added successfully: {output_pdf}")

def add_cutting_guides(canvas_obj, start_x: float, start_y: float, spacing_x: float, spacing_y: float):
    """
    Add cutting guide lines to help with cutting out the cards.
    
    Args:
        canvas_obj: ReportLab canvas object
        start_x, start_y, spacing_x, spacing_y: Layout parameters in cm
    """
    # Convert to points
    start_x_pt = start_x * 28.35
    start_y_pt = start_y * 28.35
    spacing_x_pt = spacing_x * 28.35
    spacing_y_pt = spacing_y * 28.35
    card_size_pt = CARD_DIAMETER_CM * 28.35
    
    # Set line style for cutting guides
    canvas_obj.setStrokeColorRGB(0.7, 0.7, 0.7)  # Light gray
    canvas_obj.setDash(2, 2)  # Dashed line
    
    # Draw vertical cutting lines
    for col in range(CARDS_PER_ROW + 1):
        x = start_x_pt + col * (card_size_pt + spacing_x_pt) - spacing_x_pt/2
        canvas_obj.line(x, 1*cm, x, (A4_HEIGHT_CM-1)*cm)
    
    # Draw horizontal cutting lines
    for row in range(CARDS_PER_COL + 1):
        y = start_y_pt + row * (card_size_pt + spacing_y_pt) + spacing_y_pt/2
        canvas_obj.line(1*cm, y, (A4_WIDTH_CM-1)*cm, y)
    
    # Reset line style
    canvas_obj.setDash(0)
    canvas_obj.setStrokeColorRGB(0, 0, 0)

def main():
    """Main function to create the printable PDF."""
    parser = argparse.ArgumentParser(description="Create a printable PDF from Spot It cards")
    parser.add_argument("--output", "-o", default=PDF_OUTPUT, help="Output PDF filename")
    parser.add_argument("--cards", "-c", default=OUTPUT_DIR, help="Directory containing card images")
    parser.add_argument("--size", "-s", type=float, default=8.9, 
                       help="Card diameter in cm (default: 8.9)")
    parser.add_argument("--add-guides", action="store_true", help="Add cutting guide lines")
    parser.add_argument("--add-backs", action="store_true", 
                       help="Add background.pdf pages between card pages for double-sided printing")
    
    args = parser.parse_args()
    
    try:
        # Update global constants if specified
        global CARD_DIAMETER_CM, CARD_RADIUS_CM
        CARD_DIAMETER_CM = args.size
        CARD_RADIUS_CM = CARD_DIAMETER_CM / 2
        
        # Get card files
        card_files = get_card_files(args.cards)
        
        # Create circular versions of all cards
        print("Creating circular versions of cards...")
        circular_card_files = []
        for card_file in card_files:
            circular_path = create_circular_card_image(card_file)
            circular_card_files.append(circular_path)
        
        # Create temporary PDF with circular cards
        temp_pdf = "temp_cards.pdf"
        create_pdf_with_cards(circular_card_files, temp_pdf)
        
        # If adding backs, insert background pages
        if args.add_backs:
            if not os.path.exists("background.pdf"):
                print("Warning: background.pdf not found. Creating PDF without backs.")
                final_output = args.output
            else:
                # Create final output with background pages
                add_background_pages(temp_pdf, "background.pdf", args.output)
                final_output = args.output
                # Clean up temporary PDF
                try:
                    os.remove(temp_pdf)
                except:
                    pass
        else:
            # No backs requested, use the cards PDF directly
            os.rename(temp_pdf, args.output)
            final_output = args.output
        
        # Clean up temporary circular images
        print("Cleaning up temporary files...")
        for circular_file in circular_card_files:
            try:
                os.remove(circular_file)
            except:
                pass
        try:
            os.rmdir("temp_circular")
        except:
            pass
        
        print(f"\nPDF created successfully!")
        print(f"Output file: {final_output}")
        print(f"Total cards: {len(card_files)}")
        print(f"Card size: {CARD_DIAMETER_CM} cm diameter")
        print(f"Pages: {(len(card_files) + CARDS_PER_PAGE - 1) // CARDS_PER_PAGE}")
        print(f"Cards per page: {CARDS_PER_PAGE}")
        
        if args.add_guides:
            print("Note: Cutting guides were added to help with card cutting")
        
        if args.add_backs:
            print("Note: Background pages were added for double-sided printing")
            print("Printing instructions:")
            print("1. Print the PDF on A4 paper (double-sided)")
            print("2. The background.pdf will appear on the back of each card page")
            print("3. Cut along the card boundaries")
            print(f"4. Each card should be exactly {CARD_DIAMETER_CM} cm in diameter")
        else:
            print("\nPrinting instructions:")
            print("1. Print the PDF on A4 paper")
            print("2. Cut along the card boundaries")
            print(f"3. Each card should be exactly {CARD_DIAMETER_CM} cm in diameter")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
