import os
import random
import math
from typing import List, Set, Tuple
from generate_card import generate_card

# =========================
# Constants Section
# =========================

# Spot-it game constants
TOTAL_IMAGES = 57
IMAGES_PER_CARD = 8
CARDS_NEEDED = 57

# Output settings
OUTPUT_DIR = "output"
CARD_PREFIX = "card_"

# =========================
# End of Constants Section
# =========================

def create_spot_it_combinations() -> List[List[int]]:
    """
    Create combinations of 8 images for each of 57 cards such that
    any two cards share exactly one common image.
    
    This uses a projective plane construction where:
    - Each card represents a line in a finite projective plane
    - Each image represents a point
    - Two lines intersect at exactly one point
    - Each line contains exactly 8 points
    - Each point is on exactly 8 lines
    
    Returns:
        List of 57 lists, each containing 8 image indices (0-56)
    """
    
    # For a projective plane of order 7 (7² + 7 + 1 = 57 points)
    # We can construct this using finite field arithmetic
    
    def finite_field_mult(a: int, b: int, p: int = 7) -> int:
        """Multiply two numbers in GF(7)"""
        return (a * b) % p
    
    def finite_field_add(a: int, b: int, p: int = 7) -> int:
        """Add two numbers in GF(7)"""
        return (a + b) % p
    
    def finite_field_inv(a: int, p: int = 7) -> int:
        """Find multiplicative inverse in GF(7)"""
        for i in range(1, p):
            if (a * i) % p == 1:
                return i
        return 1
    
    cards = []
    
    # Method 1: Lines through origin (affine lines)
    # y = mx + b for different m and b values
    for m in range(7):
        for b in range(7):
            card = []
            for x in range(7):
                y = finite_field_add(finite_field_mult(m, x), b)
                point = x + 7 * y
                card.append(point)
            # Add the point at infinity for this line
            card.append(49 + m)  # 49-55 for slope points at infinity
            cards.append(card)
    
    # Method 2: Vertical lines (x = constant)
    for x in range(7):
        card = []
        for y in range(7):
            point = x + 7 * y
            card.append(point)
        # Add the vertical point at infinity
        card.append(56)  # Point at infinity for vertical lines
        cards.append(card)
    
    # Method 3: The line at infinity (all slope points)
    infinity_line = list(range(49, 57))
    cards.append(infinity_line)
    
    # Verify the construction
    if len(cards) != CARDS_NEEDED:
        raise ValueError(f"Generated {len(cards)} cards, expected {CARDS_NEEDED}")
    
    # Verify each card has exactly 8 images
    for i, card in enumerate(cards):
        if len(card) != IMAGES_PER_CARD:
            raise ValueError(f"Card {i} has {len(card)} images, expected {IMAGES_PER_CARD}")
        if len(set(card)) != IMAGES_PER_CARD:
            raise ValueError(f"Card {i} has duplicate images")
    
    # Verify any two cards share exactly one image
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            intersection = set(cards[i]) & set(cards[j])
            if len(intersection) != 1:
                raise ValueError(f"Cards {i} and {j} share {len(intersection)} images, expected 1")
    
    return cards

def verify_spot_it_properties(cards: List[List[int]]) -> None:
    """
    Verify that the generated cards satisfy spot-it game properties.
    
    Args:
        cards: List of cards, each containing 8 image indices
    """
    print("Verifying spot-it properties...")
    
    # Check each card has exactly 8 images
    for i, card in enumerate(cards):
        if len(card) != IMAGES_PER_CARD:
            raise ValueError(f"Card {i} has {len(card)} images, expected {IMAGES_PER_CARD}")
        if len(set(card)) != IMAGES_PER_CARD:
            raise ValueError(f"Card {i} has duplicate images")
    
    # Check any two cards share exactly one image
    shared_counts = []
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            intersection = set(cards[i]) & set(cards[j])
            shared_counts.append(len(intersection))
            if len(intersection) != 1:
                raise ValueError(f"Cards {i} and {j} share {len(intersection)} images, expected 1")
    
    # Check each image appears on exactly 8 cards
    image_card_count = [0] * TOTAL_IMAGES
    for card in cards:
        for image_idx in card:
            image_card_count[image_idx] += 1
    
    for i, count in enumerate(image_card_count):
        if count != IMAGES_PER_CARD:
            raise ValueError(f"Image {i} appears on {count} cards, expected {IMAGES_PER_CARD}")
    
    print(f"✓ Generated {len(cards)} cards")
    print(f"✓ Each card has exactly {IMAGES_PER_CARD} images")
    print(f"✓ Any two cards share exactly 1 image")
    print(f"✓ Each image appears on exactly {IMAGES_PER_CARD} cards")
    print(f"✓ Total image intersections: {len(shared_counts)}")
    print(f"✓ All intersections have exactly 1 shared image: {all(count == 1 for count in shared_counts)}")

def get_image_paths_for_card(card_indices: List[int], image_dir: str = "images") -> List[str]:
    """
    Get the file paths for the images in a card.
    
    Args:
        card_indices: List of image indices (0-56)
        image_dir: Directory containing the numbered images
        
    Returns:
        List of image file paths
    """
    image_paths = []
    for idx in card_indices:
        # Images are named 1.png to 57.png, but indices are 0-56
        image_name = f"{idx + 1}.png"
        image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image_paths.append(image_path)
    
    return image_paths

def generate_all_cards(
    output_dir: str = OUTPUT_DIR,
    image_dir: str = "images",
    canvas_size: int = 1200,
    background_color: str = "#FFFFFF",
    seed: int = None,
    verbose: bool = False
) -> List[str]:
    """
    Generate all 57 spot-it cards.
    
    Args:
        output_dir: Directory to save generated cards
        image_dir: Directory containing numbered images (1.png to 57.png)
        canvas_size: Size of the canvas for each card
        background_color: Background color for cards
        seed: Random seed for reproducibility
        verbose: Whether to print progress information
        
    Returns:
        List of paths to generated card files
    """
    
    if seed is not None:
        random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify image directory exists and has required images
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Check for required images
    required_images = [f"{i}.png" for i in range(1, TOTAL_IMAGES + 1)]
    missing_images = []
    for img_name in required_images:
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            missing_images.append(img_name)
    
    if missing_images:
        raise FileNotFoundError(f"Missing required images: {missing_images}")
    
    print(f"Found all {TOTAL_IMAGES} required images in {image_dir}")
    
    # Generate the spot-it combinations
    print("Generating spot-it card combinations...")
    cards = create_spot_it_combinations()
    
    # Verify the properties
    verify_spot_it_properties(cards)
    
    # Generate each card
    print(f"\nGenerating {len(cards)} cards...")
    generated_paths = []
    
    for i, card_indices in enumerate(cards):
        if verbose:
            print(f"Generating card {i+1}/{len(cards)} with images: {[idx+1 for idx in card_indices]}")
        else:
            print(f"Generating card {i+1}/{len(cards)}...")
        
        # Get image paths for this card
        image_paths = get_image_paths_for_card(card_indices, image_dir)
        
        # Generate output path
        output_filename = f"{CARD_PREFIX}{i+1:02d}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # Generate the card using the existing algorithm
            generated_path = generate_card(
                image_paths=image_paths,
                output_path=output_path,
                canvas_size=canvas_size,
                background_color=background_color,
                seed=seed + i if seed is not None else None,
                verbose=False  # Keep individual card generation quiet
            )
            generated_paths.append(generated_path)
            
        except Exception as e:
            print(f"Error generating card {i+1}: {e}")
            raise
    
    print(f"\nSuccessfully generated {len(generated_paths)} cards in {output_dir}")
    return generated_paths

def main():
    """Main function to generate all spot-it cards."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate all 57 spot-it cards")
    parser.add_argument("--output", "-o", default=OUTPUT_DIR, help="Output directory for generated cards")
    parser.add_argument("--images", "-i", default="images", help="Directory containing numbered images (1.png to 57.png)")
    parser.add_argument("--size", "-s", type=int, default=1200, help="Canvas size in pixels (square)")
    parser.add_argument("--bg", default="#FFFFFF", help="Card background color (hex)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        generated_paths = generate_all_cards(
            output_dir=args.output,
            image_dir=args.images,
            canvas_size=args.size,
            background_color=args.bg,
            seed=args.seed,
            verbose=args.verbose
        )
        
        print(f"\nAll cards generated successfully!")
        print(f"Output directory: {args.output}")
        print(f"Total cards: {len(generated_paths)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
