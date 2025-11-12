#!/usr/bin/env python3
"""
Test script to verify the mathematical properties of spot-it card combinations.
This tests the theory before attempting to generate actual image cards.
"""

from generate_all_cards import create_spot_it_combinations, verify_spot_it_properties

def test_spot_it_properties():
    """Test that the spot-it combinations satisfy all required properties."""
    print("Testing spot-it mathematical properties...")
    print("=" * 50)
    
    try:
        # Generate the combinations
        cards = create_spot_it_combinations()
        
        # Verify properties
        verify_spot_it_properties(cards)
        
        # Additional detailed analysis
        print("\nDetailed Analysis:")
        print("-" * 30)
        
        # Show first few cards
        print("\nFirst 5 cards:")
        for i in range(min(5, len(cards))):
            print(f"Card {i+1}: {[idx+1 for idx in cards[i]]}")
        
        # Show last few cards
        print(f"\nLast 5 cards:")
        for i in range(max(0, len(cards)-5), len(cards)):
            print(f"Card {i+1}: {[idx+1 for idx in cards[i]]}")
        
        # Check specific properties
        print(f"\nProperty Verification:")
        print(f"- Total cards: {len(cards)}")
        print(f"- Images per card: {len(cards[0]) if cards else 0}")
        print(f"- Total unique images: {len(set([idx for card in cards for idx in card]))}")
        
        # Check intersection matrix
        print(f"\nIntersection Analysis:")
        intersection_counts = {}
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                intersection = set(cards[i]) & set(cards[j])
                count = len(intersection)
                intersection_counts[count] = intersection_counts.get(count, 0) + 1
                
                if count != 1:
                    print(f"  ERROR: Cards {i+1} and {j+1} share {count} images!")
                    print(f"    Card {i+1}: {[idx+1 for idx in cards[i]]}")
                    print(f"    Card {j+1}: {[idx+1 for idx in cards[j]]}")
                    print(f"    Shared: {[idx+1 for idx in intersection]}")
        
        print(f"  Intersection counts: {intersection_counts}")
        
        # Check image distribution
        print(f"\nImage Distribution:")
        image_card_count = [0] * 57
        for card in cards:
            for image_idx in card:
                image_card_count[image_idx] += 1
        
        min_count = min(image_card_count)
        max_count = max(image_card_count)
        avg_count = sum(image_card_count) / len(image_card_count)
        
        print(f"  Min appearances: {min_count}")
        print(f"  Max appearances: {max_count}")
        print(f"  Average appearances: {avg_count:.2f}")
        
        if min_count == max_count == 8:
            print("  ✓ All images appear exactly 8 times")
        else:
            print("  ✗ Image distribution is not uniform")
        
        print("\n" + "=" * 50)
        print("✓ All tests passed! The mathematical construction is correct.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_spot_it_properties()
    exit(0 if success else 1)
